import numpy as np
import pickle
import torch
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn import manifold
from sklearn.metrics.cluster import adjusted_rand_score
import copy
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 30000
import matplotlib.pyplot as plt


# helper function to save image to file for debugging
def temp_save_img(array, filename):
    filename = f"/home/jday/scene_understanding/Z_debugging_images/{filename}.png"
    plt.imsave(filename, array)

# Calculates ARI of each mask-groundtruth image-pair in the batch it is given
# Expects input of shape: [batch_size x num_slots x height x width] Example: [20, 4, 128, 128]
# if 'also_exclude_outliers' is True, will return both versions of the metric (ie, w and wout outliers) 
def calculate_ARI(mask_inferred, mask_gt, fg_only=True, bg_id=3, also_exclude_outliers=False, log=False):
    # Take argmax over object softmax channels to get a single channel where each pixel is an
    # indication for which object/background it represents
    labels_inferred = torch.flatten(torch.argmax(mask_inferred, 1), 1, 2).cpu().numpy()
    labels_gt = torch.flatten(torch.argmax(mask_gt, 1), 1, 2).cpu().numpy()
    # labels_inferred and labels_gt resulting expected shape: (batch_size x height*width) ie, (20, 16384)
        
    ARI_batch_list = []
    ARI_batch_list_wout_outilers = []
    outliers_indices = []
    for i in range(labels_gt.shape[0]): # for each item in batch
        if fg_only:
            idx = labels_gt[i] != bg_id
            ari = adjusted_rand_score(labels_gt[i][idx], labels_inferred[i][idx])
            if also_exclude_outliers and len(np.unique(labels_gt[i][idx])) == 1:
                if log: print("\toutlier detected. ARI value:", ari)
                outliers_indices.append(i)
            ARI_batch_list.append(ari)
        else:
            ari = adjusted_rand_score(labels_gt[i,:], labels_inferred[i,:])
            ARI_batch_list.append(ari)
        if log: print(f"\tARI at idx {i}:", ari)
    ARIs_batch = np.array(ARI_batch_list)

    # if also want results excluding outliers, return both
    if also_exclude_outliers:
        ARI_batch_list_wout_outilers = copy.deepcopy(ARI_batch_list)
        for j in reversed(outliers_indices): # 'reversed' bc indices relative locations aren't changed when deleting elements from the end
            ARI_batch_list_wout_outilers.pop(j)
        ARIs_batch_wout_outilers = np.array(ARI_batch_list_wout_outilers)
        return [ARIs_batch, ARIs_batch_wout_outilers]
    
    return ARIs_batch

# non-batch version of Genesis one
def iou_binary_custom(mask_A, mask_B):
    assert mask_A.shape == mask_B.shape
    assert mask_A.dtype == torch.bool
    assert mask_B.dtype == torch.bool
    intersection = (mask_A * mask_B).sum()
    union = (mask_A + mask_B).sum()
    # Return -100 if union is zero, else return IOU
    return torch.where(union == 0, torch.tensor(-100.0),
                       intersection.float() / union.float())

# copied from Genesis pytorch github repo https://github.com/applied-ai-lab/genesis/blob/master/utils/misc.py 
def iou_binary(mask_A, mask_B):
    assert mask_A.shape == mask_B.shape
    assert mask_A.dtype == torch.bool
    assert mask_B.dtype == torch.bool
    intersection = (mask_A * mask_B).sum((1, 2, 3))
    union = (mask_A + mask_B).sum((1, 2, 3))
    # Return -100 if union is zero, else return IOU
    return torch.where(union == 0, torch.tensor(-100.0),
                       intersection.float() / union.float())

# based on, but editted, from Genesis pytorch github repo https://github.com/applied-ai-lab/genesis/blob/master/utils/misc.py 
def average_segcover_custom(segA, segB, ignore_background=False, background_label=3, ignore_null_case=True):
    """
    Covering of segA by segB
    - segA.shape = [img_dim1, img_dim2]
    - segB.shape = [img_dim1, img_dim2]
    - Assumes labels in segA and segB are non-negative integers.
    Negative labels will be ignored.
    - Assumes segA is groundtruth labels, and as such, will use
    labels in segA as basis for IoU comparisons (so, if only
    foreground SC is desired, remove background prior to this
    function call)
    - Expects torch tensors as input
    - Returns torch tensors
    - If ignore_null_case is True:
        - when segA (GT) is only background, will return -1 for each score
    - If ignore_null_case is False:
        - Will produce same behavior as Genesis version (return scores of 0)
    """

    # ensure same shape
    assert segA.shape == segB.shape, f"{segA.shape} - {segB.shape}"
    # ensure no negative elements
    assert (segA < 0).count_nonzero() == 0 and (segA < 0).count_nonzero() == 0

    scores_summed = torch.tensor([0.0])
    scaled_scores = torch.tensor([0.0])
    scaling_sum = torch.tensor([0])

    if ignore_background:
        iter_segA = torch.unique(segA[segA != background_label]).tolist()
    else: 
        iter_segA = torch.unique(segA).tolist()
    iter_segB = torch.unique(segB).tolist()
    num_objects = len(iter_segA)
    
    if num_objects == 0:
        if ignore_null_case: return -1, -1
        # This line below shoudl produce same behavior as Genesis version
        else: return torch.tensor([0.0]), torch.tensor([0.0])

    # Loop over segA (groundtruth)
    for i in iter_segA:
        binaryA = (segA == i)
        max_iou = torch.tensor([0.0])
        # Loop over segB to find max IOU
        for j in iter_segB:
            binaryB = (segB == j)
            iou = iou_binary_custom(binaryA, binaryB)
            if iou > max_iou: max_iou = iou
        # Accumulate scores
        scores_summed += max_iou
        scaled_scores += binaryA.sum().float() * max_iou
        scaling_sum += binaryA.sum()

    # # Compute coverage
    mean_sc = scores_summed / num_objects
    scaled_sc = scaled_scores / scaling_sum

    # Sanity check
    assert (mean_sc >= 0) and (mean_sc <= 1), mean_sc
    assert (scaled_sc >= 0) and (scaled_sc <= 1), scaled_sc
    # assert (scores_summed[N == 0] == 0).all()
    # # assert (scores_summed[nonignore.sum((1, 2, 3)) == 0] == 0).all()
    # assert (scaled_scores[N == 0] == 0).all()
    # # assert (scaled_scores[nonignore.sum((1, 2, 3)) == 0] == 0).all()

    # Return mean over batch dimension 
    return mean_sc, scaled_sc

# wrapper for average_segcover_custom which allows it to operate over a batch
def average_segcover_batch(segA, segB, ignore_background=False, background_label=3, ignore_null_case=True):
    '''
    expects torch tensors
    returns list of floats
    '''
    batch_size = segA.shape[0]
    batch_mean_sc = []
    batch_scaled_sc = []
    
    for idx in range(batch_size):
        mean_sc, scaled_sc = average_segcover_custom(segA[idx], segB[idx], ignore_background, background_label, ignore_null_case)
        # if returned val is negative, it means the given GT labeled image was bacground only, 
        # so we skip it for SC calculation
        if mean_sc == -1 and scaled_sc == -1: continue
        batch_mean_sc.append(mean_sc.item())
        batch_scaled_sc.append(scaled_sc.item())
    
    return batch_mean_sc, batch_scaled_sc

# Genesis version (with very slight edits); url: https://github.com/applied-ai-lab/genesis/blob/master/utils/misc.py 
def average_segcover_Genesis(segA, segB, ignore_background=False, background_label=3):
    """
    Covering of segA by segB
    segA.shape = [batch size, 1, img_dim1, img_dim2]
    segB.shape = [batch size, 1, img_dim1, img_dim2]
    scale: If true, take weighted mean over IOU values proportional to the
           the number of pixels of the mask being covered.
    Assumes labels in segA and segB are non-negative integers.
    Negative labels will be ignored.
    OTHER ASSUMPTIONS:
    - assumes segA is the groundtruth
    - assumes that the background label for GT is 0
    - produces a score of 0 when the GT label mask has no objects (ie, bg only)
        (this is different from our version, which skips said input)
    """

    assert segA.shape == segB.shape, f"{segA.shape} - {segB.shape}"
    assert segA.shape[1] == 1 and segB.shape[1] == 1
    bsz = segA.shape[0]
    #nonignore = (segA >= 0) #EDITTED (ie, commented out)

    mean_scores = torch.tensor(bsz*[0.0])
    N = torch.tensor(bsz*[0])
    scaled_scores = torch.tensor(bsz*[0.0])
    scaling_sum = torch.tensor(bsz*[0])

    # Find unique label indices to iterate over
    if ignore_background:
        iter_segA = torch.unique(segA[segA != background_label]).tolist()
    else:
        iter_segA = torch.unique(segA).tolist()
    iter_segB = torch.unique(segB[segB >= 0]).tolist()
    # Loop over segA
    for i in iter_segA:
        binaryA = segA == i
        if not binaryA.any():
            continue
        max_iou = torch.tensor(bsz*[0.0])
        # Loop over segB to find max IOU
        for j in iter_segB:
            # Do not penalise pixels that are in ignore regions
            binaryB = (segB == j) #* nonignore #EDITTED (ie, commented out)
            if not binaryB.any():
                continue
            iou = iou_binary(binaryA, binaryB)
            max_iou = torch.where(iou > max_iou, iou, max_iou)
        # Accumulate scores
        mean_scores += max_iou
        N = torch.where(binaryA.sum((1, 2, 3)) > 0, N+1, N)
        scaled_scores += binaryA.sum((1, 2, 3)).float() * max_iou
        scaling_sum += binaryA.sum((1, 2, 3))

    # Compute coverage
    mean_sc = mean_scores / torch.max(N, torch.tensor(1)).float() #NOTE: it seems this method reports a core of 0 for cases when there is only background in the GT frame
    scaled_sc = scaled_scores / torch.max(scaling_sum, torch.tensor(1)).float()

    # Sanity check
    assert (mean_sc >= 0).all() and (mean_sc <= 1).all(), mean_sc
    assert (scaled_sc >= 0).all() and (scaled_sc <= 1).all(), scaled_sc
    assert (mean_scores[N == 0] == 0).all()
    # assert (mean_scores[nonignore.sum((1, 2, 3)) == 0] == 0).all()
    assert (scaled_scores[N == 0] == 0).all()
    # assert (scaled_scores[nonignore.sum((1, 2, 3)) == 0] == 0).all()

    # Return mean over batch dimension 
    # return mean_sc.mean(0), scaled_sc.mean(0)
    return np.array(mean_sc), np.array(scaled_sc)

def object_detection_null_case_count(segA, segB):
    """
    For every GT image frame where there are no objects present (a 'null' scene), 
    we detect whether the model correctly produced a 'null' prediction (ie, an label map
    with only a single label). A count of 1 is returned if there is a True Null Predictions
    (ie, a True Negative), a count of 0 otherwise if the model incorrectly detected and object. 
    In the case that the GT does contain object labels, a -1 is returned
    These counts are later used to compute the following ratio:
        -  True Null Predictions / Null Ground Truth Frames
    Assumptions
    - segA is taken to be ground truth
    """
    assert segA.shape == segB.shape, f"{segA.shape} - {segB.shape}"

    num_unique_labels_gt = len(torch.unique(segA).tolist())
    num_unique_labels_pred = len(torch.unique(segB).tolist())

    if num_unique_labels_gt == 1:
        if num_unique_labels_pred == 1:
            return 1
        else:
            return 0
    else: return -1

# Wrapper for 'object_detection_null_case_count'
def object_detection_null_case_count_batch(segA, segB):
    batch_size = segA.shape[0]
    gt_null_count = 0
    pred_true_null_count = 0
    for idx in range(batch_size):
        result = object_detection_null_case_count(segA[idx], segB[idx])
        if result != -1:
            gt_null_count += 1
            pred_true_null_count += result
    return gt_null_count, pred_true_null_count

def map_intersection_metric(masks_inferred, masks_gt, latent=None, latent_imagine=None,
                            opple=False, threshold=0.25, argmax=True):
    #### assuming input to be batch x slots x h x w\

    if latent is not None:
        if masks_gt.shape[1] > latent.shape[1]:
            # the last slot is background
            latent = torch.cat([latent, torch.zeros(latent.shape[0], 1, latent.shape[2], device=latent.device)], 1)
        latent_aligned = torch.zeros_like(latent, device=latent.device)        
    else: latent_aligned = None
    
    if latent_imagine is not None:
        if masks_gt.shape[1] > latent_imagine.shape[1]:
            # the last slot is background
            latent_imagine = torch.cat([latent_imagine,
                                        torch.zeros(latent_imagine.shape[0], 1, latent_imagine.shape[2],
                                                    device=latent.device)], 1)
        latent_imagine_aligned = torch.zeros_like(latent_imagine, device=latent_imagine.device)
    else: latent_imagine_aligned = None

    pixel_per_object_inferredaligned = torch.zeros(masks_inferred.shape[0], masks_inferred.shape[1], device=masks_inferred.device)
    pixel_per_object_gt = torch.zeros(masks_inferred.shape[0], masks_inferred.shape[1], device=masks_inferred.device)
    IoUs = torch.zeros(masks_gt.shape[0], masks_gt.shape[1], device=masks_inferred.device)
    for b in range(masks_inferred.shape[0]):
        
        if argmax:
            inferred_mask = masks_inferred[b] == torch.max(masks_inferred[b], 0, keepdim=True)[0]
            groundtruth_mask = masks_gt[b] == torch.max(masks_gt[b], 0, keepdim=True)[0]
        else:
            inferred_mask = masks_inferred[b] > threshold
            groundtruth_mask = masks_gt[b] > threshold
        pixel_per_object_gt[b] = torch.sum(groundtruth_mask, (1, 2))
        intersect = torch.sum(\
            torch.logical_and(groundtruth_mask[:,None,:,:], inferred_mask[None,:,:,:]), (2,3))
        union = torch.sum(\
            torch.logical_or(groundtruth_mask[:,None,:,:], inferred_mask[None,:,:,:]), (2,3))
        all_IoU = intersect / (union + 1e-10)
        all_IoU[torch.nonzero(torch.isnan(all_IoU))] = 0
        slots_considered = masks_inferred.shape[1]
        
        
        for i in range(slots_considered):
            best_match = torch.argmax(all_IoU)
            r = torch.div(best_match, masks_inferred.shape[1], rounding_mode='trunc')
            c = best_match % masks_inferred.shape[1]
            
            IoUs[b,r] = all_IoU[r,c]
            all_IoU[r,:] = -1
            all_IoU[:,c] = -1
            pixel_per_object_inferredaligned[b,r] = torch.sum(inferred_mask[c,:,:])
            if latent is not None:
                latent_aligned[b,r] = latent[b,c]
            if latent_imagine is not None:
                latent_imagine_aligned[b,r] = latent_imagine[b,c]

    if opple:
        return IoUs, pixel_per_object_inferredaligned, pixel_per_object_gt, latent_aligned, latent_imagine_aligned
    else:
        return IoUs, pixel_per_object_inferredaligned, pixel_per_object_gt

class LatentDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return self.data['latent'].shape[0]

    def __getitem__(self, idx):
        latent = torch.tensor(self.data['latent'][idx], dtype=torch.float)
        target = torch.tensor(self.data['target'][idx])
        return latent, target

class latentclassify():
    def __init__(self, contextlabels, device, context="objshape"):
        self.numbins = 32
        self.idsize = 64
        self.context = context
        self.device = device
        self.contextlabels = contextlabels.get(context)
        print("context labels:", self.contextlabels, len(self.contextlabels))
        self.labelshape = len(self.contextlabels)
    
    def linearclassifier(self, inputshape, classes, lr=0.01):
        lc = nn.Sequential(nn.Linear(inputshape, classes, bias=True), nn.LogSoftmax())
        print("lc:", lc)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(lc.parameters(), lr=lr)
        return lc, criterion, optimizer
    
    def datasetmaker(self, trainpickle, testpickle):
        trainlatent, trainobjdeets, trainscenedeets, _, _ = self.contextlatent_unpickle(trainpickle)
        testlatent, testobjdeets, testscenedeets, _, _ = self.contextlatent_unpickle(testpickle)

        print("train latent data shape:", trainlatent.shape, trainobjdeets.shape)

        trainlabels = None
        testlabels = None
        ## -1 because all labels are 1-indexed
        if(self.context == "objshape"):
            trainlabels = trainobjdeets[:,0]-1
            testlabels = testobjdeets[:,0]-1
        elif(self.context == "objtex"):
            trainlabels = trainobjdeets[:,1]-1
            testlabels = testobjdeets[:,1]-1
        elif(self.context == "walltex"):
            trainlabels = trainscenedeets[:,0]-1
            testlabels = testscenedeets[:,0]-1
        elif(self.context == "floortex"):
            trainlabels = trainscenedeets[:,1]-1
            testlabels = testscenedeets[:,1]-1

        traindataset = LatentDataset({'latent':trainlatent, 'target':trainlabels})
        testdataset = LatentDataset({'latent':testlatent, 'target':testlabels})

        return traindataset, testdataset

    def contextlatent_unpickle(self, latentspickle):
        file = open(latentspickle, 'rb')
        things = pickle.load(file)
        latents = np.asarray(things['latents'])
        objdeets = np.asarray(things['objdeets'])
        scenedeets = np.asarray(things['scenedeets'])
        # latents_imagine = np.asarray(things['latents_imagine'])
        # objloc_gtworld = np.asarray(things['objloc_gtworld'])
        objloc_gtcam = np.asarray(things['objloc_gtcam'])

        objdeets = objdeets.reshape((objdeets.shape[0]*objdeets.shape[1], objdeets.shape[2]))
        scenedeets = scenedeets.reshape((scenedeets.shape[0]*scenedeets.shape[1], scenedeets.shape[2]))
        latents = latents.reshape((latents.shape[0]*latents.shape[1], latents.shape[2]))
        latents = latents[:,75:]
        objloc_pcam = latents[:,:3]
        objloc_gtcam = objloc_gtcam.reshape((objloc_gtcam.shape[0]*objloc_gtcam.shape[1], objloc_gtcam.shape[2]))
        return latents, objdeets, scenedeets, objloc_gtcam, objloc_pcam

    def main(self, trainpickle, testpickle, epochs):

        traindataset, testdataset = self.datasetmaker(trainpickle, testpickle)
        trainloader = torch.utils.data.DataLoader(traindataset, batch_size=24, shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(testdataset, batch_size=24, shuffle=True, num_workers=4)
        lc, criterion, optimizer = self.linearclassifier(self.idsize, self.labelshape, 0.001)

        lc.to(self.device)

        for e in range(epochs):
            correct = 0
            total = 0
            print(len(trainloader))
            for b, batch in enumerate(trainloader):
                optimizer.zero_grad()
                latent, target = batch
                latent = latent.to(self.device)
                target = target.to(self.device)
                output = lc(latent)

                _, prediction = torch.max(output,1)
                total += prediction.shape[0]
                correct += (target == prediction).sum().item()
                
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                if(b%2000 == 0):
                    print("train loss at epoch:{} and batch:{} is {}".format(e, b, loss))
                    lcw = lc[0].weight
                    print("lc weights: mean = {}, min = {}, max = {}".format(torch.mean(lcw), torch.min(lcw), torch.max(lcw)))
            print("accuracy of the network on latent based classification during training is {}".format(correct/total))

        correct = 0
        total = 0
        with torch.no_grad():
            for b,batch in enumerate(testloader):
                latent, target = batch
                latent = latent.to(self.device)
                target = target.to(self.device)
                output = lc(latent)
                _, prediction = torch.max(output,1)
                total += prediction.shape[0]
                correct += (target == prediction).sum().item()
        print("accuracy of the network on latent based classification during test is {}".format(correct/total))

class latentcontext():
    def __init__(self, contextlabels, device, context="objshape"):
        self.numlabels = len(contextlabels[context])
        self.contextlabels = contextlabels[context]
        self.contextlabels_all = contextlabels
        self.onlinepaircount = torch.zeros((self.numlabels, self.numlabels), device=device)
        self.onlinecorrelation = torch.zeros((self.numlabels, self.numlabels), device=device)
        self.context = context
        self.device = device
        self.numpairs = self.numlabels**2
        self.batchcount = 0

    def getdistmatrix(self):
        return self.onlinecorrelation

    def onlineupdate(self, latents, objdeets, scenedeets):
        labels = None
        ## -1 because all labels are 1-indexed
        if(self.context == "objshape"):
            labels = objdeets[:,0]-1
        elif(self.context == "objtex"):
            labels = objdeets[:,1]-1
        elif(self.context == "walltex"):
            labels = scenedeets[:,0]-1
        elif(self.context == "floortex"):
            labels = scenedeets[:,1]-1

        latentdist = torch.mean((latents[:,None,:] - latents[None,:,:])**2, -1).unsqueeze(0).repeat(self.numpairs,1,1)
        labelcross = labels[:,None]*labels[None,:]
        labelcross = labelcross.unsqueeze(0).repeat(self.numpairs,1,1).to(self.device)
        labelval = torch.ones_like(labelcross, device=self.device)*(torch.range(0, self.numpairs-1, device=self.device))[:,None,None]
        latentdistpair = torch.where(labelcross.float() == labelval.float(), latentdist.float(), torch.tensor([0.0], dtype=torch.float, device=self.device))
        latentdistpair = torch.mean(latentdistpair, dim=(1,2))
        latentdistpair = latentdistpair.reshape(self.onlinepaircount.shape)
        self.onlinecorrelation = (self.batchcount*self.onlinecorrelation + latentdistpair)/(self.batchcount+1)
        self.batchcount += 1
        gc.collect()

    def contextlatent_unpickle(self, latentspickle):
        
        file = open(latentspickle, 'rb')
        things = pickle.load(file)
        latents = np.asarray(things['latents'])
        print('latents',latents.shape)
        latents = np.reshape(latents,[-1,3, 139])
        objdeets = np.asarray(things['objdeets'])
        scenedeets = np.asarray(things['scenedeets'])
        # latents_imagine = np.asarray(things['latents_imagine'])
        # objloc_gtworld = np.asarray(things['objloc_gtworld'])
        objloc_gtcam = np.asarray(things['objloc_gtcam'])
        print('objloc_gtcam',objloc_gtcam.shape)
        objloc_gtcam = np.reshape(objloc_gtcam, [-1, 3, 3])
        objdeets = objdeets.reshape((objdeets.shape[0]*objdeets.shape[1], objdeets.shape[2]))
        scenedeets = scenedeets.reshape((scenedeets.shape[0]*scenedeets.shape[1], scenedeets.shape[2]))
        objloc_pcam = latents[:,:,:3]
        objloc_gtcam = objloc_gtcam.reshape((objloc_gtcam.shape[0],objloc_gtcam.shape[1], objloc_gtcam.shape[2]))
        print(objdeets.shape, latents.shape, scenedeets.shape, objloc_gtcam.shape, objloc_pcam.shape)

        return latents, objdeets, scenedeets, objloc_gtcam, objloc_pcam

    def latent_context_correlation(self, latentspickle):
        onlinecorr = self.onlinecorrelation.numpy()

        latents, objdeets, scenedeets, _, _ = self.contextlatent_unpickle(latentspickle)

        labels = None
        ## -1 because all labels are 1-indexed
        if(self.context == "objshape"):
            labels = objdeets[:,0]-1
        elif(self.context == "objtex"):
            labels = objdeets[:,1]-1
        elif(self.context == "walltex"):
            labels = scenedeets[:,0]-1
        elif(self.context == "floortex"):
            labels = scenedeets[:,1]-1

        idxs = []
        for lb in range(self.numlabels):
            idx = np.nonzero(np.where(labels == lb, 1, 0))
            idx = np.asarray(idx).flatten()
            idx = np.random.choice(idx, 1000)
            idxs.append(idx)

        for i in range(len(idxs)):
            for j in range(len(idxs)):
                latentdist = np.mean(np.sqrt(np.sum((latents[tuple(idxs[i]),None,:] - latents[None,tuple(idxs[j]),:])**2, 2)), axis=(0,1))
                onlinecorr[i,j] = latentdist

        return onlinecorr

    def plot3d(self, embed, labels, keys, vals, loc):

        key_list, key_list2 = keys
        val_list, val_list2 = vals
        label_idx, label2_idx, idxs2, labels, labels2 = labels

        print(embed.shape, label_idx.shape, label2_idx.shape)
        cmap = matplotlib.cm.get_cmap('Spectral') 
        markers = ["o", "2", "^", "8", "+", "D", "s", "X", "*", "p", "P", "H", "v", "1", "3", "4", "<",">", "h"] 
        fig = plt.figure(dpi=200, figsize=[12,6])

        ax = fig.add_subplot(1,2,1, projection='3d')
        for i in range(embed.shape[0]):
            ax.scatter(embed[i,0], embed[i,1], embed[i,2], color=cmap((label_idx[i]+1)/(self.numlabels+1)), label=key_list[val_list.index(labels[i])])#, marker = markers[label2_idx[i]-1])

        plt.axis('off')

        ax = fig.add_subplot(1,2,2, projection='3d')
        for i in range(embed.shape[0]):
            ax.scatter(embed[i,0], embed[i,1], embed[i,2], color=cmap((label2_idx[i])/(len(idxs2[0]))), label=key_list2[val_list2.index(labels2[i])])

        # plt.legend(loc='upper left', bbox_to_anchor = (1.05, 0.7, 0.1, 0.3))
        plt.axis('off')
        plt.show()
        plt.savefig(loc+'.png')#, bbox_inches='tight')
        plt.cla()
        plt.close(fig)

    def latent_context_mds(self, latentspickle, loc):
        latents, objdeets, scenedeets, _, _ = self.contextlatent_unpickle(latentspickle)
        labels = None
        labels2 = objdeets[:,1]
        context2 = "objtex"
        contextlabels2 = self.contextlabels_all[context2]
        numlabels2 = len(contextlabels2)

        ## -1 because all labels are 1-indexed
        if(self.context == "objshape"):
            labels = objdeets[:,0]
        elif(self.context == "objtex"):
            labels = objdeets[:,1]
        elif(self.context == "walltex"):
            labels = scenedeets[:,0]
        elif(self.context == "floortex"):
            labels = scenedeets[:,1]

        print(labels2.shape, labels.shape)

        idxs = []
        idxs2 = []
        label_idx = []
        label2_idx = []
        for lb in range(self.numlabels):
            idx = np.nonzero(np.where(labels == lb+1, 1, 0))
            idx = np.asarray(idx).flatten()
            idx = np.random.choice(idx, 50)
            label_idxt = labels[idx]
            idxs2_t = []
            idxs.append(idx)
            for lb2 in range(numlabels2):
                idx2 = np.nonzero(np.where(label_idxt==lb2+1, 1, 0))
                idx2 = np.asarray(idx2).flatten()
                idxs2_t.append(idx2)

            label_idx.append(labels[idx])
            label2_idx.append(labels2[idx])
            idxs2.append(idxs2_t)

        label_idx = np.concatenate(label_idx, 0)
        label2_idx = np.concatenate(label2_idx, 0)

        idxchoice = idxs[0].shape[0] 
        fulldistmat = np.zeros((len(idxs)*idxs[0].shape[0], len(idxs)*idxs[0].shape[0]))
        for i in range(len(idxs)):
            for j in range(len(idxs)):
                latentdist = np.sqrt(np.sum((latents[tuple(idxs[i]),None,:] - latents[None,tuple(idxs[j]),:])**2, axis=2))
                fulldistmat[i*idxchoice:(i+1)*idxchoice, j*idxchoice:(j+1)*idxchoice] = latentdist

        key_list = list(self.contextlabels.keys())
        val_list = list(self.contextlabels.values())
        
        key_list2 = list(contextlabels2.keys())
        val_list2 = list(contextlabels2.values())
        print(len(key_list2))

        seed = np.random.RandomState(seed=3)
        mds = manifold.MDS(n_components=3, max_iter=3000, eps=1e-9,
                   dissimilarity="precomputed", verbose=1, n_jobs=10, random_state=seed)
        embed = mds.fit_transform(fulldistmat)

        # tsne = manifold.TSNE(n_components=2, metric="precomputed", verbose=1)
        # embed = tsne.fit_transform(fulldistmat)

        self.plot3d(embed, [label_idx, label2_idx, idxs2, labels, labels2], [key_list, key_list2], [val_list, val_list2], loc)


    def latent_multicontext_mds(self, latentspickle, loc):
        latents, objdeets, scenedeets, _, _ = self.contextlatent_unpickle(latentspickle)

        labels = None

        labels2 = objdeets[:,1]
        context2 = "objtex"
        contextlabels2 = self.contextlabels_all[context2]
        numlabels2 = len(contextlabels2)

        ## -1 because all labels are 1-indexed
        if(self.context == "objshape"):
            labels = objdeets[:,0]
        elif(self.context == "objtex"):
            labels = objdeets[:,1]
        elif(self.context == "walltex"):
            labels = scenedeets[:,0]
        elif(self.context == "floortex"):
            labels = scenedeets[:,1]

        print(labels2.shape, labels.shape)

        idxs = []
        idxs2 = []
        label_idx = []
        label2_idx = []
        for lb in range(self.numlabels):
            idx = np.nonzero(np.where(labels == lb+1, 1, 0))
            idx = np.asarray(idx).flatten()
            # idx = np.random.choice(idx, 50)
            label_idxt = labels[idx]
            label2_idxt = labels2[idx]
            idxs2_t = []
            idxs.append(idx)

            for lb2 in range(numlabels2):
                idx2 = np.nonzero(np.where(label2_idxt==lb2+1, 1, 0))
                idx2 = np.asarray(idx2).flatten()
                idx2 = np.random.choice(idx2, 10)
                idxs2_t.append(idx[idx2])

                label_idx.append(label_idxt[idx2])
                label2_idx.append(label2_idxt[idx2])
            idxs2.extend(idxs2_t)

        print("idxs2 shape", len(idxs2))
        label_idx = np.concatenate(label_idx, 0)
        label2_idx = np.concatenate(label2_idx, 0)

        idxchoice = idxs2[0].shape[0] 
        fulldistmat = np.zeros((len(idxs2)*idxs2[0].shape[0], len(idxs2)*idxs2[0].shape[0]))
        for i in range(len(idxs2)):
            for j in range(len(idxs2)):
                latentdist = np.sqrt(np.sum((latents[tuple(idxs2[i]),None,:] - latents[None,tuple(idxs2[j]),:])**2, 2))
                fulldistmat[i*idxchoice:(i+1)*idxchoice, j*idxchoice:(j+1)*idxchoice] = latentdist

        key_list = list(self.contextlabels.keys())
        val_list = list(self.contextlabels.values())
        
        key_list2 = list(contextlabels2.keys())
        val_list2 = list(contextlabels2.values())
        print(len(key_list2))

        seed = np.random.RandomState(seed=3)
        mds = manifold.MDS(n_components=3, max_iter=3000, eps=1e-9,
                   dissimilarity="precomputed", verbose=1, n_jobs=10, random_state=seed)
        embed = mds.fit_transform(fulldistmat)

        # tsne = manifold.TSNE(n_components=2, metric="precomputed", verbose=1)
        # embed = tsne.fit_transform(fulldistmat)

        self.plot3d(embed, [label_idx, label2_idx, idxs2, labels, labels2], [key_list, key_list2], [val_list, val_list2], loc)

    def object_localization(self, latentspickle, loc):
        latents, objdeets, scenedeets, objloc_gtcam, objloc_pcam = self.contextlatent_unpickle(latentspickle)
        print('objloc_gtcam range', np.min(objloc_gtcam), np.max(objloc_gtcam))
        print('objloc_pcam range', np.min(objloc_pcam[:,:,0]), np.max(objloc_pcam[:,:,0]),
             np.min(objloc_pcam[:,:,1]), np.max(objloc_pcam[:,:,1]))
        labels = None
        ## -1 because all labels are 1-indexed
        if(self.context == "objshape"):
            labels = objdeets[:,0]-1
        elif(self.context == "objtex"):
            labels = objdeets[:,1]-1
        elif(self.context == "walltex"):
            labels = scenedeets[:,0]-1
        elif(self.context == "floortex"):
            labels = scenedeets[:,1]-1

        objerrordist = np.sqrt(np.sum((objloc_gtcam[:,:2]-objloc_pcam[:,:2])**2, 1))
        objpreddist = np.sqrt(np.sum((objloc_pcam[:,:2])**2, 1))
        objtruedist = np.sqrt(np.sum((objloc_gtcam[:,:2])**2, 1))
        objtrueyangle = np.arctan2(objloc_gtcam[:,1], objloc_gtcam[:,0])
        objpredyangle = np.arctan2(objloc_pcam[:,1], objloc_pcam[:,0])

        # maxangle = 25*(np.pi/180)
        usablepoints1 = objloc_gtcam[:,1]>0
        usablepoints = np.nonzero(usablepoints1)[0]
        print("usable points:", usablepoints.shape)

        objpreddist = objpreddist[usablepoints]
        objerrordist = objerrordist[usablepoints]
        objtruedist = objtruedist[usablepoints]
        objtrueyangle = objtrueyangle[usablepoints]*(180/np.pi)
        print(np.min(objtrueyangle), np.max(objtrueyangle))

        objdistratio = objpreddist/(objtruedist+1e-8)

        angle_bins = [i for i in range(90-50, 90+56, 5)] 
        yaxis_mean = []
        yaxis_std = []
        for i in range(len(angle_bins)-1):
            idx1 = np.where(objtrueyangle>angle_bins[i], objtrueyangle, 350)
            idxs = np.nonzero(np.where(idx1<=angle_bins[i+1], 1, 0))
            yvals = objerrordist[idxs]
            if(yvals.shape[0]==0):
                yaxis_mean.append(0)
                yaxis_std.append(0)
            else:
                yaxis_mean.append(np.mean(yvals))
                yaxis_std.append(np.std(yvals))


        fig,ax = plt.subplots()
        ax.errorbar(angle_bins,
                    yaxis_mean, yerr = yaxis_std, linewidth=4, fmt='o', capsize=10)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(bottom=0)
        plt.savefig(loc.format(x="localize-angle"))
        plt.cla()
        plt.close(fig)

        dist_bins = [i/2 for i in range(1, 16)] 
        yaxis_mean = []
        yaxis_std = []
        for i in range(len(dist_bins)-1):
            idx1 = np.where(objtruedist>dist_bins[i], objtruedist, 350)
            idxs = np.nonzero(np.where(idx1<=dist_bins[i+1], 1, 0))
            yvals = objerrordist[idxs]
            if(yvals.shape[0]==0):
                yaxis_mean.append(0)
                yaxis_std.append(0)
            else:
                yaxis_mean.append(np.mean(yvals))
                yaxis_std.append(np.std(yvals))

        fig,ax = plt.subplots()
        ax.errorbar(dist_bins, yaxis_mean, yerr = yaxis_std, linewidth=4, fmt='o', capsize=10)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(bottom=0)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        ax.set_ylim(bottom=0)
        # plt.scatter(objtrueyangle, objerrordist, alpha=0.5, s=2)
        plt.savefig(loc.format(x="localize-dist"))

        plt.cla()
        plt.close(fig)

