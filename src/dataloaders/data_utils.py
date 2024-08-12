import sys
import torch
from torch.utils.data import WeightedRandomSampler, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
import cv2
# cv2.setNumThreads(0)
import numpy as np
import faulthandler; faulthandler.enable()
from copy import copy

# Dataloader function for use when dataset files are already split between train, val, and test
def get_dataloader(dataset, batch_size, num_workers, sample_weights=None, random_sampling=False, pin=False):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    if random_sampling:
        sampler = RandomSampler(np.arange(len(indices)))
    else:
        sampler = SequentialSampler(np.arange(len(indices)))
    if sample_weights is not None:
        print("ERROR: dataloader hasn't been configured to use WeightedRandomSampler yet")
        exit(1)

    loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size,\
                                    num_workers=num_workers, pin_memory=pin, drop_last=True)
    return loader

# This func is written for backwards compatibility with non-presplit datasets. It may be removed in a future update
def bkwrd_compat_set_dataloaders(cfg, params, dataloader_class):
    
    dataset_path = cfg.DATASET.root_dir_path
    images_dir_name = cfg.DATASET.images_dir_name
    pickled_datamap_filename = cfg.DATASET.pickled_datamap_filename
    dataloader_class = dataloader_class

    ## Initialize the Dataset
    # For multimoving experiments
    if cfg.EXPERIMENT.exp_type == 'multimoving':
        dataset = dataloader_class(cfg, params, dataset_path, images_dir_name,\
            pickled_datamap_filename, split=None)
    else:
        print(f"ERROR: Unrecognized exp_type '{cfg.EXPERIMENT.exp_type}'. Training/testing has not yet been configured for this type.")
        sys.exit("Exitting...")

    if params["test"]:
        batch_size = cfg.TEST.batch_size
        num_workers = cfg.TEST.num_workers
    else: # for train and val sets
        batch_size = cfg.TRAIN.batch_size
        num_workers = cfg.TRAIN.num_workers

    # Split the Dataset into Subsets and instantiate Dataloaders
    sample_weights = None
    if cfg.DATASET.reweight:
        sample_weights = dataset.get_sampling_weights()

    data_split = DataSplit(dataset, cfg.TRAIN.test_train_split, cfg.TRAIN.val_train_split,\
            random_sampling=cfg.TRAIN.random_sampling, sample_weights=sample_weights, triplets=(not params['perframe']))
    train_loader, val_loader, test_loader = data_split.get_split(batch_size, num_workers, pin=True)

    return {'train_dataloader' : train_loader, 'validation_dataloader' : val_loader,
            'test_dataloader': test_loader}

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):    
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def camera_viewpoint_transform(v):
    '''
    Returns a transformed version of campera position data.
    Specifically, this conversion func converts cam pos data represented in a Unity
    based coordinate system (left-handed, Y-up) to the coordinate system our model
    assumes (right-handed, Z-up). It also fulfills other data representation expectations
    our model expects (explained further in the code documentation files).

    Specifcally, this functions makes the following transformations:
    - converts from +yaw change coressponding to clockwise rotation to
    counter-clockwise rotation
    - reduces the XYZ coord points by a factor of 10

    Input:
        v:      torch tensor of size [6], representing the following:
                [ X, Y, Z, yaw, pitch, roll]

    Returned:
        v_hat:  torch tensor of size [9], represetning the following:
                [ X, Y, Z, cos(yaw), sin(yaw), cos(pitch),
                             sin(pitch), cos(roll), sin(roll) ]
    '''
    transform, rotation = torch.split(v, 3, dim=-1)
    transform = transform / 10.0
    yaw, pitch, roll = torch.split(rotation*(np.pi/180), 1, dim=-1)
    x_orig, y_orig, z_orig = torch.split(transform, 1, dim=-1)
    
    x = x_orig
    y = z_orig
    z = y_orig
    # From observation of the data, positive angle is rotating clockwise from y-axis.
    # and negetive angle is rotating counter-clockwise from y-axis.
    yaw = -1 * yaw 
    
    # position, [yaw, pitch]
    view_vector = [x, y, z, torch.cos(yaw), torch.sin(yaw),
                   torch.cos(pitch), torch.sin(pitch), torch.cos(roll), torch.sin(roll)]
    v_hat = torch.cat(view_vector, dim=-1)
    return v_hat

def camera_viewpoint_transform_waymo(v): ## transforming camera tensor for Waymo data
    transform, rotation = torch.split(v, 3, dim=-1)
    yaw, pitch, roll = torch.split(rotation, 1, dim=-1)
    x_orig, y_orig, z_orig = torch.split(transform, 1, dim=-1)
    
    x = x_orig
    y = z_orig
    z = y_orig

    # From observation of the Waymo data, positive angle is rotating counterclockwise 
    # from x-axis (at which yaw is zero).
    
    # position, [yaw, pitch]
    view_vector = [x, y, z, torch.cos(yaw), torch.sin(yaw),
                   torch.cos(pitch), torch.sin(pitch), torch.cos(roll), torch.sin(roll)]
    v_hat = torch.cat(view_vector, dim=-1)
    return v_hat

def camera_viewpoint_transform_movi(v):
    transform, rotation = torch.split(v, 3, dim=-1)
    yaw, pitch, roll = torch.split(rotation, 1, dim=-1)
    x, y, z = torch.split(transform, 1, dim=-1)
    
    view_vector = [x, y, z, torch.cos(yaw), torch.sin(yaw),
                   torch.cos(pitch), torch.sin(pitch), torch.cos(roll), torch.sin(roll)]
    v_hat = torch.cat(view_vector, dim=-1)
    return v_hat

def camera_viewpoint_transform_gqn(v): ## transforming camera tensor for GQN data
    transform, rotation = torch.split(v, 3, dim=-1)
    yaw, pitch, roll = torch.split(rotation, 1, dim=-1)
    x_orig, y_orig, z_orig = torch.split(transform, 1, dim=-1)
    
    # The original range of translation and depth is too large.
    x = x_orig
    y = -z_orig
    z = y_orig 
    yaw = yaw + (np.pi / 2)
    
    # position, [yaw, pitch]
    view_vector = [x, y, z, torch.cos(yaw), torch.sin(yaw),
                   torch.cos(pitch), torch.sin(pitch), torch.cos(roll), torch.sin(roll)]
    v_hat = torch.cat(view_vector, dim=-1)

    return v_hat

def object_viewpoint_transform(v): ## transforming object tensor
    '''
        TODO: input and output (format and representations (ie, coord systems)
        ought to be explicitly defined here, along with any other assumptions/expectations
        this funtion makes) - John
    '''

    # _, transform = torch.split(v, 1, dim=-1)
    # transform = v[1:]
    transform = v / 10.0 
    # yaw, pitch, roll = torch.split(rotation*(np.pi/180), 1, dim=-1)
    x_orig, y_orig, z_orig = torch.split(transform, 1, dim=-1)
    
    # The original range of translation and depth is too large.
    x = x_orig
    y = z_orig
    z = y_orig
    # From observation of the data, positive angle is rotating clockwise from y-axis.
    # and negetive angle is rotating counter-clockwise from y-axis.
    
    # position, [yaw, pitch]
    # view_vector = [x, y, z, torch.cos(yaw), torch.sin(yaw),
                   # torch.cos(pitch), torch.sin(pitch), torch.cos(roll), torch.sin(roll)]
    view_vector = [x,y,z]
    v_hat = torch.cat(view_vector, dim=-1)

    return v_hat

def object_pose_viewpoint_transform(v): ## transforming object pose tensor
    transform = v
    yaw, pitch, roll = torch.split(transform*(np.pi/180), 1, dim=-1)
    yaw = np.pi / 2 - yaw # From observation of the data, positive angle is rotating clockwise from y-axis.
    # and negetive angle is rotating counter-clockwise from y-axis.
    view_vector = [torch.cos(yaw), torch.sin(yaw), torch.cos(pitch), torch.sin(pitch), torch.cos(roll), torch.sin(roll)]
    v_hat = torch.cat(view_vector, dim=-1)

    return v_hat

def transform_c(i):
    mx = 255.0
    i = torch.div(i, mx)
    return i

def to_grayscale(i):
    image_transformed = (torch.sum(i, -1)/3).unsqueeze(0)
    return image_transformed

def rgb_2_label_map(rgb_img, bg_color = (255, 0, 0)):
    '''
        Decription: Converts 3-channel RGB Image to Single-Channel Label Map
                    In the label map representation, the background will be mapped to 
                    label 0 (according to bg_color)
        
        rgb_img:    3 channel RGB image representing segmentation labels
                    dtype: numpy.ndarray or torch.tensor
                    shape: [height x width x 3]
        bg_color:   specifies which RGB value in the input image to treat as background
                    and therefore to map to label 0 in the outputted map

        output:     Single-Channel Label Map
                    dtype: np.ndarray
                    shape [height x width]
    '''

    # If input is torch tensor, convert to numpy array
    if type(rgb_img) == torch.Tensor:
        rgb_img = rgb_img.detach().cpu().numpy()

    # The color values expected in input image (which will each be mapped to a unique label num)
    expected_colors = [
        (255, 0, 0),  #red
        (0, 255, 0),  #green
        (0, 0, 255),  #blue
        (255, 255, 0) #yellow
    ]

    # Reorder the list of colors such that the background color is at front of the list
    bg_color_index = expected_colors.index(bg_color)
    expected_colors[bg_color_index], expected_colors[0] =\
        expected_colors[0], expected_colors[bg_color_index]
    
    # Convert to numpy array
    expected_colors = np.array(expected_colors, dtype=np.int)

    # Check for unexpected colors in the input image
    # flatten the spatial dims of the img data
    img_flat = np.reshape(rgb_img, (rgb_img.shape[0]*rgb_img.shape[1], -1))
    found_colors = np.unique(img_flat, axis=0)
    for color in found_colors:
        if not np.any(np.all(color == expected_colors, axis=1)):
            print("Unexpected color found in RGB segmentation label image:", color)
            print("Exiting...")
            exit(1)

    # Map the pixels of each unique color in the input image to a unique label for the
    # correpsonding pixels in the label map
    label_map = np.zeros(shape=(rgb_img.shape[:2]), dtype=np.int)
    for label_index, color in enumerate(expected_colors):
        label_map[(rgb_img==color).all(axis=2)] = label_index

    return label_map

def label_map_2_multi_channel_mask(label_map, num_slots):
    '''
        Description:    Converts a single channel label map representaion (of
                        segmentation) to a multi-channel binary mask representation
                        Will put the background channel as the last channel/slot
        Assumptions:    *The background label is assumed to always be label 0
                        *TODO: consider adding the assumption that the input image always
                        contains at least one background pixel (ie, pixel of label 0)

        label_map:      shape:  [height x width]
                        dtype:  numpy.ndarray 
                        
        num_slots:      determines the number of binary channels for the output
                        dtype:  int

        output:         shape:  [height x width x slots]
                        dtype:  torch.tensor of dtype=torch.float
    '''

    # Get unique labels
    unique_labels = np.unique(label_map)

    # Create binary masks
    # (because torch.unique returns sorted list of values, we can expect the first
    # channel (ie, label 0) to be the background channel at this point)
    # NOTE: EXCEPTION: the above assumption breaks when the input image does not have any
    # bground pixels (and therefore the first entry in the list returned from 
    # torch.unique() will be a value other than 0). 
    # The code currently accounts for this by just not doing the swapping step below
    # for such cases (see the more detailed NOTE below)
    binary_masks = []
    for label in unique_labels:
        mask = np.where(label_map == label, 1, 0)
        binary_masks.append(mask)

    # Append channels of all zeros for channels that are not represented (ie, when
    # number of unique objects present is less than number of slots)
    if(len(binary_masks)<num_slots):
        diff = num_slots-len(binary_masks)
        for i in range(diff):
            binary_masks.append(np.zeros((label_map.shape)))

    if len(binary_masks) > num_slots:
            print("ERROR: during conversion from label map to binary masks, found more\
                   unique colors than number of slots")

    # We want to ensure the background channel is the last channel in the multi-channel
    # mask representation. We do this here by swapping the mask corresponding to the
    # background label value (assumed to be 0) with whichever mask is currently in the
    # last channel/slot
    # NOTE: this conditional is here to resolve the edge cases when there are no bground
    # pixels. In such cases, the first slot (at this point in the code) will be a nonzero
    # object label and the last slot will already be filled with 0 values (and therefore
    # appropriately represent background) so no swapping should be done.
    if 0 in unique_labels:
        BG_IDX = 0
        binary_masks[BG_IDX], binary_masks[-1] = binary_masks[-1], binary_masks[BG_IDX]

    # Now, binary_masks contains a list of binary masks, one for each unique label/slot.
    # Stack them to create a multi-channel binary mask
    return torch.tensor(np.stack(binary_masks,axis=-1), dtype=torch.float)
    """
        *NOTE: this is the original code (by Tushar) that was being used 
        to convert from RGB to a multi-channel binary bask representation of
        segmentation gorund truth as well as to correct
        for miscolorings due to interpolation/anti-aliasing artifacts. This correcting
        is no longer necesary in the newly regenerated version of our datasets. We have replaced
        this method/function with 2 new functions called in succesion (
        rgb_2_label_map() and label_map_2_multi_channel_mask()).

        *We will keep this method in the code for now for reference sake, but ideally
        it should be removed soon and especially should be removed (TODO) for publication
        - John

    """
    """
        Important function for converting images of maps in RGB to object bit maks
        Edges can be a problem to assign to the correct object as the RGB values are
        interpolated.
        We use main colors from color list to decode main objects and then do a near color
        clustering to approximate correct object edge decoding
        
        Due to interpolation in rendering the pixel values are not distinct and we end up
        getting intermediate values 
        Problem by count: a small map will be ignored for contour of a bigger map
        Problem by check for intermediate colour: we can have two object maps connected which
        will have a different intermediate colour value; check location and count both. 
        We fix this by directly decoding from the encoding scheme and set the interpolated
        colours as the nearest neighbour colour

        *Note: used to be called "slot_attention()"

        TODO: We still see some incorrect pixel to object class assignment; check how to fix
        these incorrect assignment. See if the arbitrary parameters can be fixed more sensibly.o
        ====
        attention: torch tensor: channels, width, height
    """
    # attention = to_grayscale(attention)[0]  # easier to just decode from RGB

    flat_attention = torch.flatten(attention,0,1).numpy()
    # pixel_loc = torch.flatten(self.pixel_loc, 0, 1).numpy()
    colors, idx, counts = np.unique(flat_attention, \
            return_inverse=True, axis=0, return_counts=True)
    colors = colors.astype(int)
    idx = np.reshape(idx, attention.shape[:2])
    randcheck = 10 # number of pixels to check randomly
    maxv = attention.shape[1]
    knn = 5 # random spatial pixel correlation check for pixel to object class mapping
    attention_output = []
    attention_alignment = [num_slots-1]*(num_slots)


    unique_cidxs = []
    corresponding_ucidx = []
    # pixel_loc_uc = []
    for cidx, c in enumerate(colors):
        for ucidx, uc in enumerate(unique_colors):
            if((c==uc).all()):
                unique_cidxs.append(cidx)
                corresponding_ucidx.append(ucidx)
                # uc_idxs = np.nonzero(np.where(idx == cidx, 1, 0))
                # uc_rand_idxs = np.random.choice(uc_idxs[0].shape[0], randcheck)
                # uc_random_idxs = np.vstack((uc_idxs[0][uc_rand_idxs], uc_idxs[1][uc_rand_idxs]))
                # pixel_loc_uc.append(pixel_loc[uc_random_idxs])

    for cidx, c in enumerate(colors):
        if(cidx in unique_cidxs):
            continue
        count = np.zeros(len(unique_cidxs))
        c_idxs = np.nonzero(np.where(idx == cidx, 1, 0))
        rand_idxs = np.random.choice(c_idxs[0].shape[0], randcheck)
        random_idxs = np.vstack((c_idxs[0][rand_idxs], c_idxs[1][rand_idxs]))
        # pixel_loc_c = pixel_loc[random_idxs]

        # for r in range(pixel_loc_c.shape[0]):
            # for i in range(len(unique_cidxs)):
                # for ucr in range(pixel_loc_uc[i].shape[0]):
                    # count[i]+= np.sqrt(np.sum((pixel_loc_c[r]-pixel_loc_uc[i][ucr])**2))

            # r_knn = idx[max(0, r[0]-knn):min(maxv, r[0]+knn), max(0,
                # r[1]-knn):min(maxv, r[1]+knn)]
            # for i in range(len(unique_cidxs)):
                # count[i]+=len(np.nonzero(np.where(r_knn==unique_cidxs[i], 1, 0)))
        for i in range(len(unique_cidxs)):
            count[i] = np.sqrt(np.sum((colors[unique_cidxs[i]] - c)**2)) ## matching with the closest color
        change_label = unique_cidxs[np.argmin(count)]
        idx = np.where(idx==cidx, change_label, idx)

    for k in range(len(unique_cidxs)):
        attention_output.append(np.where(idx==unique_cidxs[k], 1, 0))

    if(len(attention_output)<num_slots):
        diff = num_slots-len(attention_output)
        for i in range(diff):
            attention_output.append(np.zeros(idx.shape))

    # decoding alignment from colours 
    # TODO: decoding hardcoded for 4 maps; make it general
    for i in range(len(corresponding_ucidx)):
        if(corresponding_ucidx[i]==8):
            attention_alignment[-1]=i
        elif(corresponding_ucidx[i]==9):
            attention_alignment[0]=i
        elif(corresponding_ucidx[i]==10):
            attention_alignment[1]=i
        elif(corresponding_ucidx[i]==11):
            attention_alignment[2]=i

    attention_output = np.stack([attention_output[i] for i in attention_alignment])

    if(len(attention_output)>num_slots):
        print("error in attention map parsing; more unique colors than number of slots")

    return torch.tensor(attention_output, dtype=torch.float)

    """
        For converting (grounnd truth) segmentation maps represented as RGB to object
        bit maks. We use main colors from color list to decode

        *NOTE: this is a simplified version of the original code that was being used
        to convert from RGB to a multi-channel binary bask representation of
        segmentation gorund truth (as well, in the non-simplified version) to correct
        for miscolorings due to interpolation/anti-aliasing artifacts. We have choosen
        to replace this method/function with 2 new functions called in succesion (
        rgb_2_label_map() and label_map_2_multi_channel_mask()). This code, however,
        has been checked to be emprically equivalent to the combination of the 2 new methods
        (besides for the particular ordering of the object channels)

        *We will keep this method in the code for now for reference sake, but ideally
        it should be removed soon and especially should be removed (TODO) for publication
        - John
        
        ====
        gt_seg_img: torch tensor: channels, width, height
    """

    #gt_seg_img: torch tensor: channels, width, height
    flat_gt_seg_img = torch.flatten(gt_seg_img,0,1).numpy()
    colors, idx, = np.unique(flat_gt_seg_img, return_inverse=True, axis=0)
    assert (not np.any(np.all(colors == np.array((0,0,0)), axis=1))) # should be no 0,0,0 pixel values (in custom dataset)
    colors = colors.astype(int)
    idx = np.reshape(idx, gt_seg_img.shape[:2])
    
    attention_output = []
    attention_alignment = [num_slots-1]*(num_slots)

    unique_color_idxs = []
    corresponding_ucidx = []
    for color_idx, color in enumerate(colors):
        for uc_idx, uniq_color in enumerate(unique_colors):
            if((color==uniq_color).all()):
                unique_color_idxs.append(color_idx)
                corresponding_ucidx.append(uc_idx)

    # create a binary channnel for each distinct label/object/color found
    for k in range(len(unique_color_idxs)):
        attention_output.append(np.where(idx==unique_color_idxs[k], 1, 0))

    # append channels of all zeros for channels that are not represented (ie, when
    # number of distinct objects is less than number of slots)
    if(len(attention_output)<num_slots):
        diff = num_slots-len(attention_output)
        for i in range(diff):
            attention_output.append(np.zeros(idx.shape))

    # decoding alignment from colours 
    # TODO: decoding hardcoded for 4 maps; make it general - Tushar
    # so this segment is basically hardcoding the mapping of orig RGB color to the ordering of the channels
    for i in range(len(corresponding_ucidx)):
        if(corresponding_ucidx[i]==8):
            attention_alignment[-1]=i
        elif(corresponding_ucidx[i]==9):
            attention_alignment[0]=i
        elif(corresponding_ucidx[i]==10):
            attention_alignment[1]=i
        elif(corresponding_ucidx[i]==11):
            attention_alignment[2]=i

    # the for loop here should order the channels according to the above hardcoded
    # ordering (ie, such that background will always be last channel)
    attention_output = np.stack([attention_output[i] for i in attention_alignment])

    if(len(attention_output)>num_slots):
        print("error in attention map parsing; more unique colors than number of slots")

    return torch.tensor(attention_output, dtype=torch.float)

def plot_cameras(cam):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for j,d in enumerate(cam):
        for i,c in enumerate(d):
            c= c.numpy()[0]
            ax.scatter(c[0],c[1],c[2])
            ax.text(c[0], c[1], c[2], str(j))
            ax.quiver(c[0], c[1], c[2], c[3]*c[5], c[4]*c[5], c[6], length = 0.1, normalize = True)
        
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


def show_image_pair(frames):
    rows = len(frames) 
    cols = 2 
    figsize = [8,8]
    
    fig, ax = plt.subplots(nrows = rows, ncols = cols, figsize = figsize)
    
    # for i, axi in enumerate(ax.flat):
    for j, frs in enumerate(frames):
        for k, fr in enumerate(frs):
            img = fr.numpy()[0]
            ax[j][k].imshow(img)
            ax[j][k].set_title("set: "+str(j)+" img: "+str(k))

