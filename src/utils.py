import numpy as np
import torch
import os, copy, sys
import torch
import torch.nn.functional as F
import torch_sparse
from einops import rearrange
import yaml
import shutil
from yacs.config import CfgNode as CN
import pickle, dill
from scipy.spatial.transform import Rotation

def check_and_create_dirs(path):
    isExist = os.path.exists(path)
    if not isExist:
      # Create a new directory because it does not exist
      os.makedirs(path)

def create_train_directories(cfg, params):
    # create the folders for saving outputs during training
    check_and_create_dirs(params["exp_dir_path"] ) 
    check_and_create_dirs(os.path.join( params["exp_dir_path"], "train_viz/"))
    check_and_create_dirs(os.path.join( params["exp_dir_path"], "train_viz/imm_results/"))
    check_and_create_dirs(os.path.join( params["exp_dir_path"], "train_viz/visualizations/"))
    check_and_create_dirs(os.path.join( params["exp_dir_path"], "train_viz/dumped_outputs/"))
    check_and_create_dirs(os.path.join( params["exp_dir_path"], "val_viz/"))
    check_and_create_dirs(os.path.join( params["exp_dir_path"], "val_viz/imm_results/"))
    check_and_create_dirs(os.path.join( params["exp_dir_path"], "val_viz/dumped_outputs/"))
    check_and_create_dirs(os.path.join( params["exp_dir_path"], "plots/"))
    check_and_create_dirs(os.path.join( params["exp_dir_path"], "checkpoints/"))

def create_test_directories(cfg, params):
    # create the folders for saving outputs during testing
    check_and_create_dirs(params["test_dir_path"] ) 
    check_and_create_dirs(os.path.join( params["test_dir_path"], "imm_results/"))
    check_and_create_dirs(os.path.join( params["test_dir_path"], "visualizations/"))
    check_and_create_dirs(os.path.join( params["test_dir_path"], "dumped_outputs/"))
    check_and_create_dirs(os.path.join( params["test_dir_path"], "evals_" + params['run_time']  + "/"))

# Used to log a snapshot of the codebase at runtime 
def copy_codebase(destination_dir, codebase_dir="./"):
    if (os.path.abspath(codebase_dir).split('/')[-1] != "src"):
        print("ERROR: Path given for copying codebase seems to be incorrect. Path given:",
            os.path.abspath(codebase_dir))
        sys.exit("Exitting...")
    if os.path.exists(destination_dir):
        print("ERROR: Destination directory for copying codebase already exists. Path given:",
            os.path.abspath(destination_dir))
        sys.exit("Exitting...")
    shutil.copytree(codebase_dir, destination_dir, ignore=shutil.ignore_patterns('*mlruns*', '*misc_johns*', '*__pycache__*'))
    print("Finished copying codebase.")

def numpify_dict(outputs):
    for key in outputs.keys():
        if(torch.is_tensor(outputs[key])):
            outputs[key] = numpify(outputs[key])
            continue
        if(isinstance(outputs[key], tuple)):
            list_from_tup = []
            for key_in in range(len(outputs[key])):
                if(torch.is_tensor(outputs[key][key_in])):
                    list_from_tup.append(numpify(outputs[key][key_in]))
                    # outputs[key][key_in] = numpify(outputs[key][key_in])
            outputs[key] = list_from_tup
    return outputs

def dump_outputs(cfg, params, outputs, epoch, batch, phase):
    #NOTE: 'phase' should be set to one of ["train", "valid", "test"]
    # scene_num = outputs['X_updated']['scene_num']

    if phase == 'train':
        h5_path = os.path.join(params["exp_dir_path"],
            "train_viz/dumped_outputs/viz_ep:{}_batch:{}.pkl".format(epoch, batch))
    elif phase == 'valid':
        h5_path = os.path.join(params["exp_dir_path"],
            "val_viz/dumped_outputs/viz_ep:{}_batch:{}.pkl".format(epoch, batch))
    elif phase == 'test':
        h5_path = os.path.join(params["test_dir_path"],
            "dumped_outputs/viz_ep:{}_batch:{}.pkl".format(epoch, batch))
    else:
        sys.exit("ERROR: 'dump_outputs' function. Exitting...")

    hf = open(h5_path, 'wb')
    dill.dump(outputs, hf)
    hf.close()

def save_model(cfg, params, model, epoch=None):
    modelweights_out_path = params["exp_dir_path"] + "checkpoints/epoch{}.pt".format(epoch)
    torch.save(model.state_dict(), modelweights_out_path)
    if "_final" in str(epoch):
        print("Final model weights saved to:\n\t", modelweights_out_path)

# save model details and training details
def save_model_details(cfg, params, model, losses, loss_X, epoch):
    save_model(cfg, params, model, epoch)
    loss_XY = [losses, loss_X]
    with open(params["exp_dir_path"] + 'losses.pkl', 'wb') as f:
        pickle.dump(loss_XY, f)

# Writes the state of the config and params varaibles to file to save within experiment directory
def save_config_and_params(cfg, params, test=False):
    # save the config file
    if test: stream = open(params["test_dir_path"] + "config.yaml", "w")
    else: stream = open(params["exp_dir_path"] + "config.yaml", "w")
    cfg.dump(stream=stream)
    # save config in yacs format as well
    if params['cfg_to_load_path'] is not None:
        if test: shutil.copyfile(params['cfg_to_load_path'], params["test_dir_path"] + "config.py")
        else: shutil.copyfile(params['cfg_to_load_path'], params["exp_dir_path"] + "config.py")

    # save the params file
    if test: stream = open(params["test_dir_path"] + "params.yaml", "w")
    else: stream = open(params["exp_dir_path"] + "params.yaml", "w")
    params_copy = numpify_dict(copy.deepcopy(params))
    yaml.dump(params_copy, stream=stream)

# Extract experiment folder's name from full folder path
def extract_exp_dir_name(exp_dir_path):
    # this assumes that the path is in the following format:
    # example: .../scene_understanding/outputs/debug_20221226_150826/
    return (exp_dir_path.split('/')[-2])

# Extract experiment folder from checkpoint filepath
def extract_exp_dir_path(checkpoint_filepath):
    # this assumes that the path is in the following format:
    # example: .../scene_understanding/outputs/debug_20221226_150826/checkpoints/epoch0.pt
    return "/".join(checkpoint_filepath.split('/')[:-2]) + "/"

# Extract epoch number from checkpoint filepath; returns type int
def extract_epoch_num(checkpoint_filepath):
    # this assumes that the path is in the following format:
    # example: .../scene_understanding/outputs/debug_20221226_150826/checkpoints/epoch0.pt
    # returns -1 if the checkpoint is saved as initial checkpoint (ie, 'epoch_initial.pt')
    if "initial" in checkpoint_filepath: return -1
    if "epoch_final" in checkpoint_filepath: split_str = "_final."
    else: split_str = "."
    return int(checkpoint_filepath.split("epoch")[-1].split(split_str)[0])

def get_most_recent_modelweights():
    ''' This is a helper utility for restarting training on a model that did not finish 
    training. If this function is called, it is assumed that it is being called within
    the saved codebase snapshot in the original Exp output dir and that the 'checkpoints'
    folder is located at '../checkpoints/', relative to where this python file was executed.
    
    It will return the "most recent" model weight ckpt file found, where "most recent" is
    defined as being the ckpt filename with the highest epoch number.

    If only initialy saved weights are found, or if "final" weights are found, this function
    will exit the runtime.
    '''

    # for restarting training, make sure that code is being run from a snapshot directory
    executables_dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    if "codebase_snapshot_training" not in executables_dir_path.split('/'):
        input("WARNING: It does not seem like the code is being run from within the codebase snapshot" +
        "dir of an Exp output dir. Check requirements for running w/ '--restart_training' flag for " +
        "more info.\nPress ENTER to continue...")

    # get checkpont folder
    related_ckpt_folder = os.path.join(executables_dir_path, "../../", "checkpoints/")

    # get most recent checkpoint (defined by epoch number)
    most_recent_ckpt = -1
    
    for ckpt_file_name in os.listdir(related_ckpt_folder):
        if "final" in ckpt_file_name:
            print("ERROR: Attempting to restart training of a model that finished",
            "training. If you wish to train this model further, please start a new",
            "Experiment Run.")
            sys.exit("Exitting...")
        if "initial" in ckpt_file_name: continue
        if "interrupt" in ckpt_file_name: continue
        
        if most_recent_ckpt < extract_epoch_num(ckpt_file_name):
            most_recent_ckpt = extract_epoch_num(ckpt_file_name)
    
    if most_recent_ckpt == -1:
        print("ERROR: You tried to continue training a model that did not finish training",
        "the first epoch -- either only initial weights or no weights were found.",
        "If you want to restart training with epoch_initial weights, please set",
        "the --modelweights cmd line argument explicitly.")
        sys.exit("Exitting...")
    
    most_recent_ckpt_file_path = os.path.join(related_ckpt_folder, f"epoch{most_recent_ckpt}.pt")
    most_recent_ckpt_file_path = os.path.abspath(most_recent_ckpt_file_path)

    return most_recent_ckpt_file_path

# Asks user for command line input to write description of experiment to a file in the provided folder path
# If 'test' is true, it will save the description uniquely from the default description  
def write_description(params, no_comment=False, appended_note = None, cmd_line_comment = None):
    if cmd_line_comment is not None:
        simple_description = cmd_line_comment
    elif no_comment:
        simple_description = "na"
    else:
        simple_description = input("\n(Optional) Enter a description for this experiment: \n\t")
    description = simple_description + "\n"
    print("")

    # log the command line argument
    description += "\ncommand line args:\n" + " ".join(sys.argv)

    # append relevant messages
    if params['test']:
        if appended_note is not None:
            description += "\ncheckpoint used for test:\n" + appended_note
        else:
            description += "\nno checkpoint was given for this test run (random initial weights used)\n"
    if params["restart_training"]:
        description += "\nThis Run was restarted at: " + params["run_time"]

    if params['test']: filename = os.path.join(params['test_dir_path'], "test_description.txt")
    elif params["restart_training"]: filename = os.path.join(params['exp_dir_path'], "restart_notice_"+params["run_time"]+".txt")
    else: filename = os.path.join(params['exp_dir_path'], "description.txt")
    
    desc_file = open(filename, "w")
    desc_file.write(description)
    desc_file.close()
    return simple_description

def save_cmd_for_train_restart_to_file(params, run_ID):
    ''' 
    Writes the cmd line argument that would be used for a generic restart of this
    training run. Simply for the sake of convenience.
    '''
    if params["restart_training"]: return

    path_to_train_code_snapshot = os.path.join(params['exp_dir_path'], "codebase_snapshot_training")
    restart_cmd_str_1 = "cd " + os.path.join(path_to_train_code_snapshot, "src")
    restart_cmd_str2 = \
        "PYTHONPATH=" + path_to_train_code_snapshot + f" python train.py --device {params['device']} --run_ID {run_ID} --restart_training"
    description = "## approximate cmds to restart this training run:\n" \
        + restart_cmd_str_1 + "\n" + restart_cmd_str2

    filename = os.path.join(params['exp_dir_path'], "restart_train_cmd.txt")
    desc_file = open(filename, "w")
    desc_file.write(description)
    desc_file.close()

def ensure_cfg_key_equivalency(default_cfg, path_to_cfg_to_load, force_config=False):
    """ Checks to make sure the set of keys between configs is the same. If key sets are
        not equivalent, and 'force_config' flag is not set, this will exit the program.

        Note: 
        - If there are extra keys present in the default config, it probably means that
        the current code has changed from the code that generated the config that you are
        attempting to load.
        - If there are extra keys present in the loaded config, it likely means that
        those extra keys/parameters are no longer used in the current code (but may not
        have been used in the previous code either)
        - If you want to ensure strict reproducibility, ensure that any new keys will
        not change the behavior of the code before setting the 'force_config'
    """

    # Get key set of current code's default config
    cfg_subkeys = []
    for key in default_cfg.keys():
        for cfg_subkey in default_cfg[key].keys():
            key_subkey_pair = key + " : " + cfg_subkey
            cfg_subkeys.append(key_subkey_pair)

    # Get key set of loaded yaml config
    new_cfg_subkeys = []
    with open(path_to_cfg_to_load, "r") as f:
        new_cfg = CN.load_cfg(f)
        for key in new_cfg.keys():
            for new_cfg_subkey in new_cfg[key].keys():
                key_subkey_pair = key + " : " + new_cfg_subkey
                new_cfg_subkeys.append(key_subkey_pair)

    # Check if there are any extra keys in the current default config
    extra_cfg_subkeys = []
    for cfg_subkey in cfg_subkeys:
        if cfg_subkey not in new_cfg_subkeys:
            extra_cfg_subkeys.append(cfg_subkey)
    if len(extra_cfg_subkeys) > 0:
        print("WARNING: At least one key present in the default config was not found in the loaded yaml:")
        print("\t", extra_cfg_subkeys)
        if force_config:
            print("'force_config' flag was set to True. Will continue with the new keys in default config...")
        else:
            print("If this is acceptable, please rerun with the 'force_config' flag set to True.")
            print("Exitting...")
            exit(1)

    # Check if there are any extra keys in loaded yaml config
    extra_new_cfg_subkeys = []
    for new_cfg_subkey in new_cfg_subkeys:
        if new_cfg_subkey not in cfg_subkeys:
            extra_new_cfg_subkeys.append(new_cfg_subkey)
    if len(extra_new_cfg_subkeys) > 0:
        print("WARNING: At least one key present in the loaded yaml was not found in the default config:")
        print("\t", extra_new_cfg_subkeys)
        print("This cannot be forced currently. Please remove these extra keys and then rerun.")
        # TODO: write code to automatically delete/ignore these extra keys
        print("Exitting...")
        exit(1)

# Edits the given run's mlflow yaml file and redirects where MLflow will look for artifacts to the given folder 
def redirect_mlflow_artifact_path(params, mlflow_folder, EXPERIMENT_ID, RUN_ID):
        desired_path_to_artifacts = "file://"+params['exp_dir_path']
        yaml_file_path = os.path.join(mlflow_folder, EXPERIMENT_ID, RUN_ID, "meta.yaml")
        with open(yaml_file_path) as f:
            yaml_doc = yaml.safe_load(f)
        for key in yaml_doc:
            if key == "artifact_uri":
                yaml_doc[key] = desired_path_to_artifacts
        with open(yaml_file_path, "w") as f:
            yaml.dump(yaml_doc, f)

# Convert the naming scheme of the pytorch weights so that they can be loaded into the network
# this is the weight for each network is loaded individually (the newer versions of the code should
# now be saving the entire model weights as one weight file, so this code may no longer be neccesary
# unless training from old weights)
def convert_weight_format(weights, type):
    if type == "imagination":
        weights_copy = copy.deepcopy(weights)
        for key in weights_copy.keys():
            newkey = key.replace("imagination_model.imag_net.imag_net.", "")
            weights[newkey] = copy.deepcopy(weights[key])

        for key in weights_copy.keys():
            del weights[key]
        return weights

    elif type == "segmentation":
        weights_copy = copy.deepcopy(weights)
        for key in weights_copy.keys():
            newkey = key.replace("segmentation_model.seg_net.seg_net.", "")
            weights[newkey] = copy.deepcopy(weights[key])

        for key in weights_copy.keys():
            del weights[key]
        return weights

    elif type == "depth":
        weights_copy = copy.deepcopy(weights)
        for key in weights_copy.keys():
            newkey = key.replace("depth_model.depth_net.", "")
            weights[newkey] = copy.deepcopy(weights[key])

        for key in weights_copy.keys():
            del weights[key]
        return weights
    
    else:
        print("Error in loading weight conversion")
        exit(1)


def merge_dicts(dict1, dict2):
    return(dict1.update(dict2))

def label_maps_to_colour(masks, default_color_mapping=True):
    '''
        masks:
            np.ndarray of shape [batches x slots x height x width] or
            shape [slots x height x width]
        default_color_mapping:
            True for the coloring of our cutom dataset. For other datasets, other
            mappings might be desired. We use the other map definition here for some
            GQN and O3V data

        output:
            np.ndarray of shape [batches x height x width x RGB] orshape [batches x
            height x width x RGB]
    '''

    # if there is a batch dimension, adjust which dim to treat as the label dim
    label_dim = 0 if len(masks.shape)==3 else 1

    if default_color_mapping: 
        colors = np.asarray(
            [[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0], [0.8,0.6,0.0]]) # ['r', 'g', 'b', 'y]
    else:
        colors = np.asarray(
        [ [0.8,0.6,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0], [1.0,0.0,0.0] ]) # ['y', 'g', 'b', 'r']
    imgs = colors[np.argmax(masks, axis=label_dim)]
    return imgs

def numpify(tensor):
    return tensor.cpu().detach().numpy()

def calc_IJ(frame_size, device):
    I = np.tile(np.arange(frame_size[1], dtype=np.float32) - (frame_size[1] - 1) / 2, (frame_size[0], 1))
    J = np.tile(np.arange(frame_size[0], dtype=np.float32) - (frame_size[0] - 1) / 2, (frame_size[1], 1)).T
    IJ = torch.tensor(np.stack((I,J)), device=device)
    IJ = rearrange(IJ, 'ij h w -> 1 ij h w')
    return IJ

def calculate_weight_ratio(warping_weight, base_sum, device):
    one = torch.tensor([[[1.0]]], device = device)
    zero = torch.tensor([[[0.0]]], device = device)
    ratio = torch.max(torch.min(warping_weight, one), zero) * 0.99
    inv_ratio = one - ratio
    return ratio, inv_ratio

def predict_final_image(d_weight, i_weight, d_pred, i_pred):
    pred_img = i_weight[:, None, :, :] * i_pred + d_weight[:, None, :, :] * d_pred
    return pred_img

def predict_final_depth(d_weight, i_weight, d_pred, i_pred):
    pred_img = i_weight * i_pred + d_weight * d_pred
    return pred_img

def average_location(cfg, pos, attention):
    ### pos - b x w x h x 4; attention - b x s x w x h
    pos = pos.permute(0,3,1,2)
    pos_attweight = pos[:,None,:,:,:]*attention[:,:-1,None,:,:]
    pos_attweight = torch.sum(pos_attweight, (3,4))
    pos_attweight_norm =  pos_attweight / (torch.sum(attention[:,:-1], (2,3))\
            + cfg.DEPTH.epsilon)[:,:,None]
    return pos_attweight_norm ## shape - b x s-1 x 3

def latent_object_id_alignment(cfg, latent1, latent2):
    ## matching on the basis of id dist using KL divergence and MSE of spatial distance
    neg_id_dist = - torch.sum((latent2[:, :, None, 3+cfg.ROTATION.num_bins:3 + cfg.ROTATION.num_bins\
            + cfg.TRAIN.segment_char_size] - latent1[:, None, :, 3+cfg.ROTATION.num_bins:3 +\
            cfg.ROTATION.num_bins + cfg.TRAIN.segment_char_size]) ** 2, 3)
    # cosine is restricted to [-1, 1]. The relative difference 
    # may become too small after passing through softmax. So we use Euclidean distance.

    weight_logits = torch.nn.functional.log_softmax(neg_id_dist, dim=2)
    weights = torch.exp(weight_logits)
    return weights, weight_logits

def latent_object_id_alignment_withbg(cfg, latent1, latent2):
    """
    ## matching on the basis of id dist using KL divergence and MSE of spatial distance
    # This version add an additional code of zero for the background. The similarity
    # matrix is calculated also against this zero code. If an object is very similar
    # to the background code, then we also give a high weight to not moving (essentially
    # means the pixels should not move and should belong to the background
    """
    code1 = latent1[:, :, 3+cfg.ROTATION.num_bins:\
            3+cfg.ROTATION.num_bins+cfg.TRAIN.segment_char_size]
    code2 = latent2[:, :, 3+cfg.ROTATION.num_bins:\
            3+cfg.ROTATION.num_bins+cfg.TRAIN.segment_char_size]
    zeros = torch.zeros_like(code2[:,0,:]).unsqueeze(1)
    code1 = torch.cat((code1, zeros), 1)
    code2 = torch.cat((code2, zeros), 1)
    """
    # we consider the distance from the current code to zero code
    # zero code is represents the background. Although we don't explicitly
    # estimate the location and pose of the background, we assume background is static.
    # so we will weight the final estimation of object movement based on how 
    # close the code we are considering is to the background. 
    # Thus we keep a matching score which indicates 1 - p(object is background)
    # Notice that weights and weight_logits are still normalized.
    """
    neg_id_dist = - torch.sum((code2[:, :, None, :] - code1[:, None, :, :]) ** 2, 3)
    # cosine is restricted to [-1, 1]. The relative difference 
    # may become too small after passing through softmax. So we use Euclidean distance.
    weight_logits_all = torch.nn.functional.log_softmax(neg_id_dist, dim=2)
    weight_logits = torch.nn.functional.log_softmax(neg_id_dist[:,:-1,:-1], dim=2)
    weights = torch.exp(weight_logits)
    matching_score = torch.sum(torch.exp(weight_logits_all[:,:-1,:-1]), dim=2)
    return weights, weight_logits, matching_score

def weight_locpose(object_loc, object_pose, weights, weight_logits):
    weighted_object_loc = torch.matmul(weights, object_loc)
    weighted_object_pose = torch.logsumexp(weight_logits[:,:,:,None] + object_pose[:,None,:,:], 2)
    # pose is coded in log scale, so we use logsumexp to replace matrix multiplication.
    # and we normalize below.
    weighted_object_pose = torch.nn.functional.log_softmax(weighted_object_pose, 2)
    return weighted_object_loc, weighted_object_pose

def weight_code(cfg, latent, weights, weight_logits):
    object_loc = latent[:,:,:3]
    object_pose = latent[:,:,3:3+cfg.TRAIN.num_bins]
    object_id = latent[:,:,3+cfg.TRAIN.num_bins:]
    weighted_object_loc = torch.matmul(weights, object_loc)
    weighted_object_id = torch.matmul(weights, object_id)
    weighted_object_pose = torch.logsumexp(weight_logits[:,:,:,None] +\
            object_pose[:,None,:,:], 2)
    # pose is coded in log scale, so we use logsumexp to replace matrix multiplication.
    # and we normalize below.

    weighted_object_pose = torch.nn.functional.log_softmax(weighted_object_pose, 2)
    weighted_latent = torch.cat((weighted_object_loc, weighted_object_pose,\
            weighted_object_id), 2)
    return weighted_latent

def calc_correction_ratios_warping(IJ, frame_size, fovy, dev):
    fovy_rad = (fovy/180)*np.pi
    h = frame_size[0]
    w = frame_size[1]
    f = (h/2)/np.tan(fovy_rad/2)
    t = (((1/f)* IJ[:,0,:,:]) **2 \
        + ((1/f)*IJ[:,1,:,:])**2 + torch.tensor(1.0,device=dev)) ** -0.5
    x_correction_ratio = (1/f)* IJ[:,0,:,:]
    y_correction_ratio = (1/f)* IJ[:,1,:,:]
    return (f, t, x_correction_ratio, y_correction_ratio)

def coordshift(obj, cam, device):
    new_obj = []
    new_obj_transformed = []
    origin = torch.zeros(cam[0].shape, device = device)
    origin[:, 4]=1
    origin[:, 5]=1
    origin[:, 7]=1

    new_obj_transformed = torch.ones([obj[0].shape[0], obj[0].shape[1], 4], device = device)
    for i in range(len(obj)):
        mat = camera_move_mat_pixel(origin, cam[i], device)
        new_obj_transformed[:, :, 0:3] = obj[i]
        new_obj.append(torch.matmul(mat[:,None,:,:], new_obj_transformed[:,:,:,None])[:,:,:3,0])
    return new_obj

def coordshift_pose(obj, cam, device):
    new_obj = []
    new_obj_transformed = []

    for i in range(obj.shape[0]):
        camposes = cam[i][:, 3:]
        new_obj_transformed = torch.ones(obj[0].shape, device = device)
        for k in range(obj[0].shape[1], 2):
            new_obj_transformed[:, k] = obj[:, k]*camposes[:,k] + obj[:, k+1]*camposes[:,k+1]
            new_obj_transformed[:, k+1] = obj[:, k+1]*camposes[:,k] - obj[:, k]*camposes[:,k+1]
        new_obj.append(new_obj_transformed)
    return new_obj

def update_sigma(sigma):
    sum_inv_sigma = np.sum([1 / s for s in self.sigma])
    sigma_weight = [1/s/sum_inv_sigma for s in self.sigma]
    return sigma, sigma_weight

def rbf_base(sigmas, frame_size, IJ): 
    print(sigmas)
    rbf_diff = IJ[:, :, :, :, None, None] - IJ[:, :, None, None, :, :]
    rbf_diff_pow = rbf_diff.pow(2)
    rbf_dist2 = torch.sum(rbf_diff_pow, dim=1)
    # rbf_dist should be w*h*w*h. The first two dimensions are predicted image coordinate.
    # The last two dimensions coorespond to the grid of the image to be drawn
    
    sum_inv_sigmas = np.sum([1/s for s in sigmas])
    sigmas_weight = [1/s/sum_inv_sigmas for s in sigmas]
    weighting = [] 
    for i_s, sigma in enumerate(sigmas):
        weighting.append(torch.sum(torch.exp(-(rbf_dist2/sigma**2) / 2) / sigma * sigmas_weight[i_s], dim = (1,2)))
        # we ignore the devision by sqrt(2pi)
    if len(weighting) > 1:
        weighting = torch.stack(weighting, dim=0)
        weighting = torch.sum(weighting, dim=0)
    else:
        weighting = weighting[0]
    return weighting

def camera_self_motion_yawonly(cam1, cam2, device=None):
    # works on tensors
    # This code calculates the camera's ego-motion, but only the x,y coordinate translation and yaw angle
    # In our data, camera only moves horizontally and rotate in yaw direction.
    # https://www.wikiwand.com/en/Rotation_matrix
    # https://math.stackexchange.com/questions/709622/relative-camera-matrix-pose-from-global-camera-matrixes
    ## camera cordinates: x - right, y - away , z - upwards
    ## the input is assumed to hold camera tensor (1-d) with element 0-2 being xyz and 3-4 being cos and sin of yaw.
    ## Returns: torch.tensor of shape [batch x 3]

    def _rz(a,b):
        return torch.stack([torch.stack([a, b], 1),
                            torch.stack([-1*b, a], 1)], 1)

    R1z = _rz(cam1[:,3], cam1[:,4])
    # r90 = _rz(torch.tensor([0.0], device=device),
    #           torch.tensor([-1.0], device=device)) # this matrix adds 90 degree to a vector's angle
    T21 = cam2[:,:2] - cam1[:,:2] # camera movement in world coordinate
    T21_camera = R1z @ T21[:,:,None]
    # T21_camera = R1z @ r90 @ T21[:,:,None]
    T21_camera = T21_camera.squeeze(-1)
    
    cos_angle_z = cam2[:,3] * cam1[:,3] + cam2[:,4] * cam1[:,4]
    sin_angle_z = cam2[:,4] * cam1[:,3] - cam1[:,4] * cam2[:,3]
    cos_angle_z = torch.min(torch.max(cos_angle_z, torch.tensor([-1.0], device=device)),
                            torch.tensor([1.0], device=device))
    angle_z = torch.acos(cos_angle_z) * torch.sign(sin_angle_z)
    # Positive value in angle_z means the camera turns left.
    motion_param = torch.cat([T21_camera, angle_z[:,None]], dim=1)
    return motion_param

def object_rotation_uponly(cos, sin, device=None):
    # works on tensors
    # This code calculates the rotation matrix of pixels of an object relative to the center
    # of the object, given a tensor of the cosine and sine of the object rotation degree.
    # Notice the effect should be adding this rotation angle to the original polar
    # angle of the pixel. This is different from rotating camera, because when rotating camera,
    # pixels rotate to the opposite direction
    # https://www.wikiwand.com/en/Rotation_matrix
    # https://math.stackexchange.com/questions/709622/relative-camera-matrix-pose-from-global-camera-matrixes

    ## camera corrdinates: x - right, y - away , z - upwards
    ## required coordinates: x - right, y - down, z - away

    # cos and sin are tensor of size batch x object

    rot_matrix = torch.eye(4, device=device).repeat(cos.shape[0], cos.shape[1], 1, 1)
    rot_matrix[:,:,:2,:2] = torch.stack([torch.stack([cos, -sin], -1),torch.stack([sin, cos], -1)], -2)

    return rot_matrix

def get_rotation_matrix(angles, camquat, device, order='ZXY'):
    '''
    angles -- torch tensor of shape (batch, 6) where the 6 elements represent
              rotations around each axis such that:
              [cos(Z-axis), sin(Z-axis), cos(X-axis), sin(X-axis), cos(Y-axis), sin(Y-axis)]
    returns:
    R      -- a torch tensor of shape (batch, 3, 3)

    Description 
    Get a 3x3 rotation matrix from euler angles or quaternion
    This rotation matrix, when applied to a vector, can be thought to 
    transform a coord represented in the Camera's coord system to that
    coord rerpresented in the World coordinate system

    Assumptions:
    - the order of rotations are INTRINSIC 'ZXY'
    - NOTE: although the order of rotations are ZXY, our custom implementation may
        represent a differnt order of calculations and still get the expected result
        due to the fact that the signs of our component matrices may be different (from
        that of the scipy implementation, for example) as well as the fact that the movi
        dataset (which was the one considered when making this code) never has rotation
        around the Y-axis
    '''

    ## Custom Implementation:
    # Create constants for Zero and One tensors
    if len(angles.size()) > 1: #if there is a batch dimesion
        batch_size = angles.size()[0]
        ZERO = torch.zeros((batch_size,1), device=device)
        ONE = torch.ones((batch_size,1), device=device)
    else:
        ZERO = torch.zeros((1), device=device)
        ONE = torch.ones((1), device=device)
    
    cos_Z, sin_Z, cos_X, sin_X, cos_Y, sin_Y = torch.split(angles, 1, dim=-1)
    
    # rotation around Z-axis
    Z = torch.stack((torch.cat((cos_Z, sin_Z, ZERO), dim=-1),
                        torch.cat((-sin_Z, cos_Z, ZERO), dim=-1),
                        torch.cat((ZERO, ZERO, ONE), dim=-1)),
                        dim=-1)

    # rotation around X-axis
    X = torch.stack((torch.cat((ONE, ZERO, ZERO), dim=-1),
                        torch.cat((ZERO, cos_X, sin_X), dim=-1),
                        torch.cat((ZERO, -sin_X, cos_X), dim=-1)),
                        dim=-1)
    
    # rotation around Y-axis
    Y = torch.stack((torch.cat((cos_Y, ZERO, -sin_Y), dim=-1),
                        torch.cat((ZERO, ONE, ZERO), dim=-1),
                        torch.cat((sin_Y, ZERO, cos_Y), dim=-1)),
                        dim=-1)

    R_custom = Z @ X @ Y

    R = R_custom
    return R

def get_F_matrix(R, cam_coords, device):
    '''
    R           -- torch tensor of shape (batch, 3, 3)
    cam_coords  -- torch tensor of shape (batch, 3) where
                   the 3 elements are x, y, z coords/translation
    returns:
    F           -- torch tensor of shape (batch, 4, 4)

    Desciption:
    Returns a transformation matrix which represents both rotation and translation
    by combining a the [4x4] rotation matrix R and the 3x1 translation vector T 
    (ie, cam coords) in the following way:
            [ R  T ]
            [ 0  1 ]
    '''

    # Create the constant for the last row of the matrix
    LAST_ROW = torch.tensor((0.,0.,0.,1.), device=device)
    batch_size = cam_coords.size()[0]
    LAST_ROW = LAST_ROW[None,:].repeat(batch_size,1) #repeat over the batch dimension

    F = torch.cat( [torch.cat((R, cam_coords[:,:,None]), dim=2),
                                LAST_ROW[:,None,:]    ], dim=1)
    
    return F

def get_Q_matrix(cam1, cam2, camquat1, camquat2, device):
    '''
    cam1, cam2  -- torch tensors of shape (batch, 9) where the 9 elements are:
        [x,y,z, cos(z-axis), sin(z-axis), cos(x-axis), sin(x-axis), cos(y-axis), sin(y-axis)]
    
    Description:
    This function will give a 4x4 matrix (which we refer to as Q), which when matrix multiplied with
    a 4x1 vector b_hat_1 with elements [x, y, z, w] (where x, y, z are an
    objects coords relative to the cam1 (and w always equals 1)), will give
    b_hat_2 which is the objects coords relative to cam2

    We calculate this in a few step process
    - We get the rotation matrix representations for the euler angles of the camera
    - Such a rotation matrix can be thought of as a conversion from a coordinate represented in that camera's coord system
      to a world coord system
    - Likewise, the inverse of such a matrix can be thought of as a conversion of a coord represented in the world coord system
      to the other particular coord system (ie, the coord system of the camera)
    - If we apply both in succesion (ie, R2_inv * R1, where R1 and R1 are the Rotation matrices of cam1 and cam2 respectively)
      then we obtain the matrix Q which transforms from a camera coordiante system to a new camera coord system
    
    - We first determine the Rotation matrices, and then determine the matrix F which represents
      both the rotations and translations of the camera

    '''
    
    # Get rotation matrix representation for the cam's euler angles
    R1 = get_rotation_matrix(cam1[:,3:], camquat1, device)
    R2 = get_rotation_matrix(cam2[:,3:], camquat2, device)

    # Create a new matrix, F, from the R matrices that also account for translation
    F1 = get_F_matrix(R1, cam1[:,:3], device)
    F2 = get_F_matrix(R2, cam2[:,:3], device)

    Q = F2.inverse() @ F1

    return Q

def camera_move_mat_pixel(cam1, cam2, device, camquat1=None, camquat2=None):
    # cam1 and cam2 are coded as 
    # [x,y,z, cos, sin, cos, sin, cos, sin]
    return get_Q_matrix(cam1, cam2, camquat1, camquat2, device)


def camera_to_world_coord(cam, latent, device):
    batch_size = cam.shape[0]
    def _ry(a,b):
        return torch.stack([torch.stack([a, torch.zeros_like(a, device=device), -1*b], 1),
                            torch.stack([torch.zeros_like(a, device=device), torch.ones_like(a, device=device),
                                         torch.zeros_like(a, device=device)], 1),
                            torch.stack([b, torch.zeros_like(a, device=device), a],1)], 1)
    

    def _rz(a,b):
        return torch.stack([torch.stack([a, b, torch.zeros_like(a, device=device)], 1),
                            torch.stack([-1*b, a, torch.zeros_like(a, device=device)], 1),
                            torch.stack([torch.zeros_like(a, device=device), torch.zeros_like(a, device=device),
                                         torch.ones_like(a, device=device)], 1)], 1)

    def _rx(a,b):
        return torch.stack([torch.stack([torch.ones_like(a, device=device), torch.zeros_like(a, device=device),
                                         torch.zeros_like(a, device=device)], 1),
                            torch.stack([torch.zeros_like(a, device=device), a, b], 1),
                            torch.stack([torch.zeros_like(a, device=device), -1*b, a], 1)], 1)

    Rx = _rx(torch.ones(batch_size, device=device), torch.zeros(batch_size, device=device))
    Ry = _ry(cam[:,5], cam[:,6])
    Rz = _rz(cam[:,3], cam[:,4])

    R = torch.eye(4, device=device).repeat(batch_size, 1, 1)
    R[:, :3, :3] = Rz@Ry@Rx
    r90 = torch.eye(4, device=device).repeat(batch_size, 1, 1) 
    r90[:, :3, :3] = _rz(torch.zeros(batch_size, device=device),
                         torch.ones(batch_size, device=device))
    T = torch.eye(4, device=device).repeat(batch_size, 1, 1) 
    T[:, :3, 3] = cam[:,:3]
    obj_pos = latent[:,:,:3]
    obj_pos_T = torch.ones((obj_pos.shape[0], obj_pos.shape[1], obj_pos.shape[2]+1), device=device)
    obj_pos_T[:,:,:3] = obj_pos

    Rworld = T@r90@R.transpose(1,2)

    obj_pos_world = Rworld[:,None,:,:]@obj_pos_T[:,:,:,None]
    return obj_pos_world[:,:,:3,0]


def world_to_camera_coord(cam, latent, device):
    batch_size = cam.shape[0]
    def _ry(a,b):
        return torch.stack([torch.stack([a, torch.zeros_like(a, device=device), -1*b], 1),
                            torch.stack([torch.zeros_like(a, device=device), torch.ones_like(a, device=device),
                                         torch.zeros_like(a, device=device)], 1),
                            torch.stack([b, torch.zeros_like(a, device=device), a],1)], 1)
    

    def _rz(a,b):
        return torch.stack([torch.stack([a, b, torch.zeros_like(a, device=device)], 1),
                            torch.stack([-1*b, a, torch.zeros_like(a, device=device)], 1),
                            torch.stack([torch.zeros_like(a, device=device), torch.zeros_like(a, device=device),
                                         torch.ones_like(a, device=device)], 1)], 1)

    def _rx(a,b):
        return torch.stack([torch.stack([torch.ones_like(a, device=device), torch.zeros_like(a, device=device),
                                         torch.zeros_like(a, device=device)], 1),
                            torch.stack([torch.zeros_like(a, device=device), a, b], 1),
                            torch.stack([torch.zeros_like(a, device=device), -1*b, a], 1)], 1)

    Rx = _rx(torch.ones(batch_size, device=device), torch.zeros(batch_size, device=device))
    Ry = _ry(cam[:,5], cam[:,6])
    Rz = _rz(cam[:,3], cam[:,4])

    R = torch.eye(4, device=device).repeat(batch_size, 1, 1)
    R[:, :3, :3] = Rz@Ry@Rx
    r90 = torch.eye(4, device=device).repeat(batch_size, 1, 1) 
    r90[:, :3, :3] = _rz(torch.zeros(batch_size, device=device),
                         torch.ones(batch_size, device=device))
    T = torch.eye(4, device=device).repeat(batch_size, 1, 1) 
    T[:, :3, 3] = cam[:,:3]
    obj_pos = latent[:,:,:3]
    obj_pos_T = torch.ones((obj_pos.shape[0], obj_pos.shape[1], obj_pos.shape[2]+1), device=device)
    obj_pos_T[:,:,:3] = obj_pos

    Rcam = T@r90.transpose(1,2)@R

    obj_pos_camera = Rcam[:,None,:,:]@obj_pos_T[:,:,:,None]
    return obj_pos_camera[:,:,:3,0]


def warping_sparse(frame1, frame_map, frame_grid, depth, device, params = {'depth_softmax_beta':5.0, 'epsilon':1e-22, 'epsilon_depth':1e-5}):

    h, w = frame_map[0].shape ## adding batchwise sparse matrix
    depth_softmax_beta = params['depth_softmax_beta']
    epsilon = params['epsilon']

    # project = frame_map * 4 
    project = frame_map

    source = frame1
    grids = frame_grid

    # assuming the picture has even number of pixels in each dimension
    project_left = torch.floor(project[0,:,:] + 0.5) + w / 2 
    project_right = project_left + 1
    # pixel space, so y inverted
    project_top = torch.floor(project[1,:,:] + 0.5) + h / 2
    project_bottom = project_top + 1

    # adding bounndary tensors and zero tensors for GPU instantiation
    boundarytop = torch.tensor(h+1, device=device)
    boundaryright = torch.tensor(w+1, device=device)
    zerotensor = torch.tensor(0, device=device)

    # pixel projection boundary conditions
    # target is padded by one. Anything that projects outside will be treated as projected to the padded border (which will be discarded)
    project_left = torch.maximum(torch.minimum(project_left, boundaryright), zerotensor)
    project_right = torch.maximum(torch.minimum(project_right, boundaryright), zerotensor)
    project_top = torch.maximum(torch.minimum(project_top, boundarytop), zerotensor)
    project_bottom = torch.maximum(torch.minimum(project_bottom, boundarytop), zerotensor)

    project_topleft = torch.flatten(project_left.long()) + torch.flatten(project_top.long()) * (w + 2) 
    project_topright = torch.flatten(project_right.long()) + torch.flatten(project_top.long()) * (w + 2) 
    project_bottomleft = torch.flatten(project_left.long()) + torch.flatten(project_bottom.long()) * (w + 2)
    project_bottomright = torch.flatten(project_right.long()) + torch.flatten(project_bottom.long()) * (w + 2)

    source_idx = torch.arange(0, w*h, dtype=torch.long, device=device)

    contrib_topleft = torch.abs((project_right - (w + 1) / 2 - project[0, :, :])
                                * (project_bottom - (h + 1) / 2 - project[1, :, :]))
    # strictly speaking we should not need to use absolute value. But for pixels projected outside,
    # the weights calculated this way may be negative. We are just slightly worried about any problem
    # with this even though eventually the projection to the padded area will be discarded
    contrib_bottomleft = torch.abs((project_right - (w + 1) / 2 - project[0, :, :])
                                   * (project[1, :, :] - project_top + (h + 1) / 2))
    contrib_topright = torch.abs((project[0, :, :] - project_left + (w + 1) / 2)
                                 * (project_bottom - (h + 1) / 2 - project[1, :, :]))
    contrib_bottomright = torch.abs((project[0, :, :] - project_left + (w + 1) / 2)
                                    * (project[1, :, :] - project_top + (h + 1) / 2))


    sparse_idx_topleft = torch.cuda.LongTensor(torch.stack([project_topleft, source_idx]))
    sparse_idx_topright = torch.cuda.LongTensor(torch.stack([project_topright, source_idx]))
    sparse_idx_bottomleft = torch.cuda.LongTensor(torch.stack([project_bottomleft, source_idx]))
    sparse_idx_bottomright = torch.cuda.LongTensor(torch.stack([project_bottomright, source_idx]))

    contrib_sparse_topleft = torch.sparse.FloatTensor(sparse_idx_topleft, contrib_topleft.flatten(), [(w + 2) * (h + 2), w * h])
    contrib_sparse_topright = torch.sparse.FloatTensor(sparse_idx_topright, contrib_topright.flatten(), [(w + 2) * (h + 2), w * h])
    contrib_sparse_bottomleft = torch.sparse.FloatTensor(sparse_idx_bottomleft, contrib_bottomleft.flatten(), [(w + 2) * (h + 2), w * h])
    contrib_sparse_bottomright = torch.sparse.FloatTensor(sparse_idx_bottomright, contrib_bottomright.flatten(), [(w + 2) * (h + 2), w * h])

    contrib = contrib_sparse_topleft + contrib_sparse_topright + contrib_sparse_bottomleft + contrib_sparse_bottomright

    mask_by_depth = F.relu(torch.sign(depth - params['epsilon_depth']))
    warp_ratio = torch.sparse.mm(contrib, mask_by_depth.flatten()[:, None])
    depth_weight = torch.exp(- depth_softmax_beta * depth) * mask_by_depth # a mask on negative depths, to stop them from contributing on pixels that can be seen
    depth_weighted_total_contrib = torch.sparse.mm(contrib, depth_weight.flatten()[:, None]) + epsilon
    depth_weighted_source = depth_weight * source
    depth_weighted_source = torch.flatten(depth_weighted_source.permute(1,2,0), start_dim=0, end_dim=1)
    depth_weighted_output = torch.sparse.mm(contrib, depth_weighted_source)

    depth_weighted_output = depth_weighted_output / depth_weighted_total_contrib
    depth_weighted_output = depth_weighted_output.reshape(h+2, w+2, 4)[1:-1, 1:-1, :].permute(2,0,1).contiguous()
    warping_weight = torch.reshape(warp_ratio, (h+2,w+2))[1:-1, 1:-1]  ## we need the mask on the weight as well if we get a bad depth value at some pixel

    return depth_weighted_output, warping_weight

def warping_sparse_batch_slotwise(cfg, frame1, frame_map, frame_grid, depth, attmaps, device):
    ### shapes: frame1 - b x s x c x w x h; frame_map = b x s x xy x w x h; frame_grid - b x xy x w x h;
    ### depth - b x s x w x h; attmaps - b x s x w x h;

    b, numslots, _, h, w = frame_map.shape ## adding batchwise sparse matrix
    depth_softmax_beta = cfg.DEPTH.depth_softmax_beta
    epsilon = cfg.DEPTH.epsilon

    project = frame_map
    source = frame1
    grids = frame_grid

    # assuming the picture has even number of pixels in each dimension
    project_left = torch.floor(project[:,:,0,:,:] + 0.5) + w / 2 
    project_right = project_left + 1
    # pixel space, so y inverted
    project_top = torch.floor(project[:,:,1,:,:] + 0.5) + h / 2
    project_bottom = project_top + 1

    # adding bounndary tensors and zero tensors for GPU instantiation
    boundarytop = torch.tensor(h+1, device=device)
    boundaryright = torch.tensor(w+1, device=device)
    zerotensor = torch.tensor(0, device=device)
    batchtensor = torch.flatten(torch.stack([btch*torch.ones((numslots,w,h), dtype=torch.long,
        device=device) for btch in range(b)])) ### flag

    # pixel projection boundary conditions
    # target is padded by one. Anything that projects outside will be treated as
    # projected to the padded border (which will be discarded)

    project_left = torch.maximum(torch.minimum(project_left, boundaryright), zerotensor)
    project_right = torch.maximum(torch.minimum(project_right, boundaryright), zerotensor)
    project_top = torch.maximum(torch.minimum(project_top, boundarytop), zerotensor)
    project_bottom = torch.maximum(torch.minimum(project_bottom, boundarytop), zerotensor)

    ## to get i,j idx in flat indexin it should be id = (j-1)*w + i
    project_topleft = torch.flatten(project_left.long()) + torch.flatten(project_top.long()) *\
            (w + 2) + batchtensor*(w+2)*(h+2)
    project_topright = torch.flatten(project_right.long()) + torch.flatten(project_top.long()) *\
            (w + 2) + batchtensor*(w+2)*(h+2)
    project_bottomleft = torch.flatten(project_left.long()) + torch.flatten(project_bottom.long())\
            * (w + 2) + batchtensor*(w+2)*(h+2)
    project_bottomright = torch.flatten(project_right.long()) +\
            torch.flatten(project_bottom.long()) * (w + 2) + batchtensor*(w+2)*(h+2)

    projection_idxs = torch.cat((project_topleft, project_topright, project_bottomleft,\
            project_bottomright))

    source_idx = torch.arange(0, b*numslots*w*h, dtype=torch.long, device=device)
    source_idxs = torch.cat((source_idx, source_idx, source_idx, source_idx))

    contrib_topleft = torch.abs((project_right - (w + 1) / 2 - project[:,:, 0, :, :])
                                * (project_bottom - (h + 1) / 2 - project[:, :, 1, :, :])) \
                                * attmaps
    # strictly speaking we should not need to use absolute value. But for pixels projected outside,
    # the weights calculated this way may be negative. We are just slightly worried about any problem with this
    # even though eventually the projection to the padded area will be discarded
    contrib_bottomleft = torch.abs((project_right - (w + 1) / 2 - project[:,:, 0, :, :])
                                   * (project[:,:, 1, :, :] - project_top + (h + 1) / 2)) \
                                   * attmaps
    contrib_topright = torch.abs((project[:,:, 0, :, :] - project_left + (w + 1) / 2)
                                 * (project_bottom - (h + 1) / 2 - project[:,:, 1, :, :])) \
                                 * attmaps
    contrib_bottomright = torch.abs((project[:,:, 0, :, :] - project_left + (w + 1) / 2)
                                    * (project[:,:, 1, :, :] - project_top + (h + 1) / 2)) \
                                    * attmaps

    contrib_sparse_topleft = torch_sparse.SparseTensor(row=project_topleft.flatten(),\
            col=source_idx, value=contrib_topleft.flatten(),\
            sparse_sizes=((w+2)*(h+2)*b, b*w*h*numslots))
    contrib_sparse_topright = torch_sparse.SparseTensor(row=project_topright.flatten(),\
            col=source_idx, value=contrib_topright.flatten(),\
            sparse_sizes=((w+2)*(h+2)*b, b*w*h*numslots))
    contrib_sparse_bottomleft = torch_sparse.SparseTensor(row=project_bottomleft.flatten(),\
            col=source_idx, value=contrib_bottomleft.flatten(),\
            sparse_sizes=((w+2)*(h+2)*b, b*w*h*numslots))
    contrib_sparse_bottomright = torch_sparse.SparseTensor(row=project_bottomright.flatten(),\
            col=source_idx, value=contrib_bottomright.flatten(),\
            sparse_sizes=((w+2)*(h+2)*b, b*w*h*numslots))

    contributions = torch.cat((contrib_topleft.flatten(), contrib_topright.flatten(),\
            contrib_bottomleft.flatten(), contrib_bottomright.flatten()))

    sparse_idxs = torch.stack([projection_idxs, source_idxs]).to(device=device)

    idxs_coa, contrib_coa = torch_sparse.coalesce(sparse_idxs, contributions,\
            m=b*(w+2)*(h+2), n=numslots*b*w*h, op="add")
    contrib = torch_sparse.SparseTensor(row=idxs_coa[0], col=idxs_coa[1],\
            value=contrib_coa, sparse_sizes=(b*(w+2)*(h+2), b*numslots*w*h))

    mask_by_depth = F.relu(torch.sign(depth-cfg.DEPTH.epsilon_depth))
    
    warp_ratio = contrib @ mask_by_depth.flatten()[:,None]

    depth_e = depth * mask_by_depth ## can have negative values; remove before exponentiation
    depth_weight = torch.where(depth<=0, torch.zeros(depth.shape, device=device), torch.exp(-
        depth_softmax_beta * depth_e) * mask_by_depth)
    depth_weighted_total_contrib = (contrib @ depth_weight.flatten()[:,None]) + epsilon

    if(torch.isnan(depth_weight).any()):
        print("depth weight has nan")
    if(torch.isnan(depth_weighted_total_contrib).any()):
        print("depth weighted contrib is nan")

    depth_weighted_source = depth_weight[:, :, None, :, :] * source[:, :, :, :, :]
    depth_weighted_source = torch.flatten(depth_weighted_source.permute(0,1,3,4,2), start_dim=0,
            end_dim=3) ### b x slot x c x w x h -> b x s x w x h x c -> b*s*w*h x c
    depth_weighted_output = contrib @ depth_weighted_source 

    depth_weighted_output = depth_weighted_output / depth_weighted_total_contrib
    depth_weighted_output = depth_weighted_output.reshape(b,\
            h+2, w+2, 4)[:,1:-1, 1:-1,:].permute(0,3,1,2)
    warping_weight = torch.reshape(warp_ratio, (b, h+2,w+2))[:,1:-1, 1:-1]


    return depth_weighted_output, warping_weight

# function for curating data
def data_curation(cfg, params, data_batch):
    '''
    Description:
    This function organizes and processes ('curates') what is immediatly returned from the dataloader. 
    - changes ordering of axis (ie, batch and triplet dimension)
    - does some stuff with object and camera motion involving matrix calculation
    '''

    img, cam, camquats, obj, depth_gt, segmentation, objposes, objdeets, scenedeets, scene_num = data_batch
    device = params['device']

    if(img[0].shape[0]==0):
        print("shape 0 tensor found from the loader", img[0].shape[0])
        # continue
    
    ## Change ordering of axis
    ### img Input:  Batch x Triplet x Width x Height x Channels (float32)
    ### img Output: Triplet x Batch x Channels x Width x Height (float32)
    img = img.permute(1,0,4,2,3).contiguous().to(device, torch.float32, non_blocking=True) 
    ### cam Input:  Batch x Triplet x 9
    ### cam Output: Triplet x Batch x 9
    cam = cam.permute(1,0,2).to(device, torch.float32, non_blocking=True) 
    ### obj Input:  Batch x Triplet x num_objs x 3 {position of center of mass}
    ### obj Output: Triplet x Batch x num_objs x 3 {position of center of mass}
    obj = obj.permute(1,0,2,3).to(device, torch.float32, non_blocking=True) 
    ### obj_pose Input:  Batch x Triplet x num_objs x 6 (cos and sin of yaw, pitch and roll)
    ### obj_pose Output: Triplet x Batch x num_objs x 6 (cos and sin of yaw, pitch and roll)
    obj_pose = objposes.permute(1,0,2,3).to(device, torch.float32, non_blocking=True)
    ### camquats Input:  Batch x Triplet x 4
    ### camquats Output: Triplet x Batch x 4
    camquats = camquats.permute(1,0,2).to(device, torch.float32, non_blocking=True)

    if torch.isnan(cam).any() or torch.isinf(cam).any():
        print('there is anomaly in camera parameter')

    ### convert object shift in position from world coordinates to camera coordinates
    obj = coordshift(obj, cam, device)
    obj = torch.stack(obj, dim=0)
    objposes = coordshift_pose(obj_pose, cam, device)
    obj_pose = []
    for i in range(len(objposes)):
        pose_t = torch.acos(objposes[i][:, 0])*torch.sign(objposes[i][:,1])
        ob = torch.zeros((objposes[0].shape[0],cfg.ROTATION.num_bins), device = device)
        for k in range(objposes[i].shape[0]):
            ob[k, torch.floor(pose_t[k]%10).type(torch.long)] = 10  
        obj_pose.append(ob)
    obj_pose = torch.stack(obj_pose, dim=0)

    ### ground truth depth for checking
    depth_gt = depth_gt.permute(1,0,2,3).to(device, non_blocking = True).contiguous()
    segmentation_gt = segmentation.permute(1,0,2,3,4).to(device, non_blocking=True)

    m = None
    motion_params1 = None
    motion_params2 = None

    if not params['perframe']:
        # view transform matrix
        m_cam1_to_cam2 = camera_move_mat_pixel(cam[0], cam[1], device, camquats[0], camquats[1])
        m_cam2_to_cam3 = camera_move_mat_pixel(cam[1], cam[2], device, camquats[1], camquats[2])
        m = torch.stack([m_cam1_to_cam2, m_cam2_to_cam3], dim=0)

        motion_params1 = camera_self_motion_yawonly(cam[0], cam[1], device=device)
        motion_params2 = camera_self_motion_yawonly(cam[1], cam[2], device=device)

    data_batch_curated = {
            'frames':img,
            'cams':cam,
            'camquats':camquats,
            'obj_pos_gt':obj,
            'obj_pose_gt':obj_pose,
            'depths_gt':depth_gt,
            'att_maps_gt':segmentation_gt,
            'transformation_matrices':m,
            'motion_params1':motion_params1,
            'motion_params2':motion_params2,
            'scene_num': scene_num
        }
    return data_batch_curated

def fix_depth_multi_obj(cfg, depth_in, loc_lim):
    ### depth in shape - b x w x h
    farplane = 100.0
    denom = 100.0/loc_lim
    depth_out = (depth_in*(cfg.DATASET.farplane - cfg.DATASET.nearplane) +\
            cfg.DATASET.nearplane)/denom 
    depth_out_log = torch.log(depth_out + 1e-5)
    return depth_out, depth_out_log

def fix_depth_gqn(depth_in):
    ### depth in shape - b x w x h
    farplane = 100.0
    nearplane = 0.05
    denom = 1.0
    depth_in /= (255.0/100.0)
    depth_out = depth_in
    depth_out_log = torch.log(depth_out + 1e-5)
    return depth_out, depth_out_log

# function for curating data after depth estimation to make sure we update the depth in
# the data if we are not using the GT data. Calculates log depth and selects the target
# depth to be used for the experiment
def data_batch_depth_update(cfg, params, data_batch, depth_outputs):
    data_batch['depth_log'] = []
    if cfg.FLAGS.depth_truth:
        data_batch['depth'] = data_batch['depths_gt'].unsqueeze(2)
        for i in range(data_batch['depth'].shape[0]): #for each frame in triplet
            
            ## For multimoving experiments:
            if cfg.EXPERIMENT.exp_type == 'multimoving':
                _, log_depth = fix_depth_multi_obj(cfg, data_batch['depths_gt'][i],\
                cfg.SEGNET.loc_lim)

            ## For gqn experiments:
            elif cfg.EXPERIMENT.exp_type == 'gqn':
                _, log_depth = fix_depth_gqn(data_batch['depths_gt'][i])

            ## For waymo experiments:
            elif cfg.EXPERIMENT.exp_type == 'waymo':
                print("\nWARNING: Waymo is using gqn's fix_depth_gqn function. Ensure this is the desired behavior.\n")
                _, log_depth = fix_depth_gqn(data_batch['depths_gt'][i])
                
            else:
                print(f"ERROR: Unrecognized exp_type '{cfg.EXPERIMENT.exp_type}'. Exitting.")
                exit(1)

            data_batch['depth_log'].append(log_depth.unsqueeze(1))
    else:
        data_batch['depth'] = depth_outputs['depth_smoothed']
        for i in range(data_batch['depth'].shape[0]): #for each frame in triplet
            log_depth = torch.log(depth_outputs['depth_smoothed'][i]+1e-5)
            data_batch['depth_log'].append(log_depth)

    data_batch['depth_log'] = torch.stack(data_batch['depth_log'], dim=0)

    return data_batch

# common function for predicting next frame for the multi object experiment
def predict_next_frame(cfg, params, data_batch, depth_outputs,\
        segment_outputs, imag_outputs, warping_outputs):
    """
    Prediction of the frame 3 from the assumed motion and estimated warping and imagined outputs
    get the warping mask from warping output -> get the warped images -> add them to the imagined
    image -> get the prediction of the next image
    """
    # get the warped image
    warped_img_frame3 = warping_outputs['frame3_predicted']
    depth3p_warped = warping_outputs['depth3_warped']

    # get imagination image
    imag_img_frame3 = imag_outputs['img3_imag']
    depth3p_imag = imag_outputs['depth3_imag']

    # get the warping weight
    warping_weight = warping_outputs['weighting_matrix']
    weighting_base = warping_outputs['weighting_base']

    # put together the predicted image
    d_weight, i_weight = calculate_weight_ratio(warping_weight, weighting_base,\
        params['imag_device'])
    
    # Below 2 lines are used for the warping albation experiment
    # d_weight = torch.zeros_like(d_weight)
    # i_weight = torch.ones_like(i_weight)

    predicted_img3 = predict_final_image(d_weight, i_weight, warped_img_frame3, imag_img_frame3)
    predicted_img3_warp = warped_img_frame3 * d_weight[:, None, :, :]
    predicted_img3_imag = imag_img_frame3 * i_weight[:, None, :, :]

    predicted_depth3 = predict_final_depth(d_weight, i_weight, depth3p_warped, depth3p_imag)

    outputs = {
            'predicted_img3':predicted_img3,
            'predicted_img3_warp':predicted_img3_warp,
            'predicted_img3_imag':predicted_img3_imag,
            'predicted_depth3': predicted_depth3,
            'warping_weight':d_weight,
            'imag_weight':i_weight
        }

    return outputs

