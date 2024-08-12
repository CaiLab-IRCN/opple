''' 
    Config NOTE: This config is the default config loaded for all experiments. Ideally
    each of the values given here will be overwritten when the config provided at the cmd
    line is merged. It is important however for this default config to have a one-to-one
    mapping for each hyperparamter defined here and for each defined in config to be merged
'''

# my_project/config.py
from yacs.config import CfgNode as CN

_C = CN()

_C.DATASET = CN()
_C.DATASET.root_dir_path = "/media/data4/john/scene_understanding/datasets/multi_obj_mvmt/regenerate/" 
_C.DATASET.dataset_train_path = _C.DATASET.root_dir_path + "train/" # This is the path to the dir containing the the dir of input images and the pickled datamap file (for training)
_C.DATASET.dataset_val_path = _C.DATASET.root_dir_path + "validation/" # This is the path to the dir containing the the dir of input images and the pickled datamap file (for validation)
_C.DATASET.dataset_test_path = _C.DATASET.root_dir_path + "test/" # This is the path to the dir containing the the dir of input images and the pickled datamap file (for testing)
_C.DATASET.images_dir_name = "Camera_Images"
_C.DATASET.pickled_datamap_filename = "data_map.pickle" # Path to the pickle file containing the triplet mapping data

_C.DATASET.fov = [90, 90]
_C.DATASET.frame_size = [128, 128] # The size that the input images will be resized to
_C.DATASET.cam_rot_lim = 30.0
_C.DATASET.reweight = False
_C.DATASET.zlim = 5.0
_C.DATASET.reverse_data = True # This will reverse train and val triplets with probability of 50% for each triplet

_C.EXPERIMENT = CN()
_C.EXPERIMENT.experiment_name = "DEFAULT"
_C.EXPERIMENT.exp_type = "multimoving" # choices=["multimoving", "gqn", "waymo", "movi"]
_C.EXPERIMENT.mlflow_dir = "/home/jday/scene_understanding/mlruns/"
_C.EXPERIMENT.depth_weights = "tbd"
_C.EXPERIMENT.imagination_weights = "tbd"
_C.EXPERIMENT.segmentation_weights = "tbd"

_C.TEST = CN()
_C.TEST.batch_size = 8
_C.TEST.num_batches_test =  500 # number of batches to test and run evaluation on
_C.TEST.num_workers = 8 # If multiprocessing is used for test runs, the ordering of the output will be random

_C.TRAIN = CN()
_C.TRAIN.random_seed = 300
_C.TRAIN.batch_size = 20
_C.TRAIN.num_workers = 8
_C.TRAIN.rbf_sigma = [0.4, 3]
_C.TRAIN.num_slots = 3
_C.TRAIN.num_slots_withbg = _C.TRAIN.num_slots+1
_C.TRAIN.epochs = 55
_C.TRAIN.segment_id_size = 64
_C.TRAIN.segment_char_size = 10
_C.TRAIN.segment_use_rnn =  True
_C.TRAIN.test_train_split = 0.9
_C.TRAIN.val_train_split = 0.05
_C.TRAIN.learning_rate = 3e-4
_C.TRAIN.imagnet_learning_rate = _C.TRAIN.learning_rate
_C.TRAIN.depthnet_learning_rate = _C.TRAIN.learning_rate
_C.TRAIN.segnet_learning_rate = _C.TRAIN.learning_rate
_C.TRAIN.warpnet_learning_rate = 3e-4
_C.TRAIN.random_sampling = True # This determines whether Dataloader will draw samples randomly/shuffle (currently, this hyperparam is also used for test dataloader)

# visualization parameters
_C.TRAIN.output_dir = "../outputs/"
_C.TRAIN.visualization_in_every = 10000
_C.TRAIN.plotting_in_every = 2000
_C.TRAIN.print_in_every = 200
_C.TRAIN.save_in_every = 1 #epoch
_C.TRAIN.eval_num_batches = 40 #NOTE: total num samples evaluated = TRAIN.batch_size * TRAIN.eval_num_batches

_C.VALIDATION = CN()
_C.VALIDATION.visualization_in_every = 250
_C.VALIDATION.eval_num_batches = 40 #NOTE: total num samples evaluated = TRAIN.batch_size * VALIDATION.eval_num_batches

_C.FLAGS = CN()
_C.FLAGS.depth_truth =  False
_C.FLAGS.segmentation_truth =  False
_C.FLAGS.do_segmentation =  True
_C.FLAGS.obj_truth =  False
_C.FLAGS.obj_pose =  False

_C.DEPTH = CN()
_C.DEPTH.depth_softmax_beta =  1.0
_C.DEPTH.epsilon =  1e-22
_C.DEPTH.epsilon_depth = 1e-5

_C.REGULARIZE = CN()
_C.REGULARIZE.spatial_smoothing = 0.0
_C.REGULARIZE.object_localign_reg = 1.0

_C.ROTATION = CN()
_C.ROTATION.vms_prior_conc =  0.2 # vonmises_prior_concentration
_C.ROTATION.vms_interp_conc =  100.0 # vonmises_interpolation_concentration
_C.ROTATION.max_yaw =  180.0

_C.ROTATION.num_bins =  120

_C.TRAIN.num_bins = _C.ROTATION.num_bins

_C.SEGMENTATION = CN()
_C.SEGMENTATION.loc_polar =  True
_C.SEGMENTATION.seq_mask =  False
_C.SEGMENTATION.skip_in_higher =  True
_C.SEGMENTATION.latent_midlayer =  256
_C.SEGMENTATION.contrastive_loss_pose =  False
_C.SEGMENTATION.output_size_latent = _C.TRAIN.num_slots * (3 + _C.ROTATION.num_bins +\
        _C.TRAIN.segment_id_size)

_C.IMAGINATION = CN()
_C.IMAGINATION.input_loc = False
_C.IMAGINATION.imagination_params = 7

_C.DEPTHNET = CN()
_C.DEPTHNET.n_channels = 3 # input channels
_C.DEPTHNET.n_classes = 1 # output channels
_C.DEPTHNET.bilinear = False
_C.DEPTHNET.eps = 0.01
_C.DEPTHNET.depth_range = 7.2
_C.DEPTHNET.special_nonlin = False
_C.DEPTHNET.mu = 4.0
_C.DEPTHNET.sigma = 2.0
_C.DEPTHNET.input_size = _C.DATASET.frame_size

_C.SEGNET = CN()
_C.SEGNET.n_channels = 3 # RGB input channels; 4 if depth is given as input
_C.SEGNET.n_classes = 1 # output channels - one for each object
_C.SEGNET.bilinear = False
_C.SEGNET.eps = 0.01
_C.SEGNET.special_nonlin = False
_C.SEGNET.mu = 4.0
_C.SEGNET.sigma = 2.0
_C.SEGNET.intermediate_latent_split = 0.0
_C.SEGNET.input_size = _C.DATASET.frame_size
_C.SEGNET.output_size_latent = _C.SEGMENTATION.output_size_latent
_C.SEGNET.use_RNN = _C.TRAIN.segment_use_rnn
_C.SEGNET.latent_midlayer = _C.SEGMENTATION.latent_midlayer
_C.SEGNET.skip_in_higher = _C.SEGMENTATION.skip_in_higher
_C.SEGNET.use_norm = True
_C.SEGNET.denomin = 8
_C.SEGNET.loc_lim = 7.2

_C.IMAGNET = CN()
_C.IMAGNET.n_channels = 4 # RGB input channels; 4 if depth is given as input
_C.IMAGNET.n_classes = 5 # output channels - one for each object
_C.IMAGNET.bilinear = False #NOTE: was True before
_C.IMAGNET.eps = 0.01
_C.IMAGNET.special_nonlin = False
_C.IMAGNET.depth_range = _C.DEPTHNET.depth_range
_C.IMAGNET.skip_in_higher = True
_C.IMAGNET.denomin = 8

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`
