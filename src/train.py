''' train.py

    Decription: 
    This is the main script for training an Opple model. The basic way to interact with
    this script is to run it at the command line while providing a device ID and config.py file.

    Example:
        > python train.py --device cuda:1 --config my_config_01.py

    To further customize the run enter other command line arguments. Run with --help for
    more information on those options
        
    @Author: Mingbo Cai, Tushar Aurora, John Day
'''

import os, sys, time
# Add this working directory to PYTHONPATH 
if os.path.abspath('../') not in sys.path: #this assumes this file is being run from within scene_understanding/src/ or a snapshot of it
    sys.path.insert(0, os.path.abspath('../')) 
    print("*Added the following to front of PYTHONPATH: ", os.path.abspath('../'))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
from datetime import datetime
import random
from multiprocessing import set_start_method
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import argparse
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# import dataloaders
from src.dataloaders.data_utils import get_dataloader, bkwrd_compat_set_dataloaders
from dataloaders.dataloader_customdata_map import CustomDatasetMap
# from src.dataloaders.dataloader_customdata_map_gqn import CustomDatasetMapGQN
# from src.dataloaders.dataloader_customdata_map_waymo import CustomDatasetMapWaymo
from src.dataloaders.dataloader_customdata_map_movi import CustomDatasetMapMOVi

# import other modules
from src.depth_perception import Depth_Inference
from src.warping import Warping
from src.deep_warping import DeepWarping
from src.segmentation import Segmentation_Network, Segmentation
from src.imagination import Imagination_Network, Imagination

# import utils
import src.utils as utils
import src.plotting_utils as plotting_utils
from src.loss import compute_losses_multi_obj

# import default config
from configs.default_cfg import get_cfg_defaults 
import evaluate


class Opple_model(nn.Module):
    def __init__(self, cfg, params):
        super(Opple_model, self).__init__()
        self.cfg = cfg
        self.params = params

        # define depth inference model
        self.depth_model = Depth_Inference(cfg, params)

        # define warping model
        if params["deep_warping"]:
            self.warping_model = DeepWarping(cfg, params)
        else:
            self.warping_model = Warping(cfg, params)

        # define segmentation model
        segmentation_network = Segmentation_Network(cfg, params)
        self.segmentation_model = Segmentation(cfg, params, segmentation_network)

        # define imagination model
        imagination_network = Imagination_Network(cfg, params)
        self.imagination_model = Imagination(cfg, params, imagination_network)

        # setup optimizers
        self.setup_optimizers()

    def setup_optimizers(self):
        self.imagination_model_opt = torch.optim.Adam(self.imagination_model.parameters(),\
                lr=self.cfg.TRAIN.imagnet_learning_rate, eps=1e-6)
        self.depth_model_opt = torch.optim.Adam(self.depth_model.parameters(),\
                lr=self.cfg.TRAIN.depthnet_learning_rate, eps=1e-6)
        self.segmentation_model_opt = torch.optim.Adam(self.segmentation_model.parameters(),\
                lr=self.cfg.TRAIN.segnet_learning_rate, eps=1e-6)
        if self.params["deep_warping"]:
            self.warp_model_opt = torch.optim.Adam(self.warping_model.parameters(),\
                    lr=self.cfg.TRAIN.warpnet_learning_rate, eps=1e-6)

    def zero_grad_for_all(self):
        self.imagination_model_opt.zero_grad()
        self.depth_model_opt.zero_grad()
        self.segmentation_model_opt.zero_grad()
        if self.params["deep_warping"]:
            self.warp_model_opt.zero_grad()

    def optimizer_step_for_all(self):
        torch.nn.utils.clip_grad_value_(self.imagination_model.parameters(), 1.0)
        self.imagination_model_opt.step()
        torch.nn.utils.clip_grad_value_(self.depth_model.parameters(), 2.5)
        self.depth_model_opt.step()
        torch.nn.utils.clip_grad_value_(self.segmentation_model.parameters(), 1.0)
        self.segmentation_model_opt.step()
        if self.params["deep_warping"]:
            torch.nn.utils.clip_grad_value_(self.warping_model.parameters(), 1.0)
            self.warp_model_opt.step()

    def match_latent_obj_representations(self, segment_outputs):
        ## Identity computations
        # align all the latent representations and attention maps 
        latent_representation_transformed1 = segment_outputs[0]["latent_representation_transformed"]
        latent_representation_transformed2 = segment_outputs[1]["latent_representation_transformed"]
        alignment_weight12, alignment_weight_logit12, matching_score12 = \
            utils.latent_object_id_alignment_withbg(self.cfg, latent_representation_transformed1, latent_representation_transformed2)

        # slot 1 -> slot 2 alignment
        representation1match = utils.weight_code(self.cfg, latent_representation_transformed1, alignment_weight12,\
                alignment_weight_logit12)

        segment_outputs_updated = {
                'seg_out1':segment_outputs[0],
                'seg_out2':segment_outputs[1],
                'seg_out3':segment_outputs[2],
                'representation_aligned1':representation1match,
                'representation_aligned2':latent_representation_transformed2,
                'matching_score_12':matching_score12
            }
        return segment_outputs_updated

    def forward(self, X):        
        # collect the outputs from depth model 
        depth_outputs = self.depth_model(X)

        # set data values based on experiment and ground truths to be used
        X_updated = utils.data_batch_depth_update(self.cfg, self.params, X, depth_outputs)

        # get attention maps from the segmentation model
        segment_outputs = self.segmentation_model(X_updated) # list of len(triplet) is returned

        # if only using single frame as input, return seg and depth output and end forward pass here
        if self.params["perframe"]:
            outputs = {
                'X_updated':X_updated,
                'segment_outputs':{"seg_out1":segment_outputs[0]},
                'depth_outputs':depth_outputs,
            }
            return outputs
        
        segment_outputs = self.match_latent_obj_representations(segment_outputs)
        
        # get warping weights and warp the images
        warping_outputs = self.warping_model(X_updated, segment_outputs)

        # get imagined image from frame2 and frame3
        imag_outputs = self.imagination_model(X_updated, segment_outputs, warping_outputs)

        # put the depth, warped image, attention map, and imagined image together to predict frame3
        predicted_outputs = utils.predict_next_frame(self.cfg, self.params, X_updated,\
                depth_outputs, segment_outputs, imag_outputs, warping_outputs)

        # put together all the outputs and return
        outputs = {
                'X_updated':X_updated,
                'segment_outputs':segment_outputs,
                'depth_outputs':depth_outputs,
                'warping_outputs':warping_outputs,
                'imag_outputs':imag_outputs,
                'predicted_outputs':predicted_outputs
            }
        return outputs

class Train_Model():
    def __init__(self, cfg, params):
        self.cfg = cfg
        self.params = params

    # def train function caller
    def train_opple(self, train_dataloader, validation_dataloader, model):
        # set up training loop from the training dataloader
        outputs = {}
        losses = None
        loss_X = []
        Nth_last_batch = (len(train_dataloader) - self.cfg.TRAIN.eval_num_batches)
        eval_batch_range_list = list(range(Nth_last_batch, len(train_dataloader))) #ie, last N batches

        model.train() #puts model in 'train' mode
        for epoch in range(self.params['starting_epoch'], self.cfg.TRAIN.epochs):
            tqdm._instances.clear()
            pbar = tqdm(total=len(train_dataloader))
            pbar.set_description(f"Training epoch {epoch}")
            pbar.set_postfix(loss=float('Inf'))
            mlflow.log_metric("epoch", epoch) 
            
            avg_epoch_loss = []
            start_time = time.time()

            for batch_num, data_batch in enumerate(train_dataloader):

                # get data from the dataloader and curate the data
                data_batch_curated = utils.data_curation(self.cfg, self.params, data_batch)

                model.train() # put model back into train mode 
                # set gradient to zero
                model.zero_grad_for_all()

                # call model, get outputs, plots, and gradients
                outputs = model(data_batch_curated)

                # call loss function calculator and get loss
                loss = compute_losses_multi_obj(self.cfg, self.params,\
                    outputs['X_updated'], outputs)
                
                # backwards and update the model
                loss['total_loss'].backward()
                model.optimizer_step_for_all()
                    
                # save loss for each batch to calc avg epoch loss at end of epoch
                avg_epoch_loss.append(loss['total_loss'].item())

                # make and write plots figures at the frequency specified in the config
                if(batch_num%self.cfg.TRAIN.plotting_in_every==0) or (epoch == 0 and batch_num % 500 == 0)\
                    and (not self.params["testbench"]):
                    for key in loss.keys():
                        loss[key] = utils.numpify(loss[key])
                    if(losses is None):
                        losses = {}
                        for key in loss.keys():
                            losses[key] = []
                            losses[key].append(loss[key])
                    else:
                        for key in loss.keys():
                            losses[key].append(loss[key])

                    # losses.append(loss)
                    loss_X.append(epoch*len(train_dataloader)+batch_num)

                    outputs = plotting_utils.numpify_all_outputs(outputs)
                    outputs['losses'] = loss

                    plotting_utils.plot_immediate_results(self.cfg, self.params, outputs,
                            epoch, batch_num)

                # If doing train evaluation, dump the outputs of the specified number
                # of the last batches of this epoch (to be used for eval)
                if(batch_num >= Nth_last_batch):
                    if(torch.is_tensor(outputs['X_updated']['cams'])):
                        outputs = plotting_utils.numpify_all_outputs(outputs)
                        for key in loss.keys():
                            loss[key] = utils.numpify(loss[key])
                        outputs['losses'] = loss

                    utils.dump_outputs(self.cfg, self.params, outputs, epoch, batch_num, phase='train')
                
                pbar.set_postfix(loss=loss['total_loss'].item())
                pbar.update(1)
                # at the end of batch --

            # at the end of epoch --
            pbar.close()

            # calculate and report the average loss over the epoch
            avg_epoch_loss = np.mean(avg_epoch_loss)
            if not self.params["testbench"]:
                mlflow.log_metric("ep avg loss", avg_epoch_loss, step=epoch) # average loss over a given training epoch
            print(f"Training epoch {epoch} complete. Average loss over epoch was {avg_epoch_loss}")
            
            # save losses and model weights after each epoch
            if((epoch%self.cfg.TRAIN.save_in_every)==0) and not self.params["testbench"]:
                utils.save_model_details(self.cfg, self.params, model, losses, loss_X, epoch)
                plotting_utils.plot_losses(self.cfg, self.params, losses['total_loss'], loss_X, epoch)

            # Run metric evaluations for this training epoch
            if self.cfg.TRAIN.eval_num_batches > 0:
                print(f"\nRunning evaluation on training epoch {epoch} (using {self.cfg.TRAIN.eval_num_batches} batches)...")
                metrics = evaluate.eval_helper(self.cfg, self.params, epoch, "train", eval_batch_range_list, verbose=False)
                print("Train set evaluation finished:\n")
                print(f"Mean train ARI:\t\t\t", metrics["avg_ARI"])
                print(f"Mean train IoU:\t\t\t", metrics["avg_IoU"].item(), "\n")
                mlflow.log_metric("train ARI", metrics["avg_ARI"], step=epoch)
                mlflow.log_metric("train IoU", metrics["avg_IoU"].item(), step=epoch)

            # run validation on the validation set
            if not self.params["testbench"]:
                self.validation_opple(validation_dataloader, model, epoch)

        print(f"(Total elapsed time for training: (w num workers:{self.cfg.TRAIN.num_workers}): {time.time() - start_time:.2f} seconds)")

        return losses, loss_X, model

    def validation_opple(self, validation_dataloader, model, epoch):
        # set up training loop from the training dataloader
        outputs = {}
        losses = None
        loss_X = []
        avg_val_loss = []
        eval_batch_range_list = list(range(self.cfg.VALIDATION.eval_num_batches))

        print("\nvalidation loader length: ", len(validation_dataloader), "at epoch: ", epoch)
        
        model.eval() # puts model in 'evaluation' mode
        with torch.no_grad():
            tqdm._instances.clear()
            pbar = tqdm(total=len(validation_dataloader))
            pbar.set_description(f"Validation epoch {epoch}")
            pbar.set_postfix(loss=float('Inf'))

            for batch_num, data_batch in enumerate(validation_dataloader):
                # get data from the dataloader and curate the data
                data_batch_curated = utils.data_curation(self.cfg, self.params, data_batch)

                # call model, get outputs, plots, and gradients
                outputs = model(data_batch_curated)

                # call loss function calculator and get loss
                loss = compute_losses_multi_obj(self.cfg, self.params,\
                    outputs['X_updated'], outputs)
                
                avg_val_loss.append(loss['total_loss'].item())

                if(batch_num%self.cfg.VALIDATION.visualization_in_every==0 or batch_num<self.cfg.VALIDATION.eval_num_batches):
                    # output cleanup
                    outputs = plotting_utils.numpify_all_outputs(outputs)
                    for key in loss.keys():
                        loss[key] = utils.numpify(loss[key])
                    outputs['losses'] = loss
                    
                    if batch_num<self.cfg.VALIDATION.eval_num_batches:
                        utils.dump_outputs(self.cfg, self.params, outputs, epoch, batch_num, phase='valid')

                    if (batch_num%self.cfg.VALIDATION.visualization_in_every==0):
                        # loss val collection
                        if(losses is None):
                            losses = {}
                            for key in loss.keys():
                                losses[key] = []
                                losses[key].append(loss[key])
                        else:
                            for key in loss.keys():
                                losses[key].append(loss[key])

                        loss_X.append(epoch*len(validation_dataloader)+batch_num)

                        plotting_utils.plot_immediate_results_valid(self.cfg, self.params, outputs,
                                epoch, batch_num)
                        plotting_utils.plot_losses_valid(self.cfg, self.params, losses['total_loss'], loss_X, epoch)
            
                pbar.set_postfix(loss=loss['total_loss'].item())
                pbar.update(1)
                # at the end of val batch --

            # at the end of val epoch --
            pbar.close()

            avg_val_loss = np.mean(avg_val_loss)
            mlflow.log_metric("val avg loss", avg_val_loss, step=epoch)
            
            # Run metric evaluations for this validation epoch
            if self.cfg.VALIDATION.eval_num_batches > 0:
                print(f"Running evaluation on validation epoch {epoch} (using {self.cfg.VALIDATION.eval_num_batches} batches)...")
                metrics = evaluate.eval_helper(self.cfg, self.params, epoch, "val", eval_batch_range_list)
                print("Validation set evaluation finished:")
                print(f"Mean val ARI:\t\t\t", metrics["avg_ARI"])
                print(f"Mean val IoU:\t\t\t", metrics["avg_IoU"].item(), "\n")
                mlflow.log_metric("val ARI", metrics["avg_ARI"], step=epoch)
                mlflow.log_metric("val IoU", metrics["avg_IoU"].item(), step=epoch)            

# setting up the dataloaders
def set_dataloaders(cfg, params):
    if params["non_presplit"]: # for some backwards compatiblity for old dataloading version
        dataloader_class = CustomDatasetMap
        return bkwrd_compat_set_dataloaders(cfg, params, dataloader_class)

    images_dir_name = cfg.DATASET.images_dir_name
    pickled_datamap_filename = cfg.DATASET.pickled_datamap_filename

    # For multimoving experiments
    if cfg.EXPERIMENT.exp_type == 'multimoving':
        dataloader_class = CustomDatasetMap
    # # For GQN experiments
    # elif cfg.EXPERIMENT.exp_type == 'gqn':
    #     dataloader_class = CustomDatasetMapGQN
    # # For Waymo experiments
    # elif cfg.EXPERIMENT.exp_type == 'waymo':
    #     dataloader_class = CustomDatasetMapWaymo
    # For MOVi experiments
    elif cfg.EXPERIMENT.exp_type == 'movi':
        dataloader_class = CustomDatasetMapMOVi
    else:
        print(f"ERROR: Unrecognized exp_type '{cfg.EXPERIMENT.exp_type}'. Training/testing has not yet been configured for this type.")
        sys.exit("Exitting...")

    ## Initialize the Dataset
    if not params["test"]:
        dataset_train = dataloader_class(cfg, params, cfg.DATASET.dataset_train_path, images_dir_name, pickled_datamap_filename, split="train")
        dataset_val = dataloader_class(cfg, params, cfg.DATASET.dataset_val_path, images_dir_name, pickled_datamap_filename, split="val")
    else:
        dataset_test = dataloader_class(cfg, params, cfg.DATASET.dataset_test_path, images_dir_name, pickled_datamap_filename, split="test")

    sample_weights = None
    if cfg.DATASET.reweight:
        # sample_weights = dataset.get_sampling_weights()
        sys.exit("ERROR: DATASET.reweight not implemented in this version of the code")

    train_loader, val_loader, test_loader = None, None, None
    # Get dataloaders for train/val sets
    if not params["test"]:
        train_loader = get_dataloader(dataset=dataset_train, batch_size=cfg.TRAIN.batch_size, num_workers=cfg.TRAIN.num_workers, sample_weights=sample_weights, random_sampling=cfg.TRAIN.random_sampling, pin=True)
        val_loader = get_dataloader(dataset=dataset_val, batch_size=cfg.TRAIN.batch_size, num_workers=cfg.TRAIN.num_workers, sample_weights=sample_weights, random_sampling=cfg.TRAIN.random_sampling, pin=True)
        print("train set size: {} batches, validation set size: {} batches. (train batch size is {})".format(len(train_loader),\
                len(val_loader), cfg.TRAIN.batch_size))
    # Get dataloaders for test set
    else:
        test_loader = get_dataloader(dataset=dataset_test, batch_size=cfg.TEST.batch_size, num_workers=cfg.TEST.num_workers, sample_weights=sample_weights, random_sampling=cfg.TRAIN.random_sampling, pin=True)
        print("test set size: {} batches. (test batch size is {})".format(len(test_loader),\
                cfg.TEST.batch_size))

    return {'train_dataloader' : train_loader, 'validation_dataloader' : val_loader,
            'test_dataloader': test_loader}

# def parameters and config maker for each experiment
def parse_args(parser):
    parser.add_argument("--device", "-dev", help="arguments should either be 'cpu' or\
    'cuda:X' where X is the gpu number", required=True)
    parser.add_argument("--config", "-c", help="config yaml file for setup", default=None)
    parser.add_argument("--use_original_config", help="use cfg associated with experiment where weights are loaded from", action="store_true")
    parser.add_argument("--force_config", "-fg", help="force merging between cfgs when default cfg has extra keys", action="store_true")
    parser.add_argument("--modelweights", "-mw", help="opple model weight files in .pt format")
    parser.add_argument("--anomaly", "-a", action="store_true", help="set the torch gradient\
            anomaly to true")
    parser.add_argument("--no_comment", "-nc", help="skip asking for experiment desciption input", action="store_true")
    parser.add_argument("--restart_training", help="only use when intend to resuse original directory and code.\
                         Modelweights do not need to be specified, as they will be automatically found based on\
                         the EXP directory this code is run inside of. Refer to README.", action="store_true")
    parser.add_argument("--run_ID", help="if set will associate this Run with the existing mlflow Run with this ID.")
    parser.add_argument("--non_presplit", action="store_true", help="if set, will split dataset into train/val/test during runtime.")
    parser.add_argument("--deep_warping", action="store_true", help="toggle on to train deep-warping version of the model")

    args=parser.parse_args()
    return args

def add_params(cfg, args):
    params = {}
    params['modelweights'] = args.modelweights 
    params['device'] = args.device
    params['run_time'] = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    params['depth_device'] = args.device
    params['imag_device'] = args.device
    params['seg_device'] = args.device
    params['warp_device'] = args.device
    params['IJ'] = utils.calc_IJ(cfg.DATASET.frame_size,\
            device=params['device'])
    params['cam_correction'] = utils.\
            calc_correction_ratios_warping(params['IJ'],\
            cfg.DATASET.frame_size, cfg.DATASET.fov[1], params['device'])
    params['test'] = False
    params["starting_epoch"] = 0
    if args.modelweights: 
        params["starting_epoch"] = utils.extract_epoch_num(params['modelweights']) + 1
    params["restart_training"] = args.restart_training
    params["testbench"] = False # this is just used to ignore some lines during testbench runs
    params["perframe"] = False # this param should always be False for training
    params["non_presplit"] = args.non_presplit
    params["deep_warping"] = args.deep_warping
    return params

if(__name__ == "__main__"):
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    if args.restart_training:
        # Set variables for use of original config, load most recent weights, & use same folder
        print("\nRestarting training...")
        print("Training of the found weights will be continued in the associated Exp dir",
        "and with the associated original config")
        args.use_original_config = True

        # Get path to original weights
        if args.modelweights is None:
            # set modelweights to train from to be the checkpoint from the last finished epoch
            args.modelweights = utils.get_most_recent_modelweights()
        print("This is the model path that will be used: ", args.modelweights)
        
        # Find path to associated Experiment directory
        original_exp_dir_path = utils.extract_exp_dir_path(args.modelweights)
        if not os.path.exists(original_exp_dir_path):
            sys.exit("ERROR: Checkpoint's experiment directory not found. Path:", original_exp_dir_path)
        print("This is the associated Exp dir path found: ", original_exp_dir_path)
    else:
        # Sanity check if user accidentally ran from within a snapshot folder
        if os.path.dirname(os.path.abspath(sys.argv[0])).split('/')[-1] == "codebase_snapshot_training":
            print("WARNING: This executed file seems to be within a saved codebase snapshot dir",
            "however the '--restart_training' flag is not set.")
            input("If you are sure you want to continue, press 'ENTER'...")
        
    # Load in default/base Config 
    cfg_local = get_cfg_defaults()

    # Load config to merge
    cfg_to_load_path = None
    if args.use_original_config:
        cfg_to_load_path = os.path.join(utils.extract_exp_dir_path(args.modelweights), "config.yaml")
        print("\nWill MERGE config associated with original experiment. Located at:", cfg_to_load_path)
    elif args.config is not None:
        cfg_to_load_path = args.config
        print("\nWill MERGE config given at cmd line. Located at:", cfg_to_load_path)
    else:
        input("WARNING: Are you sure you want to continue without loading a specific config file?\
            \nPress ENTER to continue...")
    assert not( (args.use_original_config) and (args.config is not None) ), \
        "Choose to load config automatically OR at cmd, not both."
    if cfg_to_load_path is not None:
        utils.ensure_cfg_key_equivalency(cfg_local, cfg_to_load_path, args.force_config)
        cfg_local.merge_from_file(cfg_to_load_path)
        print("Default config merged succesfully with loaded config:", args.config, "\n")
    cfg_local.freeze()
    
    torch.autograd.set_detect_anomaly(args.anomaly)

    # Set random seeds (for reproducability)
    torch.manual_seed(cfg_local.TRAIN.random_seed)
    random.seed(cfg_local.TRAIN.random_seed)
    np.random.seed(cfg_local.TRAIN.random_seed)
    torch.cuda.manual_seed(cfg_local.TRAIN.random_seed)
    torch.cuda.manual_seed_all(cfg_local.TRAIN.random_seed)

    params = {}
    params_more = add_params(cfg_local, args)
    utils.merge_dicts(params, params_more)

    params['experiment_name'] = cfg_local.EXPERIMENT.experiment_name
    params['cfg_to_load_path'] = cfg_to_load_path

    # Set the directories used for saving outputs and associated data during training
    if args.restart_training: params["exp_dir_path"] = original_exp_dir_path
    else:
        params["exp_dir_path"] = cfg_local.TRAIN.output_dir + params['experiment_name']+ "_"+\
            params['run_time']+"/"
        print("Creating new directory for this Experiment run:\n", params["exp_dir_path"])
        utils.create_train_directories(cfg_local, params)

    # Save an optional description for this run
    if not args.no_comment:
        description = utils.write_description(params)
    else: description = "NA: no comment"

    # Create and log a copy of the codebase at runtime
    # NOTE: Assuming no changes to the codebase were saved between runtime start 
    # and the execution of the below line, the saved codebase snapshot should relfect
    # the exact state of the code being run
    if not args.restart_training:
        print("Saving snapshot of current state of codebase...")
        destination_dir = os.path.join(params["exp_dir_path"], "codebase_snapshot_training/src")
        codebase_dir = os.path.dirname(os.path.abspath(sys.argv[0])) # get path of dir in which the executed file is within
        utils.copy_codebase(destination_dir=destination_dir, codebase_dir=codebase_dir)

    # MLflow: set the Experiment/Run name for MLflow; if experiment name does not exist, it will be created)
    mlflow.set_tracking_uri(uri='file://'+cfg_local.EXPERIMENT.mlflow_dir)
    EXPERIMENT_NAME = params['experiment_name']
    ACTIVE_EXPERIMENT = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
    EXPERIMENT_ID = ACTIVE_EXPERIMENT.experiment_id
    RUN_NAME = params['experiment_name']+ "_" + params['run_time']

    # MLflow: Start MLflow run
    # Note: if args.run_ID is set, will re-start specified Run (and run_name will be ignored)
    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_id=args.run_ID, run_name=RUN_NAME) as ACTIVE_RUN:
        RUN_ID = ACTIVE_RUN.info.run_id

        utils.save_cmd_for_train_restart_to_file(params, RUN_ID)
        
        if args.run_ID is None: # if restarting run, don't overwrite tags and params
            MlflowClient().set_tag(RUN_ID, "mlflow.note.content", description)

            # MLflow: track/log all the hyper/parameters
            mlflow.log_param("run_time", params['run_time'])
            mlflow.log_param("description", description)
            mlflow.log_param("device", params['device'])
            if args.restart_training:
                mlflow.log_param("restarted_job", True)
            for key, values in cfg_local.items():
                for value in values:
                    mlflow.log_param(key+"."+value, values[value])

        # set dataloader and get parameters
        dataloaders = set_dataloaders(cfg_local, params)

        # create a model using the config and params
        model = Opple_model(cfg_local, params)

        # if provided, load model with specified pretrained weights
        if args.modelweights:
            state_dict = torch.load(args.modelweights, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.to(params['device'])
            print("Loaded weights from", args.modelweights)

        # save the intial model, configs, and params
        if not args.restart_training:
            utils.save_config_and_params(cfg_local, params)
            torch.save(model.state_dict(), params["exp_dir_path"] + "checkpoints/epoch_initial.pt")

        # MLflow: edit this run's yaml file to redirect artifact location to our custom location
        utils.redirect_mlflow_artifact_path(params, cfg_local.EXPERIMENT.mlflow_dir, EXPERIMENT_ID, RUN_ID)

        # start training the model with dataloader
        try:
            trainer = Train_Model(cfg_local, params)
            losses, loss_X, model = trainer.train_opple(dataloaders['train_dataloader'],\
                dataloaders['validation_dataloader'], model)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt: Saving current model weights and then exitting...\n")
            utils.save_model(cfg_local, params, model, epoch=datetime.now().strftime("%Y.%m.%d_%H.%M.%S") + "_interrupt")

        else:
            # save the final model
            utils.save_model_details(cfg_local, params, model, losses, loss_X, str(cfg_local.TRAIN.epochs-1) + "_final")

            # MLflow: track model
            mlflow.pytorch.log_model(model, "classifier")


## Example command:
# Simple:                                   ## python train.py --device cuda:2
# w/ optional args for config:              ## python train.py  --device cuda:2 --config ./configs/debug_config.py
# w/ optional args for training restart:    ## python train.py  --device cuda:2 --run_ID XXXXXXX --restart_training
#*NOTE: run_ID can be found in the url of the mlflow Run
#*NOTE: for this last example of restart_training, this is expected to be run directly in the code_snapshot folder.
