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
import torch
import argparse
from tqdm import tqdm

# local imports
import src.utils as utils
import src.plotting_utils as plotting_utils
from src.loss import compute_losses_multi_obj
from train import Opple_model, set_dataloaders
import evaluate
from configs.default_cfg import get_cfg_defaults 


class Test_Model():
    def __init__(self, cfg, params):
        self.cfg = cfg
        self.params = params

    # def test function caller
    def test_opple(self, test_dataloader, model):
        print("TESTING")
        # set up training loop from the training dataloader
        outputs = {}
        losses = None
        loss_X = []
        epoch = 0

        print("\ttest loader length: ", len(test_dataloader))
        print(f"\tWill run testing on {self.cfg.TEST.num_batches_test} number of batches")
        
        tqdm._instances.clear()
        pbar = tqdm(total=self.cfg.TEST.num_batches_test)
        pbar.set_description(f"Testing progress")
        pbar.set_postfix(loss=float('Inf'))
    
        model.eval() # puts model in 'evaluation' mode
        with torch.no_grad():
            for batch_num, data_batch in enumerate(test_dataloader):
                if(batch_num >= self.cfg.TEST.num_batches_test): break

                # get data from the dataloader and curate the data
                data_batch_curated = utils.data_curation(self.cfg,\
                        self.params, data_batch)

                # call model, get outputs, plots, and gradients
                outputs = model(data_batch_curated)

                # calculate loss (if not using single-frame input)
                if not self.params['perframe']:
                    loss = compute_losses_multi_obj(self.cfg, self.params,\
                        outputs['X_updated'], outputs)
                
                # output cleanup
                outputs = plotting_utils.numpify_all_outputs(outputs)
                
                if not self.params['perframe']:
                    for key in loss.keys():
                        loss[key] = utils.numpify(loss[key])
                    outputs['losses'] = loss
                
                    # loss val collection
                    if(losses is None):
                        losses = {}
                        for key in loss.keys():
                            losses[key] = []
                            losses[key].append(loss[key])
                    else:
                        for key in loss.keys():
                            losses[key].append(loss[key])

                    loss_X.append(epoch*len(test_dataloader)+batch_num)

                utils.dump_outputs(self.cfg, self.params, outputs, epoch, batch_num, phase='test')
                
                if not params['perframe'] and (batch_num % 250 == 0):
                    plotting_utils.plot_immediate_results_test(self.cfg, self.params, outputs,
                        epoch, batch_num)
                    
                pbar.update(1)
                # at the end of batch --

            # at the end of epoch/test set --
            pbar.close()

# def parameters and config maker for each experiment
def parse_args(parser):
    parser.add_argument("--device", "-dev", help="arguments should either be 'cpu' or\
    'cuda:X' where X is the gpu number", required=True)
    parser.add_argument("--config", "-c", help="config yaml file for setup", default=None)
    parser.add_argument("--use_original_config", "-og_cfg", help="use cfg associated with experiment where weights are loaded from", action="store_true")
    parser.add_argument("--force_config", "-fg", help="force merging between cfgs when default cfg has extra keys", action="store_true")
    parser.add_argument("--modelweights", "-mw", help="opple model weight files in .pt format")
    parser.add_argument("--anomaly", "-a", action="store_true", help="set the torch gradient\
            anomaly to true")
    parser.add_argument("--no_comment", "-nc", help="skip asking for experiment desciption input", action="store_true")
    parser.add_argument("--new_exp_dir", help="do not automatically re-use experiemnt folder, instead create a new one", action="store_true")
    parser.add_argument("--perframe", help="run test one frame at a time, instead of using triplets", action="store_true")
    parser.add_argument("--non_presplit", action="store_true", help="if set, will split dataset into train/val/test during runtime.")
    parser.add_argument("--deep_warping", action="store_true", help="toggle on to train deep-warping version of the model")

    args=parser.parse_args()
    return args

def add_params(cfg, args):
    params = {}
    params['model_weighloc'] = args.modelweights
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
    params['test'] = True
    params["restart_training"] = False
    params["perframe"] = args.perframe
    params["non_presplit"] = args.non_presplit
    params["deep_warping"] = args.deep_warping
    return params

if(__name__ == "__main__"):
    start_time = time.time()
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    # Load default/base config
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
        input("\nWARNING: Are you sure you want to continue without loading a specific cfg file?\
            \nPress ENTER to continue...")
    assert not( (args.use_original_config) and (args.config is not None) ), \
        "Choose to load config automatically OR at cmd, not both."
    if cfg_to_load_path is not None:
        utils.ensure_cfg_key_equivalency(cfg_local, cfg_to_load_path, args.force_config)
        cfg_local.merge_from_file(cfg_to_load_path)
        print("Default config merged succesfully with loaded config:", cfg_to_load_path, "\n")
    cfg_local.freeze()

    # Set random seeds (for reproducability)
    torch.manual_seed(cfg_local.TRAIN.random_seed)
    random.seed(cfg_local.TRAIN.random_seed)
    np.random.seed(cfg_local.TRAIN.random_seed)
    torch.cuda.manual_seed(cfg_local.TRAIN.random_seed)
    torch.cuda.manual_seed_all(cfg_local.TRAIN.random_seed)

    params = {}
    params_more = add_params(cfg_local, args)
    utils.merge_dicts(params, params_more)
    params['cfg_to_load_path'] = cfg_to_load_path  

    # Set the experiment directory
    if args.new_exp_dir:
        print("\nWill create a new Experiment directory for this testing run")
        params['experiment_name'] = cfg_local.EXPERIMENT.experiment_name
        params["exp_dir_path"] = cfg_local.TRAIN.output_dir + params['experiment_name']+ "_"+\
            params['run_time']+"/"
    else:
        print("\nWill create testing directory within Experiment directory associated with loaded weights.")
        params["exp_dir_path"] = utils.extract_exp_dir_path(args.modelweights)
        if not os.path.exists(params["exp_dir_path"]):
            print("Checkpoint's experiment directory not found. Path:", params["exp_dir_path"])
            sys.exit("exiting")

    # Set the unique id for this test run and create directories for saving output
    test_id = "test_" + params['run_time']  + "/"
    params["test_dir_path"] = os.path.join(params["exp_dir_path"], test_id)
    utils.create_test_directories(cfg_local, params)
    print("All testing output will be saved within this directory:\n\t" , params["test_dir_path"])

    ## Log meta information about this run
    # Save an optional description for this run
    description = utils.write_description(params, args.no_comment, appended_note=args.modelweights)
    # Log the state of the cfg and params values to be used
    utils.save_config_and_params(cfg_local, params, test=True)
    # Create and log a copy of the codebase at runtime
    print("Saving snapshot of current state of codebase...")
    destination_dir = os.path.join(params["test_dir_path"], "codebase_snapshot_testing/src")
    codebase_dir = os.path.dirname(os.path.abspath(sys.argv[0])) # get path of dir in which the executed file is within
    utils.copy_codebase(destination_dir=destination_dir, codebase_dir=codebase_dir)

    # Set dataloader and get parameters
    dataloaders = set_dataloaders(cfg_local, params)

    # Create a model using the config and params
    model = Opple_model(cfg_local, params)

    # Load the weight dict for the model
    if(args.modelweights):
        state_dict = torch.load(args.modelweights, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.to(params['device'])
    else:
        if input("WARNING: No model weights loaded. Are you sure you want to test with random weights? ('Y' to continue...) ") != 'Y':
            sys.exit("exiting")
    
    tester = Test_Model(cfg_local, params)
    tester.test_opple(dataloaders['test_dataloader'], model)

    ## Evaluation
    if not params['perframe']:
        # Set paths for evaluation
        eval_folder = os.path.join(params["test_dir_path"], "evals_" + params['run_time']  + "/")
        utils.check_and_create_dirs(eval_folder)
        testing_pkl_file_loc = os.path.join(params["test_dir_path"], "dumped_outputs/")
        # Create figures for paper
        figure_folder = os.path.join(eval_folder, "figures/")
        utils.check_and_create_dirs(figure_folder)
        evaluate.get_figures_basic(testing_pkl_file_loc, figure_folder, batch_size=cfg_local.TEST.batch_size, batch_num=0, exp_type=cfg_local.EXPERIMENT.exp_type)
        # Call metric evaluations function
        log_path = os.path.join(eval_folder, "eval_log.txt")
        evaluate.calculate_metrics(testing_pkl_file_loc, log_path,cfg_local.EXPERIMENT.exp_type,
                batch_range=cfg_local.TEST.num_batches_test, batch_size=cfg_local.TEST.batch_size,
                multiprocesses=0, verbose=True, super_verbose = False)
        # Delete the used batch pkl files
        # for filename in os.listdir(testing_pkl_file_loc):
        #     os.remove(testing_pkl_file_loc + filename)

    print(f"(Elapsed time for testing: {time.time() - start_time:.2f} seconds)")


## Example command:
## > python test.py --device cuda:6 -modelweights /path/to/weights/checkpoints/epoch_final.pt 
