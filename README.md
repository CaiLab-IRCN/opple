# Scene understanding in natural scenes

### Aim of the project
Using the neuroscientific evidence on how babies learn to interact with the world around them,
we aim to develop an understandable Machine Learning algorithm for building an implicit world model.
We use a moving actor and moving objects in a simulated environment where we add complicated
artifacts and texutres. We call our method Object Perception by Predictive LEarning (OPPLE).

#### Milestones:
* Object permanence
* Better Latent representation
* Linear object motion prediction

#### Comparisons:
**Datasets on our algorithm:** GQN (with moving objects), CARLA

**Other models with our custom dataset:** GENESIS, MONET, O3V, and Slot Attention

---

### Code layout
The code has the following folders:
1. `src/` : contains the main executable files `train.py` and `test.py`
2. `src/models/` : used to store all the network model definitions
3. `src/dataloaders/` :  has the base and child classes for all the dataloaders along with the
   common dataloader utilities in the `data_utils.py` file.
4. `misc/` : it has all the previous codes and snippets which are not needed anymore but kept for
   now.
5. `externa/` : for all the comparisons with external models

The `src/` has the following files for our model:
1. `segmentation.py` : definition of the segmentation network and segmentation call
2. `imagination.py` : definition of the imagination network and call to the imagination network
3. `warping.py` : definition of the warping module and all the accompanying functions including the
   vonmises class
4. `utils.py` : contains all the individual functions important for the implementation 

### To train and test a model
#### Basic example of training a model from scratch:
* Ensure you are within the `src/` directory and then execute train.py, specifying a config file and gpu number
> python train.py --device cuda:1 --config ./configs/default_config.py 
* The following commands will execute a basic testing script on your model
> python test.py --device cuda:1 --modelweights /path/to/experiment/folder/Experiment_ID/checkpoints/epoch_final.pt  --use_original_config 

#### Note on restarting training using pre-existing weights:
The following describes how to restart training of a model that stopped before finshing.
* Locate and cd into the relevant pre-existing experiment's codebase_snapshot directory 
> cd /path/to/experiment/folder/experiment_ID/codebase_snapshot_training/src
* Execute the training script from within the codebase_snapshot_training/src directory with the following command:
> PYTHONPATH=/path/to/experiment/folder/experiment_ID/codebase_snapshot_training/ python train.py --device cuda:1 --restart_training --use_original_config --run_ID <relevant mlflow run ID>
* Note: mlflow *run_ID* can be found at end of the URL for the given experiment. If the run_ID is not given at command line, mlflow will create a new Run for this Run
* Note: The intention of having restart_training flag is only for situations when a model should begin training again *without any changes* (such as if a GPU crashed). If you want to make any changes to the code or if simply desiring to start a new experiment with pre-trainied weights, do not run with the *--restart_training* flag

#### Note on config files:
* If you wish to make edits to the config and hyperparameters, you can either specify an alternate config file on the command line (suggested), or you can directly edit the default config (not recommended)

#### Code dependencies:
Einops, PyTorch, Matplotlib, YACS, PyYaml, Pickle, h5py, ...

#### Jupyter test notebook setup
Use the notebook hook in this [link](https://github.com/ipython/ipython/issues/8009#issuecomment-78154448)
to keep the python scripts and notebooks consistent

#### How to use with MLflow UI:
To use the MLFlow UI, follow the following steps. Note that you can run the code without following these steps. These are only for accesing the UI. As long as mlflow is installed (pip install mlflow==1.26.), the training code can be run.
* ssh into server
* Activate your conda/python enviroment, which has mlflow installed 
   * if mlflow is not yet installed: 'pip install mlflow==1.26.1' shoud work
* cd into the directory which contains the 'mlruns' folder
   * mlflow will create this folder by default if one does not exist yet the first time training is run. If no such folder exists yet, please run a mlflow-integrated training session
   * by default, this should be found in 'scene_understanding/'
* Run the following command (to start the mlflow UI server)
   > mlflow ui
   * Note: use the -p flag to set the port; the default port used is 5000
* In a separate terminal session, run the following command, with [address] replaced with the ip of the server or its alias. (This is just in order to route what the mlflow server is servering, which is hosted on the ssh client, to your local computer)
   > ssh -X -L 8888:localhost:5000 [addresss]
* On a web browser, go to: localhost:8888


---

### External code sources:
1. [Resnet backbone DeepLabv3+](https://github.com/yassouali/pytorch_segmentation.git) 
2. We use Unity [ML-Agents](https://github.com/Unity-Technologies/ml-agents)
    to build our virtual world for the actor. 

