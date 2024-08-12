import os
from typing import Callable, Any
from PIL import Image
import torch
import pickle
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd

# local imports
from .data_utils import image_resize, object_viewpoint_transform,\
    object_pose_viewpoint_transform, rgb_2_label_map, label_map_2_multi_channel_mask, old_segmentation_reader,\
    transform_c, camera_viewpoint_transform
from .variables import return_contexts, return_colormapping

class CustomDatasetMap(Dataset):
    """ 
        Class to read data from png file dataset and return a pytorch dataset class object
        This dataloader is made with a map style function and to make it so the pointer directions
        are added in a dataset pkl file which is made in preprocess step from the
        ../../data/customdata_png2pickle.py

    """
    def __init__(self, cfg, params, dataset_path, images_dir_name, pickled_datamap_filename, split):
        '''
            - dataset_path:     
                path to dir containing the dir of input images and the
                pickled datamap file
            - images_dir_name   
                name of the dir inside the dataset_path dir which contains
                the input images
            - pickled_datamap_filename    
                filename of the pickled datamap file
            - split:            
                one of "val", "train", "test" or None; used to determine which data
                is required to be loaded (ie, some are skipped for training)
        '''

        super().__init__()
        self.images_dir_path : str = os.path.join(dataset_path, images_dir_name)
        self.pickled_datamap_path : str = os.path.join(dataset_path, pickled_datamap_filename)
        self.cos_threshold  :float = np.cos(cfg.DATASET.cam_rot_lim)
        self.num_slots :int = cfg.TRAIN.num_slots+1 # +1 for background but here just for number of oject):
        self.image_size : list = cfg.DATASET.frame_size # tuple of (Height x Width)
        self.skipped_count : int = 0
        self.skipped_count_depth :int  = 0
        self.reverse_data : bool = cfg.DATASET.reverse_data
        self.split : str = split
        self.test : bool = params["test"]
        self.perframe : bool = params["perframe"]
        self.non_presplit : bool = params["non_presplit"]
        
        with open(self.pickled_datamap_path, 'rb') as f: self.data_map = pickle.load(f)
        print('Pickle file loaded')
        self.data_map : Any = pd.DataFrame.from_dict(self.data_map)
        print('Converted pickle to dataframe')
        self.pixel_loc = torch.zeros((self.image_size[0], self.image_size[1], 2))
        self.pixel_loc[:,:,0] = torch.tensor([i for i in range(self.image_size[0])])[:,None]
        self.pixel_loc[:,:,1] = torch.tensor([i for i in range(self.image_size[1])])[None,:]
        
        #### colour map for decoding attention map
        self.unique_colors : Callable = return_colormapping()
        self.contexts : Callable = return_contexts()

    def get_sampling_weights(self):
        """
            Getting the sampling weight from the data if selected to weight the datapoints in the
            dataloader on the basis of the camera step size in consequent frames
        """
        weights = self.data_map['weight']
        return torch.tensor(weights)

    def getcontextlabels(self, context):
        """
            Get the context labels from initialized contexts
        """
        return self.contexts[context]

    def __len__(self):
        if self.perframe:
            # Do not require consecutive frames during test time inference (depth and segmentation)
            # So each of original data tuple can turn into 3 samples
            return self.data_map.shape[0] * 3
        else:
            return self.data_map.shape[0]

    def __getitem__(self, idx):
        if self.perframe:
            frames, cameras, camquats, objects, depths,\
                segmentations, objectposes, objectdeets,\
                scenedeets, idxs = self.load_data_per_frame(idx)
        else:
            frames, cameras, camquats, objects, depths,\
                segmentations, objectposes, objectdeets,\
                scenedeets, idxs = self.load_data(idx)

        return frames, cameras, camquats, objects,\
            depths, segmentations, objectposes,\
            objectdeets, scenedeets, idxs

    def load_frame(self, frame_str):
        '''
        Input:
        frame_str   - path string to single RGB image frame
                      Can be overidden in subclasses to customly load differently
                      formatted data

        Output:
        frame       - RGB img frame in format/representation expected by the model
                      format: torch.tensor of size [Height, Width, 3] of type torch.float
        '''
        frame = np.array(Image.open(frame_str))
        frame = image_resize(frame, self.image_size[0], self.image_size[1])
        frame = torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype = torch.float)
        frame = transform_c(frame)

        return frame
    
    def load_campos(self, campos_raw):
        '''
        Decription: loads camera position and returns in format expected by model.
                    Can be overidden in subclasses to customly load differently
                    formatted data
        
        Input:
        campos_raw   - list of floats of len (6) of 

        Output:
        campos       - torch tensor of size: [9] of type torch.float
        '''
        campos = torch.tensor(campos_raw, dtype = torch.float)
        campos = camera_viewpoint_transform(campos)

        return campos
    
    def load_camquat(self, quat_raw):
        '''
        Decription: loads camera pose/rotation as a 4x1 quaternion
        
        Input:
        quat   - np array, shape [4], dtype float32 
               - expected to be in XYZW format (following scipy's convention)

        Output:
        campos - torch tensor of size: [4] of type torch.float
        '''
        campos = torch.tensor(quat_raw, dtype = torch.float)

        return campos


    def load_obj(self, obj_raw):
        '''
        Decription: loads object position data and returns in format expected by model.
                    Can be overidden in subclasses to customly load differently
                    formatted data
        
        Input:
        obj_raw  - nested list of floats of shape (3, 3)

        Output:
        obj         - torch.tensor of size [3, 3] of type torch.float
        '''
        obj = torch.tensor(obj_raw, dtype = torch.float)
        obj = object_viewpoint_transform(obj)

        return obj
    
    def load_objpose(self, objpose_raw):
        '''
        Decription: loads object pose data and returns in format expected by model.
                    Can be overidden in subclasses to customly load differently
                    formatted data
        
        Input:
        objpos_raw    - nested list of floats of shape (3, 3)

        Output:
        objpose       - torch.tensor of size [3, 6] of type torch.float32
        '''
        objpose = torch.tensor(objpose_raw, dtype = torch.float)
        objpose = object_pose_viewpoint_transform(objpose)

        return objpose


    def load_segmentation(self, segmentation_str):
        '''
        Decription: loads and returns ground truth segmentation data in format expected
                    by upstream code (ie, evaluation code). Can be overiden in subclasses
                    to customly load differently formatted data
        
        Input:
        segmentation_str    - path string to single groundtruth segmentation image

        Output:
        segmentation        - torch.tensor of size [Slots, Height, Width] of type torch.float32
        '''
        segmentation = Image.open(segmentation_str)
        segmentation = image_resize(np.array(segmentation),\
                self.image_size[0], self.image_size[1], inter=cv2.INTER_NEAREST)
        segmentation = torch.tensor(segmentation, dtype=torch.float) #shape: [H x W x 3]
        # Converting segmentation groundtruth from RGB to multi-channel binary masks (ie, slot representation)
        if not self.non_presplit:
            segmentation = label_map_2_multi_channel_mask(rgb_2_label_map(segmentation), self.num_slots).permute(2,0,1)
        if self.non_presplit:
            segmentation = old_segmentation_reader(segmentation, self.num_slots, self.unique_colors)
        
        return segmentation

    def load_24bitdepth(self, path):
        """
            Load 24 bit depth encoded in RGB images and convert it into a float tensor
            ==
            path: str: path to the depth image (.png)
        """
        depth24bit = image_resize(np.array(Image.open(path)), self.image_size[0], self.image_size[1])
        depth24bit = cv2.cvtColor(depth24bit, cv2.COLOR_BGR2RGB)
        depthfloat = (depth24bit[:,:,0]* 2**16 + depth24bit[:,:,1] * 2**8 +\
                depth24bit[:,:,2])/(2**16)
        depthfloat = depthfloat.astype(np.float32)
        depthfloat = torch.tensor(depthfloat)

        return depthfloat

    def objdeetdecode(self, objdeet):
        """
            decodes the object details from a objdeet dictionary
        """
        objdeetdec = []
        for s in range(len(objdeet)):
            objdeetdec.append([self.contexts['objshape'][objdeet[s][0]],\
                self.contexts['objtex'][objdeet[s][1]]]) #object texture detail
        return torch.tensor(objdeetdec)

    def scenedeetdecode(self, scenedeet):
        """
            decodes the scene details from a scenedeet dictionary
        """
        scenedeetdec = []
        scenedeetdec.extend([self.contexts['floortex'][scenedeet[0]],\
            self.contexts['walltex'][scenedeet[1]]]) #scene wall texture details
        return torch.tensor(scenedeetdec)

    def load_data(self, idx):
        """
            For each idx load a data point from the pkl data map file in the datapath
        """

        d = self.data_map.iloc[idx]

        frames = None
        cameras = None
        objects = None
        objectposes = None
        objectdeets = None
        scenedeets = None
        idxs = None
        mats = None
        depths = None
        segmentations = None

        ''' 
        Image Frame Data

        Model expects...
        '''
        img_str = os.path.join(self.images_dir_path, "{}_img.png")
        fr1 = self.load_frame(img_str.format(d.get("frame1")))
        fr2 = self.load_frame(img_str.format(d.get("frame2")))
        fr3 = self.load_frame(img_str.format(d.get("frame3")))
        assert(fr1.shape == torch.Size([self.image_size[0], self.image_size[1], 3]))
        assert(fr2.shape == torch.Size([self.image_size[0], self.image_size[1], 3]))
        assert(fr3.shape == torch.Size([self.image_size[0], self.image_size[1], 3]))

        ''' 
        Camera Position/Pose Data

        Model expects camera position variables (each of cam1, cam2, cam3) returned
        from this dataloader to be in the following format:
            - torch tensor of size [9], where each entry corresponnds to the
            following, respecitvely:
            [ X, Y, Z, cos(yaw), sin(yaw), cos(pitch), sin(pitch), cos(roll), sin(roll) ]
            *note: here, yaw is rotate around Z, pitch is rotate around X, roll is rotate around Y

        Coordiante system assumptions: 
        - Postive yaw change corresponds with counter-clockwise rotation
        - When camera is facing 0 deg (ie, yaw = 0):
            - X change is +rightward/-leftward
            - Y change is +forward/-backward
            - Z is vertical axis (current model expects Z to be constant)

        *Note:
        If the campos values are not in the right format by default, a specified custom
        'camera_transform()' func can be called here, although it is recommended that such
        conversions are done prior to runtime
        '''
        cam1 = self.load_campos(d.get("campos1"))
        cam2 = self.load_campos(d.get("campos2"))
        cam3 = self.load_campos(d.get("campos3"))
        assert(cam1.shape == torch.Size([9]))
        assert(cam2.shape == torch.Size([9]))
        assert(cam3.shape == torch.Size([9]))

        ''''
        Cam position data as quaternion
        - read in as list/np.array of len 4
        - returned as tensor of torch.Size([4])
        - expect returned result to be in [XYZW] format
        '''
        if("camquat1" in d) and (d.get("camquat1") is not None):
            camquat1 = self.load_camquat(d.get("camquat1"))
            camquat2 = self.load_camquat(d.get("camquat2"))
            camquat3 = self.load_camquat(d.get("camquat3"))
            assert(camquat1.shape == torch.Size([4]))
            assert(camquat2.shape == torch.Size([4]))
            assert(camquat3.shape == torch.Size([4]))
        else:
            camquat1 = camquat2 = camquat3 = torch.zeros((4))

        ''' 
        Segmentation Data

        Model expects...
        '''
        segmentation_str = self.images_dir_path+"{}_layer.png"
        if os.path.exists(segmentation_str.format(d.get("frame1"))):
            segmentation1 = self.load_segmentation(segmentation_str.format(d.get("frame1")))
            segmentation2 = self.load_segmentation(segmentation_str.format(d.get("frame2")))
            segmentation3 = self.load_segmentation(segmentation_str.format(d.get("frame3")))

            expected_shape = (self.num_slots, self.image_size[0], self.image_size[1]) #shape: [slots x H x W]
            assert(segmentation1.shape == (expected_shape)), f"Unexpected shape: {segmentation1.shape}"
            assert(segmentation2.shape == (expected_shape)), f"Unexpected shape: {segmentation2.shape}"
            assert(segmentation3.shape == (expected_shape)), f"Unexpected shape: {segmentation3.shape}"
        else:
            segmentation1 = segmentation2 = segmentation3 = torch.zeros((self.num_slots, self.image_size[0], self.image_size[1]))


        # Set the other variables to default values
        obj1 = obj2 = obj3 = torch.zeros((3,3))
        objpose1 = objpose2 = objpose3 = torch.zeros((3,6))
        objdeet1 = objdeet2 = objdeet3 = torch.zeros(obj1.shape)
        scenedeet1 = scenedeet2 = scenedeet3 = torch.zeros(obj1.shape)
        depth1 = depth2 = depth3 = torch.zeros(fr1.shape[:2])
        
        # If val or test data, load the extra dataset values (for training, these won't
        # be used so can skip loading)
        if not self.split == "train":

            ''' 
            Object Position Data

            Model expects returned object position variables where each of (obj1, obj2, obj3)
            are the coords for all objects in each respective frame
            The expected coordinate system is ___
            Example: obj1:
                frame 1 [ 1st object[x,    y,   z],
                            2nd object[x,    y,   z],
                            3rd object[x,    y,   z] ]
            '''
            if("objpos1" in d) and (d.get("objpos1") is not None):
                obj1 = self.load_obj(d.get("objpos1"))
                obj2 = self.load_obj(d.get("objpos2"))
                obj3 = self.load_obj(d.get("objpos3"))
                assert(obj1.shape == torch.Size([3, 3]))
                assert(obj2.shape == torch.Size([3, 3]))
                assert(obj3.shape == torch.Size([3, 3]))

            ''' 
            Object Pose Data

            Model expects...
            '''
            if("objpose1" in d) and (d.get("objpose1") is not None):
                    objpose1 = self.load_objpose(d.get("objpose1"))
                    objpose2 = self.load_objpose(d.get("objpose2"))
                    objpose3 = self.load_objpose(d.get("objpose3"))

            ''' 
            Object Detail Data

            Model expects...
            '''
            if("objdeet1" in d) and (d.get("objdeet1") is not None):
                    objdeet1 = self.objdeetdecode( d.get("objdeet1"))
                    objdeet2 = self.objdeetdecode(d.get("objdeet2"))
                    objdeet3 = self.objdeetdecode(d.get("objdeet3"))
            
            ''' 
            Scene Detail Data

            Model expects...
            '''
            if("scenedeet1" in d) and (d.get("scenedeet1") is not None):
                    scenedeet1 = self.scenedeetdecode(d.get("scenedeet1"))
                    scenedeet2 = self.scenedeetdecode(d.get("scenedeet2"))
                    scenedeet3 = self.scenedeetdecode(d.get("scenedeet3"))

            ''' 
            Depth Data

            Model expects...
            '''
            depth_s = self.images_dir_path+"{}_id.png"
            depth1 = self.load_24bitdepth(depth_s.format(d.get("frame1")))
            depth2 = self.load_24bitdepth(depth_s.format(d.get("frame2")))
            depth3 = self.load_24bitdepth(depth_s.format(d.get("frame3")))
            
        # With half chance, we reverse the sequence of the data
        if (self.reverse_data and not self.test) and (np.random.rand() > 0.5):
            fr1, fr2, fr3 = fr3, fr2, fr1
            cam1, cam2, cam3 = cam3, cam2, cam1
            camquat1, camquat2, camquat3 = camquat3, camquat2, camquat1
            obj1, obj2, obj3 = obj3, obj2, obj1
            depth1, depth2, depth3 = depth3, depth2, depth1
            segmentation1, segmentation2, segmentation3 = segmentation3, segmentation2, segmentation1
            objpose1, objpose2, objpose3 = objpose3, objpose2, objpose1
            objdeet1, objdeet2, objdeet3 = objdeet3, objdeet2, objdeet1
            scenedeet1, scenedeet2, scenedeet3 = scenedeet3, scenedeet2, scenedeet1
        
        # just stack the data together as triplets
        cameras = torch.stack([cam1, cam2, cam3], 0)
        camquats = torch.stack([camquat1, camquat2, camquat3], 0)
        objects  = torch.stack([obj1, obj2, obj3], 0)
        depths = torch.stack([depth1, depth2, depth3], 0)
        objectposes = torch.stack([objpose1, objpose2, objpose3], 0)
        objectdeets = torch.stack([objdeet1, objdeet2, objdeet3], 0)
        scenedeets = torch.stack([scenedeet1, scenedeet2, scenedeet3], 0)
        frames = torch.stack([fr1, fr2, fr3], 0)
        segmentations = torch.stack([segmentation1, segmentation2, segmentation3], 0)
        
        if("scene_num" in d) and (d.get("scene_num") is not None):
            idxs = torch.tensor([d.get("scene_num")])
        # framegap = torch.tensor([d.get("frame_gap")])
        # fweight = torch.tensor([d.get("weight")])

        return frames, cameras, camquats, objects,\
                depths, segmentations, objectposes,\
                objectdeets, scenedeets, idxs


    def load_data_per_frame(self, idx):
        """
            Instead of loading a triplet tuple, load single frame and its associated data.
        """

        idx_tuple = idx // 3
        idx_img = idx % 3
        d = self.data_map.iloc[idx_tuple]

        img_str = os.path.join(self.images_dir_path, "{}_img.png") ## image name string
        fr = self.load_frame(img_str.format(d.get("frame{}".format(idx_img+1))))

        cam = self.load_campos(d.get("campos{}".format(idx_img+1)))

        obj = torch.zeros([3, 3])
        if("objpose{}".format(idx_img+1) in d):
            if(d.get("objpose{}".format(idx_img+1)) is not None):
                obj = self.load_obj(d.get("objpos{}".format(idx_img+1)))

        objpose = torch.zeros([3, 6])
        if("objpose{}".format(idx_img+1) in d):
            if(d.get("objpose{}".format(idx_img+1)) is not None):
                objpose = self.load_objpose(d.get("objpose{}".format(idx_img+1)))

        if("objdeet{}".format(idx_img+1) in d):
            if(d.get("objdeet{}".format(idx_img+1)) is not None):
                objdeet = self.objdeetdecode( d.get("objdeet{}".format(idx_img+1)))
        
        if("scenedeet{}".format(idx_img+1) in d):
            if(d.get("scenedeet{}".format(idx_img+1)) is not None):
                scenedeet = self.scenedeetdecode(d.get("scenedeet{}".format(idx_img+1)))

        depth_s = self.images_dir_path+"{}_id.png"
        depth = self.load_24bitdepth(depth_s.format(d.get("frame{}".format(idx_img+1))))

        segmentation_str = self.images_dir_path+"{}_layer.png"
        segmentation = self.load_segmentation(segmentation_str.format(d.get("frame{}".format(idx_img+1))))

        if("scene_num" in d) and (d.get("scene_num") is not None):
            idxs = torch.tensor([d.get("scene_num")])
        # framegap = torch.tensor([d.get("frame_gap")])
        # fweight = torch.tensor([d.get("weight")])
        
        fr = fr.unsqueeze(0)
        cam = cam.unsqueeze(0)
        obj = obj.unsqueeze(0)
        depth = depth.unsqueeze(0)
        segmentation = segmentation.unsqueeze(0)
        objpose = objpose.unsqueeze(0)
        objdeet = objdeet.unsqueeze(0)
        scenedeet = scenedeet.unsqueeze(0)

        return fr, cam, obj, depth, segmentation, objpose, objdeet, scenedeet, idxs
