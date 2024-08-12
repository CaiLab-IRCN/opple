import os
from PIL import Image
import torch
import cv2
import numpy as np

# local imports
from .data_utils import image_resize, camera_viewpoint_transform_movi, transform_c, label_map_2_multi_channel_mask
from .dataloader_customdata_map import CustomDatasetMap

class CustomDatasetMapMOVi(CustomDatasetMap):
    def __init__(self, cfg, params, dataset_path, images_dir_name,\
                                        pickled_datamap_filename, split):

        super().__init__(cfg, params, dataset_path, images_dir_name,\
                         pickled_datamap_filename, split)
        
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
        if frame.shape[-1] > 3:
            # get rid of the alpha channel
            frame = frame[:,:,:3]
        frame = image_resize(frame, self.image_size[0], self.image_size[1])
        # frame = torch.tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype = torch.float)
        frame = torch.tensor(frame, dtype = torch.float)
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
        campos = camera_viewpoint_transform_movi(campos) 

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
        objpos_raw  - 

        Output:
        obj         - torch.tensor of size [3, 3] of type torch.float
        '''
        # obj = torch.tensor(obj_raw, dtype = torch.float)
        obj = torch.zeros([3,3])

        return obj

    def load_objpose(self, objpose_raw):
        '''
        Decription: loads object pose data and returns in format expected by model.
                    Can be overidden in subclasses to customly load differently
                    formatted data
        
        Input:
        objpos_raw    - 

        Output:
        objpose       - torch.tensor of size [3, 6] of type torch.float32
        '''
        # objpose = torch.tensor(objpose_raw, dtype = torch.float)
        # objpose = object_pose_viewpoint_transform(objpose)
        objpose = torch.zeros([3,6])

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
        segmentation = torch.tensor(segmentation, dtype=torch.float) #shape: [H x W] (ie, single channel label map)
        # Converting MOVi segmentation groundtruth from single channel label map to multi-channel binary masks (ie, slot representation)
        segmentation = label_map_2_multi_channel_mask(segmentation, self.num_slots).permute(2,0,1)
        
        ## THE BELOW LINE WILL ASSIGN A ZERO MATRIX TO THE VARIABLE, ESSENTIALLY IGNORING IT
        # segmentation = torch.zeros([self.num_slots, self.image_size[0], self.image_size[1]])

        return segmentation

    def load_24bitdepth(self, path):
        """
            Load 24 bit depth encoded in RGB images and convert it into a float tensor
            ==
            path: str: path to the depth image (.png)
        """
        depthfloat =  torch.zeros(self.image_size[0], self.image_size[1])

        return depthfloat

    def objdeetdecode(self, objdeet):
        """
            decodes the object details from a objdeet dictionary
        """
        return torch.zeros((3,3))

    def scenedeetdecode(self, scenedeet):
        """
            decodes the scene details from a scenedeet dictionary
        """
        return torch.zeros(3,3)
