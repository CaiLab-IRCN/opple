import numpy as np
import copy
import torch
from torch import nn

# local imports
from src.models.segmentation_net.net_model import SegNet
# from src.models.segmentation_vae_net.net_model import BVAE
from src.models.slot_attention_net.net_model import SlotAttentionAutoEncoder
import src.utils as utils

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

class Segmentation_Network(nn.Module):
    def __init__(self, cfg, params):
        super(Segmentation_Network, self).__init__()
        self.seg_net = SegNet(cfg, params)
        self.device = params['seg_device'] 
        self.seg_net.to(self.device, torch.float32)
        self.cfg = cfg
        # Old code that allowed for each sub-network to load weights individually 
        if "segmentation_weightloc" in params and params['segmentation_weightloc'] is not None:
            weights = torch.load(params['segmentation_weightloc'], self.device)
            weights = utils.convert_weight_format(weights, "segmentation")
            self.seg_net.load_state_dict(copy.deepcopy(weights))
            print(f"Segmentation network weights were loaded from:  {params['segmentation_weightloc']}")

        self.smooth = nn.Sigmoid()
        self.latent_size = (cfg.TRAIN.num_slots, 3+cfg.ROTATION.num_bins+cfg.TRAIN.segment_id_size)
        self.num_bins = cfg.ROTATION.num_bins
        self.special_nonlinearity = cfg.SEGNET.special_nonlin
        self.loc_limit = cfg.SEGNET.loc_lim
        self.zlim = cfg.DATASET.zlim
        fovx_rad = (cfg.DATASET.fov[1]/180)*np.pi
        self.max_obj_angle = fovx_rad / 2.0 * 1.2 
        self.fcount = 0

    ## removing depth 
    def forward(self, frame):

        self.fcount +=1
        encoded_frame, encoded_frame_flattened, unet_links = self.seg_net.encoder(frame) # 'frame'shape: [20,3,128,128]
        # 'encoded_frame' shape:            [20,256,1,1]
        # 'encoded_frame_flattened' shape:  [20,256]
        
        latent_representation = self.seg_net.latent_out(encoded_frame_flattened)
        # 'latent_representation' shape:    [20,3,187]  (where 3 is num slots and 187 is latent_size (3+pose_bins+id_size))
        latent_representation = torch.reshape(latent_representation,\
                (encoded_frame.shape[0], self.latent_size[0], self.latent_size[1])) 

        """
        # transform x, y, z positions from the latent representation into either polar coordinates
        # or not
        """
        if self.cfg.SEGMENTATION.loc_polar:
            """
            # the output of the network is first converted to represent
            # the polar angle and distance of the object to the camera, then
            # converted to x and y. This way, object is restricted to be an area
            # slightly larger than the visible range
            """
            angle = (self.smooth(latent_representation[:,:,0]) * 2.0 - 1.0) * self.max_obj_angle
            if self.special_nonlinearity:
                distance = self.smooth(torch.exp(latent_representation[:,:,1]-1.0)) *\
                        self.loc_limit * 2 - self.loc_limit + 0.1
            else:
                distance = self.smooth(latent_representation[:,:,1]) * self.loc_limit
            x_transformed = torch.sin(angle) * distance
            if self.cfg.EXPERIMENT.exp_type == "movi":  #FOR MOVI VERSION:
                z_transformed = - torch.cos(angle) * distance
            else:  #NON-MOVI VERSION:
                y_transformed = torch.cos(angle) * distance
        else:
            x_transformed = self.loc_limit*(2*self.smooth(latent_representation[:,:,0])\
                    - torch.tensor( [1.0], device = self.device))
            y_transformed = self.loc_limit*self.smooth(latent_representation[:,:,1])

        if self.cfg.EXPERIMENT.exp_type == "movi": #FOR MOVI VERSION:
            y_transformed = self.cfg.DATASET.zlim * (2*self.smooth(latent_representation[:,:,2])\
                    - torch.tensor([1.0], device=self.device))
        else: #NON-MOVI VERSION:
            z_transformed = self.cfg.DATASET.zlim * (2*self.smooth(latent_representation[:,:,2])\
                    - torch.tensor([1.0], device=self.device))
        pose_transformed = torch.nn.functional.log_softmax(latent_representation\
                [:,:,3:3+self.num_bins], dim=2)

        latent_representation_transformed = torch.cat((x_transformed[:,:,None],\
                y_transformed[:,:,None], z_transformed[:,:,None], pose_transformed,\
                latent_representation[:,:,3+self.num_bins:]), 2)
        # 'latent_representation_transformed' shape:    [20,3,187]

        if(self.fcount == 2):
            if(encoded_frame.requires_grad):
                encoded_frame.register_hook(save_grad('encoded_frame'))
            if(latent_representation.requires_grad):
                latent_representation_transformed.register_hook(save_grad('latent_representation'))
            
        attention_maps = []
        for i in range(self.latent_size[0]):
            att = self.seg_net.decoder_attention(latent_representation_transformed\
                    [:, i, 3+self.num_bins:], unet_links, encoded_frame.shape)
            # 'att' shape: [20,1,128,128]
            attention_maps.append(att)

            if(self.fcount == 2):
                if(att.requires_grad):
                    att.register_hook(save_grad('att_grad_'+str(i)))
            elif(self.fcount == 3):
                self.fcount = 0
        
        """ 
        attention map and representations being disentangled by RNN
        """
        if self.cfg.SEGNET.use_RNN:
            attention_maps_softmax = []
            zeros = torch.zeros_like(attention_maps[0], device = self.device)

            if self.cfg.SEGMENTATION.seq_mask:
                residual_mask = torch.ones_like(attention_maps[0], device = self.device)
                for i, att in enumerate(attention_maps):
                    seg = torch.nn.functional.softmax(torch.cat((att, zeros), 1), 1) * residual_mask
                    attention_maps_softmax.append(seg[:,0].unsqueeze(1))
                    residual_mask = seg[:,1].unsqueeze(1)
                attention_maps_softmax.append(residual_mask)
                attention_maps_softmax = torch.cat(attention_maps_softmax, 1)
                attention_maps_logits = torch.cat(attention_maps + [zeros], 1)
            else:
                attention_maps_logits = torch.cat(attention_maps + [zeros], 1)
                attention_maps_softmax = torch.nn.functional.softmax(attention_maps_logits, 1)

        else:
            attention_maps = torch.cat(attention_maps, 1)
            attention_maps_logits = torch.cat((attention_maps,\
                    torch.zeros(attention_maps[:, 0].shape, device = self.device).unsqueeze(1)), 1)
            attention_maps_softmax = torch.nn.functional.softmax(attention_maps_logits, 1)
        if(attention_maps_softmax.requires_grad):
            attention_maps_softmax.register_hook(save_grad('att_aftersfm'))
        
        # 'attention_maps_softmax' shape:               [20,4,128,128]
        # 'attention_maps_logits' shape:                [20,4,128,128]
        # 'latent_representation_transformed' shape:    [20,3,187]

        output = {'attention_maps_softmax':attention_maps_softmax,
                'attention_maps_logits':attention_maps_logits,
                'latent_representation_transformed':\
                        latent_representation_transformed}
        return output

class Segmentation(nn.Module):
    def __init__(self, cfg, params, seg_net):
        super(Segmentation, self).__init__()
        self.cfg = cfg
        self.params = params
        self.device = params['seg_device']
        self.seg_net = seg_net

    def forward(self, data_batch):
        """
        shapes: 
            frames - triplet x b x c x w x h; 
        """
        
        # unpack data_batch
        frames = data_batch['frames']
        
        seg_outs = []
        for frame in frames: # for each frame in triplet
            seg_outs.append(self.seg_net(frame)) 
            # returned keys: ['attention_maps_softmax', 'attention_maps_logits', 'latent_representation_transformed']
 
        return seg_outs
