import copy
import torch
from torch import nn
from src.models.imagination_net.net_model import ImagNet

# local imports
from src.utils import convert_weight_format

class Imagination_Network(nn.Module):
    def __init__(self, cfg, params):
        super(Imagination_Network, self).__init__()
        self.imag_net = ImagNet(cfg, params)
        self.device = params['imag_device'] 
        self.imag_net.to(self.device, torch.float32)
        # allows each sub-network to load weights individually (backwards compatibility)
        if "imagination_weightloc" in params and params['imagination_weightloc'] is not None:
            weights = torch.load(params['imagination_weightloc'], self.device)
            weights = convert_weight_format(weights, "imagination")
            self.imag_net.load_state_dict(copy.deepcopy(weights))
            print(f"Imagination network weights were loaded from:  {params['imagination_weightloc']}")

        self.smooth = nn.Sigmoid()
        self.cfg = cfg
        self.depth_limit = cfg.IMAGNET.depth_range ## the far plane value
        self.special_nonlinearity = cfg.IMAGNET.special_nonlin

    def forward(self, X, motion_params):
        data_batch = {'frame1':X, 'motion_params':motion_params}
        logits, latent = self.imag_net(data_batch)
        image = self.smooth(logits[:,:3,:,:])
        if self.special_nonlinearity:
            depth = self.smooth(torch.exp(logits[:, 3, :,:]-1.0)) *\
                    self.depth_limit * 2 - self.depth_limit + 0.1
        else:
            depth = self.smooth(logits[:, 3, :,:]) * self.depth_limit +\
                    torch.tensor(1e-5, device=self.device)
        attention_map = logits[:,4,:,:]
        output = {
                'image':image,
                'depth':depth,
                'logits':logits,
                'attention_map':attention_map,
                'latent':latent
            }
        return output

class Imagination(nn.Module):
    def __init__(self, cfg, params, imag_net):
        super(Imagination, self).__init__()
        self.cfg = cfg
        self.params = params
        self.device = params['imag_device']
        self.imag_net = imag_net

    def forward(self, data_batch, segmentation_output, warping_output):
        # unpack the data
        motion_params2 = data_batch['motion_params2']
        img = data_batch['frames']
        latent_out2 = segmentation_output['seg_out2']['latent_representation_transformed']
        object_motion = warping_output['object_motion']
        segmentation_masks2 = segmentation_output['seg_out2']['attention_maps_softmax']
        depth2_log = data_batch['depth_log'][1]

        # make the motion params vector
        if(self.cfg.IMAGINATION.input_loc):
            motion_params_imagine = torch.cat((torch.nn.functional.pad(latent_out2[:,:,:2],\
                    (0,0,0,1)), motion_params2.unsqueeze(1).repeat(1,object_motion.shape[1],1)\
                    [:,:,2].unsqueeze(2), object_motion[:,:,[0,1,3,4]]), dim = 2)
        else:
            motion_params_imagine = torch.cat((motion_params2.unsqueeze(1).repeat(1,\
                object_motion.shape[1],1), object_motion[:,:,[0,1,3,4]]), dim = 2)

        # motion_params_imagine = motion_params2.unsqueeze(1).repeat(1, object_motion2.shape[1], 1)
        motion_params_imagine = motion_params_imagine.type(torch.float32)
        # adding expected object movement for imagination for each slot

        # for each slot of frame 3 create an imagined depth, image, and weight
        # TODO: vectorize it? loops are really slow
        imagination3_slots = []
        depth3_imagine_slots = []
        imagine_logits3p_slots = []
        imagine_attmap_slots = []

        for i in range(self.cfg.TRAIN.num_slots_withbg):
            img_slot = img[1]*segmentation_masks2[:,i,None,:,:] 
            depth_slot = depth2_log.squeeze(1)*segmentation_masks2[:,i,:,:]
            depth_slot = depth_slot.unsqueeze(1)
            X_input = torch.cat([img_slot, depth_slot], dim=1)

            imag_output = self.imag_net(X_input, motion_params_imagine[:,i,:])

            imagination3_slots.append(imag_output['image'])

            depth3_imagine_slots.append(imag_output['depth'])

            imagine_logits3p_slots.append(imag_output['logits'])
            imagine_attmap_slots.append(imag_output['attention_map'])

        # stack and prep the imagined attention maps, weights, depth
        depth3_imagine = torch.stack(depth3_imagine_slots, 1)
        depth3_weights = torch.nn.Softmax(dim=1)(-depth3_imagine)
        ## inversly proportional weights to depth

        imagine_attmap3 = torch.stack(imagine_attmap_slots, 1)
        imagine_map_weights = torch.nn.Softmax(dim=1)(imagine_attmap3)

        imagination3 = torch.sum(torch.stack(imagination3_slots, 1)*\
                imagine_map_weights[:,:, None, :,:],1)
        
        # put them together to get the imagined frame prediction
        depth3_imagine = torch.sum(imagine_map_weights*depth3_imagine,1)

        imagine_logits3p = torch.sum(torch.stack(imagine_logits3p_slots, 1)*\
                imagine_map_weights[:,:,None,:,:], 1)
        imagine_logit3p_raw = torch.stack(imagine_logits3p_slots, 1)

        # put the outputs together and return
        imag_outputs = {
                'img3_imag':imagination3,
                'depth3_imag':depth3_imagine,
                'logits3_imag':imagine_logits3p,
                'logits3_imag_raw':imagine_logit3p_raw}
        return imag_outputs
