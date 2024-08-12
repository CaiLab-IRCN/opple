import numpy as np
import torch
import copy
from torch import nn
import copy

# local imports
from src.models.depth_net.net_model import DepthNet
from src.utils import convert_weight_format

inf = np.inf

class Depth_Inference(nn.Module):
    def __init__(self, cfg, params):
        super(Depth_Inference, self).__init__()
        self.depth_net = DepthNet(cfg, params)
        self.depth_net.to(params['depth_device'], torch.float32)
        # Old code that allowed for each sub-network to load weights individually 
        if 'depth_weightloc' in params and params['depth_weightloc'] is not None:
            weights = torch.load(params['depth_weightloc'], params['depth_device'])
            weights = convert_weight_format(weights, "depth")
            self.depth_net.load_state_dict(copy.deepcopy(weights))
            print(f"Depth network weights were loaded from:  {params['depth_weightloc']}")

        self.smooth = nn.Sigmoid()
        self.depth_mul = cfg.DEPTHNET.depth_range ## max depth in the scene
        self.special_nonlinearity = cfg.DEPTHNET.special_nonlin
        self.perframe = params['perframe']

    def depth_inf(self, x):
        depth_logits = self.depth_net(x) # depth logits

        if self.special_nonlinearity:
            depth_smoothed = self.smooth(torch.exp((depth_logits-cfg.DEPTHNET.mu)/cfg.DEPTHNET.sigma))\
                *self.depth_mul * 2 - self.depth_mul + 0.1
        else:
            depth_smoothed = self.smooth(depth_logits) * self.depth_mul

        ### shape - x - b x 1 x W x H 
        output = {
                'depth_smoothed':depth_smoothed,
                'depth_logits':depth_logits
                }
        return output

    def forward(self, data_batch):

        depth_smoothed_combined = []
        depth_logits_combined = []

        for frame in data_batch['frames']: # for each frame in triplet
            output = self.depth_inf(frame) # returns dict with keys 'depth_smoothed' and 'depth_logits'
            depth_smoothed_combined.append(output['depth_smoothed'])
            depth_logits_combined.append(output['depth_logits'])

        depth_smoothed_combined = torch.stack(depth_smoothed_combined)
        depth_logits_combined = torch.stack(depth_logits_combined)
        # SHAPE: final shape for each should be [Triplet x Batch x 1 x Width x Height]

        output = {
            'depth_smoothed':depth_smoothed_combined,
            'depth_logits':depth_logits_combined
                }
        return output
