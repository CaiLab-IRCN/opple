import torch
import src.utils as utils

# compute all the losses for the multi object data experiment
def compute_losses_multi_obj(cfg, params, data_batch, outputs):

    device = params['device']
    losses = {}

    ### Read values in from model's output 
    predicted_img3 = outputs['predicted_outputs']['predicted_img3']

    frame3 = data_batch['frames'][2]
    
    seg_out1 = outputs['segment_outputs']['seg_out1']
    seg_out2 = outputs['segment_outputs']['seg_out2']
    seg_out3 = outputs['segment_outputs']['seg_out3']

    latent1, latent2, latent3 = seg_out1['latent_representation_transformed'],\
        seg_out2['latent_representation_transformed'], seg_out3['latent_representation_transformed']
    attention1, attention2, attention3 = seg_out1['attention_maps_softmax'],\
        seg_out2['attention_maps_softmax'], seg_out3['attention_maps_softmax']

    c2pos = outputs['warping_outputs']['cam2_position'].squeeze(1)
    object_loc3p = outputs['warping_outputs']['object3_loc_predicted']
    object_pose3p = outputs['warping_outputs']['object3_pose_predicted']

    alignment_weight_23, alignment_weight_logit_23, matching_score_23 =\
        utils.latent_object_id_alignment_withbg(cfg, latent2, latent3)

    weighted_object_loc3, weighted_object_pose3 = utils.weight_locpose(object_loc3p,\
            object_pose3p, alignment_weight_23, alignment_weight_logit_23)

    ### Calculate the loss

    ## Sum of pixel wise image prediction loss
    prediction_loss = torch.mean(torch.sum((predicted_img3 - frame3)**2, (1, 2, 3)))
    
    ## Regularization terms
    reg_losses = torch.tensor([0.0], device = device, dtype=torch.float32)

    # Spatial smoothing
    a = torch.mean((attention2[:,:,1:,:] - attention2[:,:,:-1,:])**2, (2,3))
    b = torch.mean((attention2[:,:,:,1:] - attention2[:,:,:,:-1])**2, (2,3))
    reg_losses += cfg.REGULARIZE.spatial_smoothing * torch.mean(torch.sum(a + b, 1))

    ## regularizing object position inference alignments
    ## aligning average object pixel location and object location estimate from the network
    ## we dont use background attention
    ## 'L_cons' (in the paper)
    loss_loc2pix = cfg.REGULARIZE.object_localign_reg*torch.mean((utils.average_location(cfg,\
            c2pos, attention2)[:,:,:3] - latent2[:,:,:3])**2)
    reg_losses += loss_loc2pix
    
    obj_reg = torch.tensor([0.0], device = device, dtype=torch.float32)
    obj_reg += cfg.REGULARIZE.object_localign_reg*torch.mean(torch.sum(\
        (weighted_object_loc3 - latent3[:,:,:3])**2 * matching_score_23[:,:,None], (1,2)))

    shuffle_order = torch.randperm(weighted_object_loc3.shape[0], device=device)
    obj_reg -= cfg.REGULARIZE.object_localign_reg*torch.mean(torch.sum(\
        torch.minimum(torch.sum(torch.abs(latent1[shuffle_order,:,:3] -\
        latent3[:,:,:3]), 2), torch.FloatTensor([1.0]).to(device))\
        * matching_score_23, 1))

    # contrastive loss
    ## we dont use background attention
    ## object pose alignment 
    obj_reg += cfg.REGULARIZE.object_localign_reg * torch.mean(torch.sum(matching_score_23\
        * torch.sum(torch.exp(latent3[:,:,3:3+cfg.TRAIN.num_bins])\
        * (latent3[:,:,3:3+cfg.TRAIN.num_bins] - weighted_object_pose3), 2), 1)) * 0.5

    obj_reg += cfg.REGULARIZE.object_localign_reg * torch.mean(torch.sum(matching_score_23\
        * torch.sum(torch.exp(weighted_object_pose3)\
        * (weighted_object_pose3 - latent3[:,:,3:3+cfg.TRAIN.num_bins]), 2), 1)) * 0.5

    # Use Jensen-Shannon divergence instead of KL divergence
    # contrastive loss
    if cfg.SEGMENTATION.contrastive_loss_pose:
        obj_reg -= cfg.REGULARIZE.object_localign_reg * torch.mean(torch.sum(\
            matching_score_23 * torch.minimum(torch.sum(torch.exp(\
            latent3[:,:,3:3+cfg.TRAIN.num_bins]) * (latent3[:,:,3:3+cfg.TRAIN.num_bins] -\
            latent1[shuffle_order,:,3:3+cfg.TRAIN.num_bins]), 2),\
            torch.FloatTensor([2.0]).to(device)), 1))

    reg_losses += obj_reg
    
    total_loss = reg_losses + prediction_loss

    losses = {
            'total_loss':total_loss,
            'prediction_loss':prediction_loss,
            'reg_loss':reg_losses
        }

    return losses
