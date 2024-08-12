import numpy as np
from scipy.linalg import toeplitz
import scipy.special
import scipy.stats
import scipy
import torch
from einops import rearrange, reduce, repeat
from torch import nn

# local imports
import src.utils as utils


class Vonmises_Interpolation(nn.Module):

    def __init__(self, cfg, params):
        super(Vonmises_Interpolation, self).__init__() 
        self.cfg = cfg
        self.params = params
        self.vonmises_concentration = cfg.ROTATION.vms_interp_conc
        self.device = params['device']
        self.num_bins = cfg.ROTATION.num_bins

        # pre computation in numpy and scipy
        self.logprior_rotate_matrix, self.pose_angle_bins_toeplitz,\
                self.pose_angle_bins, self.posterior_sum_template = self.prior_matrix()

        max_yaw = cfg.ROTATION.max_yaw
        self.angles_yaw = np.arange(-max_yaw, max_yaw + 1, 1)

        # pre computation in numpy and scipy
        self.population_vector = self.calc_population_vector()
        self.pre_calc_bins = self.pre_calculate_better_bins(self.angles_yaw)
        self.transform_matrices = self.pre_calculate_transform(self.pre_calc_bins)

        # epoch computation in torch after this
        self.logprior_rotate_matrix = torch.tensor(self.logprior_rotate_matrix,\
                dtype=torch.float32, device=self.device)
        self.posterior_sum_template = torch.tensor(self.posterior_sum_template,\
                dtype=torch.float32, device=self.device)

        self.pose_angle_bins = torch.tensor(self.pose_angle_bins, device = self.device)
        self.posterior_sum_template_log = torch.log(self.posterior_sum_template)
        self.angles_yaw = torch.tensor(self.angles_yaw, device=self.device)
        self.pre_calc_bins = torch.tensor(self.pre_calc_bins, device=self.device)

    def calc_population_vector(self):
        """
            create a set of population vector, each being the cos and sin of an angle bin.
        """
        angle_list = np.arange(0, 2*np.pi, 2 * np.pi / self.num_bins)
        cos_population = np.cos(angle_list)
        sin_population = np.sin(angle_list)
        population_vector = np.stack([cos_population, sin_population], 0)
        return torch.tensor(population_vector, device=self.device)

    def prior_matrix(self):
        """
            This prior on the pose estimation is designed with the assumption that the change in
            angle is always small (at 10 degree resolution for our case) and will be most likely to
            not shange pose. This however is bad for large object pose shifts and may cause a bias
            towards estimating the pose change as being very close to 0.
        """

        thetas = np.arange(0, 2*np.pi, 2 * np.pi / self.num_bins)
        thetas_flip = np.zeros(thetas.shape) 
        thetas_flip[1:] = np.flip(thetas[1:]) 

        vm = scipy.stats.vonmises(self.cfg.ROTATION.vms_prior_conc)
    
        logprior_rotate = vm.logpdf(thetas)
        logprior_rotate_flip = vm.logpdf(thetas_flip) 
        logprior_rotate_matrix = toeplitz(logprior_rotate_flip, logprior_rotate)
        
        logprior_rotate_matrix -= scipy.special.logsumexp(logprior_rotate_matrix, (0,1), keepdims=True)
         
        thetas_toeplitz = toeplitz(thetas_flip, thetas)

        # a binary addition template for pose poaterior estimation 
        sum_template = np.zeros([self.num_bins, self.num_bins, self.num_bins], dtype=float)
        for idx, theta in enumerate(thetas):
            sum_template[:,:,idx] = thetas_toeplitz == theta
        
        return logprior_rotate_matrix, thetas_toeplitz, thetas, sum_template

    def pre_calculate_better_bins(self, angles):
        """
        # if camera rotates to the left, the object rotates to the right relative to the camera.
        # This means the original profile of log likelihood rotates to the right.
        # Our goal is to interpolate the log likelihood at the normal grid (which is angles degree
        # counter-clockwise relative to the old grids)
        """
        pre_calc_bins = []
        for a in range(angles.shape[0]):
            new_bins = self.pose_angle_bins + angles[a]*np.pi/180 ## angles are in degree and theta bins in radians

            diff_thetas = self.pose_angle_bins + angles[a]*np.pi/180
            diff_thetas_flip = np.zeros(diff_thetas.shape)
            diff_thetas_flip[1:] = np.flip(diff_thetas[1:])
            diff_thetas_flip[0] = diff_thetas[0]
            pairwise_diff = toeplitz(diff_thetas_flip, diff_thetas)

            pre_calc_bins.append(pairwise_diff)
        
        return np.stack(pre_calc_bins, 0)

    def pre_calculate_transform(self, pre_calc_bins):
        transformations = []
        vm = scipy.stats.vonmises(self.vonmises_concentration)
        inverse_mat = np.linalg.inv(vm.pdf(self.pose_angle_bins_toeplitz))
        for p in range(len(pre_calc_bins)):
            transformations.append(torch.tensor(np.dot(vm.pdf(pre_calc_bins[p]), inverse_mat),\
                    dtype=torch.float, device=self.device))
        return torch.stack(transformations,dim=0)

    def objrot(self, inp, yaw):
        """
            ## forward for object motion relative motion
            ## shapes - input: batch x slot x nbins; yaw: batch x slots
        """
        round_yaw = torch.round(yaw*(180/np.pi))
        index = torch.abs(torch.min(self.angles_yaw))+round_yaw
        index = index.to(dtype=torch.long)
        output = torch.matmul(self.transform_matrices[index], inp[:,:,:,None])[:,:,:,0]
        return output


    def posterior_matrix(self, loglikelihood1, loglikelihood2):
        """
            # The loglikelihood1 and loglikelihood2 are the log likelihoods of pose for each
            # object in the view. Here we assume that objects in loglikelihood1 have 
            # already been matched to objects in loglikelihood2. This should be achieved
            # by weighting based on similarity between latent codes.
        """ 
        loglikelihood = loglikelihood1[:,:,:,None] + loglikelihood2[:,:,None,:] 
        logposterior = loglikelihood + self.logprior_rotate_matrix
        # normalize
        logposterior = logposterior - torch.logsumexp(logposterior, (2,3), keepdim=True)
        logposterior_rotate = torch.logsumexp(logposterior[:,:,:,:,None] +\
                self.posterior_sum_template_log, (2,3))
        
        posterior_rotate = torch.exp(logposterior_rotate) #size: batch x object x bins
        expected_rot_vec = torch.sum(posterior_rotate[:,:,None,:] * self.population_vector, -1)

        # a tensor of size: batch x object x 2
        # Normalize. First add a small epsilon corresponding to not rotating, in case
        # the expected rotation vector becomes close to 0.
        expected_rot_vec = expected_rot_vec + torch.tensor([1e-8, 0.0], device=self.params['device'])
        expected_rot_vec = expected_rot_vec / torch.norm(expected_rot_vec, dim=2)[:, :, None]

        expected_rot_vec = torch.min(torch.max(
            expected_rot_vec, torch.tensor([-1.0], device=self.params['device'])),
                                     torch.tensor([1.0], device=self.params['device']))
        return expected_rot_vec, logposterior_rotate # posterior_rotate, 

    def forward(self, inp, yaw):
        """
        shapes: yaw - batch x 1, input - batch x slots x num_bins
        Input: a tensor of size batch by feature
        yaw: a 1-dimensional tensor

        From theta bins - yaw angle bin find the coefficients, then use those coefficients to
             find the new posteriors
        If yaw is positive, it means the camera rotates to the left.
        This means an object with a pose at zero degree will now appear at -yaw degree in the new view.
        We want to interpolate what the distribution of pose look like in the new grid after camera rotation.
        So we essentially interpolate the values at the bins equal to the original bins plus yaw degrees.
        """
        round_yaw = torch.round(yaw*(180/np.pi))
        ## yaw in radians converting to degrees for easy conversion as the steps are in 1 degree values
        index = torch.abs(torch.min(self.angles_yaw))+round_yaw
        index = index.to(dtype=torch.long)
        output = torch.matmul(self.transform_matrices[index][:, None, :,:], inp[:,:,:,None])[:,:,:,0]
        return output

# Defining the warping class 
class Warping(nn.Module):
    def __init__(self, cfg, params):
        super(Warping, self).__init__() 
        self.params = params
        self.cfg = cfg
        self.device = params['device']
        self.IJ = params['IJ']
        self.f, self.t, self.x_correction, self.y_correction = params['cam_correction']
        self.num_bins = cfg.ROTATION.num_bins
        self.vonmises_interp = Vonmises_Interpolation(cfg, params)

    def calculate_camera_matrix(self, object_loc1, object_loc2, object_pose1, object_pose2,
                                 M1, M2, motion1, motion2, matching_score):

        """
        Object position warping
        shapes - object_loc - b x (s-1) x 3, object_pose - b x (s-1) x num_bins, motion - b x (s-1) x 3
        M1 is the rotation matrix necessary to apply to pixel from frame 1 to warp frame 1 to frame 2
        M2 is the one from frame 2 to frame 3
        """

        # adding an empty row for easy multiplication with camera rotation matrix
        p1 = torch.cat((object_loc1, torch.ones([object_loc1.shape[0],object_loc1.shape[1], 1],\
                                  device = self.device)), 2)

        # estimated location of each object from view 1 in view 2.
        p1c2 = rearrange(torch.matmul(M1[:,None,:,:], p1[:,:,:,None]),\
                'batch num_obj num_slots 1 -> batch num_obj num_slots')
        
        # Movement of object from the percepective of frame 2.
        object_translation = (object_loc2 - p1c2[:,:,:3]) * matching_score[:,:,None]

        T21 = torch.eye(4, device = self.device).repeat(object_loc1.shape[0],\
                object_loc1.shape[1], 1, 1)
        T21[:,:,0:3, 3] = object_translation
        
        # This operation makes the coordinate of any pixel to be relative to the object center
        recenter = torch.eye(4, device = self.device).repeat(object_loc1.shape[0],\
                object_loc1.shape[1], 1, 1)
        recenter[:,:,:3,3] = - object_loc2

        ## object pose warping
        if motion1.shape[1] == 3:
            rotation_idx = 2 # the motion parameter was provided as x,y translation and yaw
        else:
            rotation_idx = 3 # the motion parameter is provided as x,y,z translation and yaw, (roll), pitch

        # pose1 wrt cam2
        resampled_object_pose1 = self.vonmises_interp(object_pose1, -1*motion1[:,rotation_idx])
        resampled_object_pose1 = torch.nn.functional.log_softmax(resampled_object_pose1, dim=2)

        expected_rotate_vec, logposterior_rotate = self.vonmises_interp.posterior_matrix(\
                resampled_object_pose1, object_pose2) ## object pose 1-2 change wrt cam1

        rotate_cos = expected_rotate_vec[:,:,0]
        rotate_sin = expected_rotate_vec[:,:,1]
        rotate_angle_r21 = torch.sign(rotate_sin)*torch.acos(rotate_cos) * (matching_score + 1e-8)
        r21 = utils.object_rotation_uponly(torch.cos(rotate_angle_r21),\
                torch.sin(rotate_angle_r21), device = self.device)
        
        """
        # the first two dimensions of self.posterior_sum_template is essentially a rotation matrix
        # that we can multiply from the right side to a vector indicating probabilty
        # of current pose to get the probability of pose at the next moment.
        # the third dimension corresponds to different rotation angles.
        # We can therefore use it to rotate the pose
        # Then we further weight the resulting pose by the probability of each rotation angle
        # Below, we implement it in log scale
        """

        # ensures that object_pose3o2 is still differentiable wrt to the pose at frame 1 and 2.
        object_pose3o2 = torch.logsumexp(object_pose2[:,:,:,None,None] + self.vonmises_interp.\
                posterior_sum_template_log + logposterior_rotate[:,:,None,None,:], (2,4))
        
        # the interpolation above is not differentiable to motion2. But we don't need to learn
        # anything with respect to motion2. It will still be differentiable
        # to pose1 and pose2
        object_pose3o3 = self.vonmises_interp(object_pose3o2, -motion2[:,rotation_idx])
        
        object_pose3o3 = torch.nn.functional.log_softmax(object_pose3o3, dim=2)
        
        object_loc3p = object_loc2 + object_translation 
        T3 = torch.eye(4, device = self.device).repeat(object_loc1.shape[0], object_loc1.shape[1], 1, 1)
        T3[:,:,0:3,3] = object_loc3p

        T3shape = object_loc3p.shape
        T3c3 = torch.ones((T3shape[0], T3shape[1], T3shape[2]+1), device=self.device)
        T3c3[:,:,0:3] = object_loc3p
        object_loc3pc3 = torch.matmul(M2, T3c3.permute(0,2,1)) #converting obect location3 prediction to camera3 coord
        object_loc3pc3 = object_loc3pc3.permute(0,2,1)[:,:,0:3]

        # constrastive loss

        obj_M =  M2[:,None,:,:] @ T3 @ r21 @ recenter 
        if(torch.isnan(obj_M).any()):
            print("obj_M has nan")
            print(obj_M, M1, M2, motion1, motion2)
        # First move pixel to be centered at object center. then rotate them,
        # then move object center to be the predicted new center, then apply camera rotation
        # batch x object x 4 x 4

        return obj_M, (object_translation, expected_rotate_vec), object_loc3pc3, object_pose3o3

    def pixel_to_camera(self, d1):
        """
        ## coordinate directions: X: right, Y: down, Z: away
        ## coordinate directions MOVI: X: , Y: , Z: 
        ## target coordinates: X - right, Y - away, Z - up 
        """
        Z = d1*self.t
        X = self.x_correction*Z
        Y = self.y_correction*Z

        ### NOTE: MOVI VERISON:
        if self.cfg.EXPERIMENT.exp_type == "movi":
            Yo = -Y
            Zo = -Z
        ### NOTE: NON-MOVI VERISON:
        else:
            Yo = Z
            Zo = -Y

        return(torch.stack((X, Yo, Zo, torch.ones_like(d1)), dim=-1) )

    def camera_to_pixel_slots(self, cam_coord):
        """
        ## coordinates: X- right, Y - away, Z - up
        ## target coordinates: X - right, Y - down, Z - does not exist 
        ## input shapes - cam_coord: b x slots x W x H x 3
        #NOTE: this is wrong above! the last dim of input shapes is of size 4! not 3. Why? - John
        """
        X = cam_coord[:,:,:,:,0]

        if self.cfg.EXPERIMENT.exp_type == "movi":
            Y = - cam_coord[:,:,:,:,1] ## taking the projection axis, ie the Z axis #NOTE: MOVI
            Z_orig = -cam_coord[:,:,:,:,2] ## taking the depth direction, ie the Y axis #NOTE: MOVI
        else:
            Y = - cam_coord[:,:,:,:,2] ## taking the projection axis, ie the Z axis #NOTE: NON-MOVI
            Z_orig = cam_coord[:,:,:,:,1] ## taking the depth direction, ie the Y axis #NOTE: NON-MOVI

        Z = torch.abs(Z_orig) + self.cfg.DEPTH.epsilon_depth
        x = self.f * torch.div(X,Z)
        y = self.f * torch.div(Y,Z)

        faulty_vals = False
        faulty_idxs = []
        d = torch.norm(cam_coord[:,:,:,:,:3], dim=4) * torch.sign(Z_orig +\
                self.cfg.DEPTH.epsilon_depth)

        return torch.stack((x, y, d), dim=2), faulty_vals, faulty_idxs

    def predict_next_position(self, cam, latent, motion, M, depth2, attention2, matching_score):
        cam1, cam2, cam3 = cam
        latent1, latent2 = latent
        M1, M2 = M
        motion1, motion2 = motion

        # It is assumed that codes in view 1 and view 2 have been aligned based on
        # identity code matching.
        ## we start with camera 1 based obj coordinates
        object_loc1 = latent1[:,:,:3]
        object_loc2 = latent2[:,:,:3]

        object_pose1 = latent1[:,:,3:3+self.num_bins]
        object_pose2 = latent2[:,:,3:3+self.num_bins]

        object_code1 = latent1[:,:,3+self.num_bins:]
        object_code2 = latent2[:,:,3+self.num_bins:]
        # object_code3 = latent3[:,:,3+self.num_bins:]
        
        c2pos = self.pixel_to_camera(depth2)
        # shape: batch x height x width x 4
        obj_M, object_motion, object_loc3p, object_pose3p = self.calculate_camera_matrix(\
            object_loc1, object_loc2, object_pose1, object_pose2, M1, M2, motion1, motion2,\
            matching_score)

        object_motion = torch.cat(object_motion, dim = 2)
        # adding the object motion for the non moving object detection mask 
        # for wall, floor, and sky
        object_motion = torch.cat([object_motion, torch.tensor([[[0,0,0,1,0]]],\
                device = self.device).repeat(object_motion.shape[0],1,1)], dim=1)
        
        objectmotion_matrices = torch.cat((obj_M, M2.unsqueeze(1)), 1)

        ## when splitting each pixel into num_slots + 1
        if(torch.isnan(objectmotion_matrices).any()):
            print("object projection matrix has nan")

        c3pos = torch.matmul(objectmotion_matrices[:,:,None,None,:,:],\
                c2pos[:,:,:,:,:,None])[:,:,:,:,:,0]
        c3pos_slotavg = torch.mean(c3pos[:,:-1]*attention2[:,:-1,:,:,None], dim=(2,3))
        point3, fval, fvalidxs = self.camera_to_pixel_slots(c3pos)

        return point3[:,:,0:2,:,:], point3[:,:,2,:,:], object_motion, c2pos,\
            object_loc3p, object_pose3p, c3pos_slotavg

    def rbf_weighting(self, frame2, frame3_map, depth3, attention2):

        shift_dist = torch.sign(torch.sum(frame3_map*attention2[:,:,None,:,:], dim=1)\
                - self.IJ)[:,1] * torch.sqrt(torch.sum((torch.sum(frame3_map *\
                    attention2[:,:,None,:,:], dim=1) - self.IJ)**2, dim = (1)))

        frame_3p, weighting = utils.warping_sparse_batch_slotwise(self.cfg, frame2, frame3_map,\
                self.IJ[0], depth3, attention2, self.device)

        return frame_3p, weighting, shift_dist

    def warp(self, cam, latent, M, motion, frame2, depth2, attention2, matching_score):
        """
        Main warping function forward call
        shapes - cam- 3 x b x 9; 
        latent - 3 x b x slots x idsize; -- TO FIX
        M - 2 x b x 4 x 4;
        motion - 2 x b x 3
        """

        ### checking for nan/inf in latent output
        for i in range(len(latent)):
            if(torch.isnan(latent[i]).any()):
                print("latent {} has nan".format(i))
                print('image range',torch.min(frame2), torch.max(frame2))
            if(torch.isinf(latent[i]).any()):
                print("latent {} has inf".format(i))
                print('image range',torch.min(frame2), torch.max(frame2))
        
        frame_map, depth3p, object_motion, c2pos, object_loc3p,\
            object_pose3p, object_loc3p_avg = self.predict_next_position(
            cam, latent, motion, M, depth2, attention2, matching_score)

        frame_3p, weighting, shift_dist = self.rbf_weighting(
            torch.cat([frame2[:,None,:,:,:].expand(-1,depth3p.shape[1],-1,-1,-1).contiguous(),
                       depth3p[:,:,None,:,:].contiguous()], dim=2),
            frame_map, depth3p, attention2)
        warped_depth3 = frame_3p[:,3,:,:] # predicted depth on frame 2 by warping
        frame_3p = frame_3p[:,:3,:,:]

        return frame_3p, warped_depth3, weighting, object_motion, \
            shift_dist, c2pos, object_loc3p, object_pose3p, object_loc3p_avg

    def forward(self, data_batch, segment_outputs):
        cams = data_batch['cams']
        frame2 = data_batch['frames'][1]
        seg_out1, seg_out2, seg_out3 = segment_outputs['seg_out1'], segment_outputs['seg_out2'],\
            segment_outputs['seg_out3']

        latent1, latent2, latent3 = seg_out1['latent_representation_transformed'],\
            seg_out2['latent_representation_transformed'],\
            seg_out3['latent_representation_transformed']
        latent1_aligned = segment_outputs['representation_aligned1']
        latents = [latent1_aligned, latent2]

        M = data_batch['transformation_matrices']
        attention2 = seg_out2['attention_maps_softmax']
        matching_score = segment_outputs['matching_score_12']
        motion = [data_batch['motion_params1'], data_batch['motion_params2']]
        depth2 = data_batch['depth'][1]

        frame_3p, warped_depth3, weighting, object_motion,\
        shift_dist, c2pos, object_loc3p, object_pose3p,\
        object_loc3p_avg = self.warp(cams, latents, M, motion,\
        frame2, depth2, attention2, matching_score)

        weighting_base = None

        output = {
                'frame3_predicted':frame_3p,
                'depth3_warped':warped_depth3,
                'weighting_matrix':weighting,
                'weighting_base':weighting_base,
                'object_motion':object_motion,
                'pixel_shift_dist':shift_dist,
                'cam2_position':c2pos, 
                'object3_loc_predicted':object_loc3p,
                'object3_pose_predicted':object_pose3p,
                'object3_loc_avg_predicted':object_loc3p_avg
            }

        return output

