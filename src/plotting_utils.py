import numpy as np
import torch
import os
import gc
import multiprocessing as mp
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
from matplotlib import rcParams

# local imports
from src.utils import numpify, label_maps_to_colour


def numpify_all_outputs(outputs):
    for key in outputs.keys():
        if(key != "segment_outputs"):
            for key_in in outputs[key]:
                if(torch.is_tensor(outputs[key][key_in])):
                    outputs[key][key_in] = numpify(outputs[key][key_in])
        else:
            for key_in in outputs[key]:
                if("seg_out" in key_in):
                    for key_in2 in outputs[key][key_in]:
                        outputs[key][key_in][key_in2] =\
                            numpify(outputs[key][key_in][key_in2])
                else:
                    outputs[key][key_in] = numpify(outputs[key][key_in])
    return outputs

def plot_immediate_results(cfg, params, outputs, epoch, batch):
    # scene_num = outputs['X_updated']['scene_num']
    plot_path = os.path.join(params["exp_dir_path"],
        "train_viz/imm_results/viz_ep:{}_batch:{}.pdf".format(epoch, batch))
    immediate_plots(cfg, params, plot_path, outputs)

def plot_immediate_results_valid(cfg, params, outputs, epoch, batch):
    # scene_num = outputs['X_updated']['scene_num']
    plot_path = os.path.join(params["exp_dir_path"],
        "val_viz/imm_results/viz_ep:{}_batch:{}.pdf".format(epoch, batch))
    immediate_plots(cfg, params, plot_path, outputs)

def plot_immediate_results_test(cfg, params, outputs, epoch, batch):
    # scene_num = outputs['X_updated']['scene_num']
    plot_path = os.path.join(params["test_dir_path"],
        "imm_results/viz_ep:{}_batch:{}.pdf".format(epoch, batch))
    immediate_plots(cfg, params, plot_path, outputs)

def plot_multiple_attmaps_in_one(cfg, params, map_att):
    # define colors for n slots
    # print("map att shape: ", map_att.shape)
    n_slots = map_att.shape[1]
    colors = [[255, 156, 38], [51, 255, 92], [255, 83, 25], [0, 176, 255], [255, 13, 47]]
    ones_att = np.ones((map_att.shape[0], map_att.shape[2], map_att.shape[3]))
    # zero_att = np.zeros((map_att.shape[0], 3, map_att.shape[2], map_att.shape[3]))
    colored_att = np.zeros((map_att.shape[0], map_att.shape[2], map_att.shape[3], 3))
    # print(ones_att.shape, (colors[0][0]*ones_att).shape)
            # map_att[:,0,:,:]*(colors[0][0]*ones_att).shape)
    for i in range(n_slots):
        colored_att[:,:,:, 0] += map_att[:,i,:,:]*(colors[i][0]*ones_att)
        colored_att[:,:,:, 1] += map_att[:,i,:,:]*(colors[i][1]*ones_att)
        colored_att[:,:,:, 2] += map_att[:,i,:,:]*(colors[i][2]*ones_att)
    return colored_att


def immediate_plots(cfg, params, plot_path, outputs):
    pred_frm3 = np.transpose(outputs['predicted_outputs']['predicted_img3'], (0,2,3,1))
    actual_frm3 = np.transpose(outputs['X_updated']['frames'][2], (0,2,3,1))
    actual_frm2 = np.transpose(outputs['X_updated']['frames'][1], (0,2,3,1))

    pred_frm3_warp = np.transpose(outputs['predicted_outputs']['predicted_img3_warp'], (0,2,3,1))
    pred_frm3_imag_weighted = np.transpose(outputs['predicted_outputs']\
            ['predicted_img3_imag'], (0,2,3,1))
    pred_frm3_imag = np.transpose(outputs['imag_outputs']['img3_imag'], (0,2,3,1))

    # plot warping wight 
    weight_warp = outputs['predicted_outputs']['warping_weight']
    # plot imagination weight
    weight_imag = outputs['predicted_outputs']['imag_weight']
    # plot log depth
    depth3 = np.transpose(outputs['X_updated']\
            ['depth'][2]/cfg.DEPTHNET.depth_range, (0,2,3,1))
    map_att3 = outputs['segment_outputs']['seg_out2']\
            ['attention_maps_softmax']
    # map_att3_combined = plot_multiple_attmaps_in_one(cfg, params, map_att3)

    warped_depth3 = outputs['warping_outputs']['depth3_warped']

    batchsize = pred_frm3.shape[0]
    ncols = 9 + cfg.TRAIN.num_slots_withbg
    fig, ax = plt.subplots(nrows=batchsize, ncols=ncols, figsize=(24, batchsize*2), dpi=80)

    for b in range(batchsize):
        ax[b,0].imshow(actual_frm2[b])
        # ax[b,1].set_title('actual frame 2')
        ax[b,0].get_xaxis().set_ticks([])
        ax[b,0].get_yaxis().set_ticks([])
        scene_num = outputs['X_updated']['scene_num'][b][0]
        ax[b,0].title.set_text("scene #"+str(scene_num))

        ax[b,1].imshow(actual_frm3[b])
        ax[b,1].set_title('gt frame 3')
        ax[b,1].get_xaxis().set_ticks([])
        ax[b,1].get_yaxis().set_ticks([])

        ax[b,2].imshow(pred_frm3[b])
        ax[b,2].set_title('pred frame 3')
        ax[b,2].get_xaxis().set_ticks([])
        ax[b,2].get_yaxis().set_ticks([])

        ax[b,3].imshow(pred_frm3_warp[b])
        ax[b,3].set_title('warped frame 3')
        ax[b,3].get_xaxis().set_ticks([])
        ax[b,3].get_yaxis().set_ticks([])

        # ax[b,4].imshow(pred_frm3_imag_weighted[b])
        # # ax[b,2].set_title('predicted imagined weighted frame 3')
        # ax[b,4].get_xaxis().set_ticks([])
        # ax[b,4].get_yaxis().set_ticks([])

        ax[b,4].imshow(pred_frm3_imag[b])
        ax[b,4].set_title('imag frame 3')
        ax[b,4].get_xaxis().set_ticks([])
        ax[b,4].get_yaxis().set_ticks([])

        ax[b,5].imshow(weight_warp[b])
        ax[b,5].set_title('warp weight')
        ax[b,5].get_xaxis().set_ticks([])
        ax[b,5].get_yaxis().set_ticks([])

        ax[b,6].imshow(weight_imag[b])
        ax[b,6].set_title('imag weight')
        ax[b,6].get_xaxis().set_ticks([])
        ax[b,6].get_yaxis().set_ticks([])

        im = ax[b,7].imshow(depth3[b])
        ax[b,7].set_title('depth 3')
        ax[b,7].get_xaxis().set_ticks([])
        ax[b,7].get_yaxis().set_ticks([])
        plt.colorbar(im, ax=ax[b,7])

        im = ax[b,8].imshow(warped_depth3[b])
        ax[b,8].set_title('warp depth 3')
        ax[b,8].get_xaxis().set_ticks([])
        ax[b,8].get_yaxis().set_ticks([])
        plt.colorbar(im, ax=ax[b,8])

        # plot the attention values for each slot
        for slot_idx in range (cfg.TRAIN.num_slots_withbg):
            col_idx = 9 + slot_idx
            ax[b,col_idx].imshow(map_att3[b, slot_idx])
            ax[b,col_idx].get_xaxis().set_ticks([])
            ax[b,col_idx].get_yaxis().set_ticks([])

    plt.savefig(plot_path)
    plt.clf()
    plt.close()

def plot_losses(cfg, params, loss_Y, loss_X, epoch):
    # plot the loss values 
    plot_path = os.path.join(params["exp_dir_path"], "plots/loss_plot.pdf")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(loss_X, loss_Y, color='#6d6d6d')
    ax.set_xlabel('Bacth Number')
    ax.set_ylabel('Loss value')
    fig.savefig(plot_path)

def plot_losses_valid(cfg, params, loss_Y, loss_X, epoch):
    # plot the loss values 
    plot_path = os.path.join(params["exp_dir_path"], "plots/loss_plot_valid.pdf")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(loss_X, loss_Y, color='#6d6d6d')
    ax.set_xlabel('Bacth Number')
    ax.set_ylabel('Loss value')
    fig.savefig(plot_path)
    plt.close()

def plot_losses_test(cfg, params, loss_Y, loss_X, epoch):
    # plot the loss values 
    plot_path = os.path.join(params["exp_dir_path"], "plots/loss_plot_test.pdf")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(loss_X, loss_Y, color='#6d6d6d')
    ax.set_xlabel('Bacth Number')
    ax.set_ylabel('Loss value')
    fig.savefig(plot_path)

def plot_all_losses(cfg, params, losses, loss_X, epoch):
    # plot the loss values 
    # TODO: plot all the losses seperately because different scales
    plot_path = os.path.join(params["exp_dir_path"], "plots/all_loss_plot.pdf")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = cm.rainbow(np.linspace(0, 1, len(losses.keys())))
    for key, c in zip(losses.keys(), colors):
        if('pixel' in key):
            continue
        ax.plot(losses['key'], loss_Y, color=c, label = key)
    ax.set_xlabel('Bacth Number')
    ax.set_ylabel('Loss value')
    ax.legend(loc='upper right', fontsize='small')
    fig.savefig(plot_path)

def visualize_outputs_multithread(cfg, params, outputs, plot_title, plot_path):
    batch_size = cfg.TRAIN.batch_size if params['test'] else cfg.TEST.batch_size_test
    irange = range(batch_size)
    pool = mp.Pool(processes=16)
    pool_i = partial(visualize_outputs_singlethread, cfg, params, outputs, plot_path, plot_title)
    pool.map(pool_i, irange)
    pool.close()
    pool.join()
    pool.terminate()
    gc.collect()

def visualize_outputs_singlethread(cfg, params, outputs, plot_path_out, plot_title, thread_val): 
    scene_num = outputs['X_updated']['scene_num'][thread_val]
    epoch, batch = plot_title, plot_path_out
    plot_path = os.path.join(params["exp_dir_path"],
        "train_viz/viz_scene:{}_ep:{}_batch:{}_thread:{}.pdf".\
        format(scene_num, epoch, batch, thread_val))
    # plot_path = plot_path_out.format(scene_num, thread_val)

    fig, ax = plt.subplots(nrows=11, ncols=4, figsize = (16, 44)) 

    pred_frm3 = np.transpose(outputs['predicted_outputs']\
            ['predicted_img3'], (0,2,3,1))
    pred_frm3_warp = np.transpose(outputs['predicted_outputs']\
            ['predicted_img3_warp'], (0,2,3,1))
    pred_frm3_imag = np.transpose(outputs['predicted_outputs']\
            ['predicted_img3_imag'], (0,2,3,1))

    actual_frm1 = np.transpose(outputs['X_updated']['frames'][0], (0,2,3,1))
    actual_frm2 = np.transpose(outputs['X_updated']['frames'][1], (0,2,3,1))
    actual_frm3 = np.transpose(outputs['X_updated']['frames'][2], (0,2,3,1))

    # plot the predicted image, GT image for frame 3, warped component, imagined component
    # | 1 _ 1 _ 1 _ 1 |
    #NOTE: the following changes were because apparently the images were of the format (3, 128, 128) instead of (128, 128, 3), which matplot doesnt like
    ax[0,0].imshow(pred_frm3[thread_val])
    ax[0,0].set_title('predicted frame 3')
    ax[0,0].get_xaxis().set_ticks([])
    ax[0,0].get_yaxis().set_ticks([])

    ax[0,1].imshow(actual_frm3[thread_val])
    ax[0,1].set_title('actual frame 3')
    ax[0,1].get_xaxis().set_ticks([])
    ax[0,1].get_yaxis().set_ticks([])

    ax[0,2].imshow(pred_frm3_warp[thread_val])
    ax[0,2].set_title('predicted warped frame 3')
    ax[0,2].get_xaxis().set_ticks([])
    ax[0,2].get_yaxis().set_ticks([])

    ax[0,3].imshow(pred_frm3_imag[thread_val])
    ax[0,3].set_title('predicted imagined frame 3')
    ax[0,3].get_xaxis().set_ticks([])
    ax[0,3].get_yaxis().set_ticks([])

    # plot the predicted depth, GT depth for frame 3 
    # | 1 _ 1 _ x _ x |
    pred_depth3 = outputs['predicted_outputs']\
            ['predicted_depth3']/cfg.DEPTHNET.depth_range
    if(np.max(pred_depth3)>1):
        print("pred depth min and max", np.min(pred_depth3), np.max(pred_depth3))
    ax[1,0].imshow(pred_depth3[thread_val])
    ax[1,0].set_title('predicted depth 3')
    ax[1,0].get_xaxis().set_ticks([])
    ax[1,0].get_yaxis().set_ticks([])

    depth1 = np.transpose(outputs['X_updated']\
            ['depth'][0]/cfg.DEPTHNET.depth_range, (0,2,3,1))
    depth2 = np.transpose(outputs['X_updated']\
            ['depth'][1]/cfg.DEPTHNET.depth_range, (0,2,3,1))
    depth3 = np.transpose(outputs['X_updated']\
            ['depth'][2]/cfg.DEPTHNET.depth_range, (0,2,3,1))
    # print("depth min and max", np.min(depth3), np.max(depth3))

    ax[1,1].imshow(depth3[thread_val])
    ax[1,1].set_title('inferred depth 3')
    #TODO: add a color bar for depth
    ax[1,1].get_xaxis().set_ticks([])
    ax[1,1].get_yaxis().set_ticks([])

    # plot the actual frame1, frame2, frame3
    # | 1 _ 1 _ 1 _ x |

    ax[2,0].imshow(actual_frm1[thread_val])
    ax[2,0].set_title('actual frame 1')
    ax[2,0].get_xaxis().set_ticks([])
    ax[2,0].get_yaxis().set_ticks([])

    ax[2,1].imshow(actual_frm2[thread_val])
    ax[2,1].set_title('actual frame 2')
    ax[2,1].get_xaxis().set_ticks([])
    ax[2,1].get_yaxis().set_ticks([])

    ax[2,2].imshow(actual_frm3[thread_val])
    ax[2,2].set_title('actual frame 3')
    ax[2,2].get_xaxis().set_ticks([])
    ax[2,2].get_yaxis().set_ticks([])

    # plot the inferred depth1, depth2, depth3
    # | 1 _ 1 _ 1 _ x |
    ax[3,0].imshow(depth1[thread_val])
    ax[3,0].set_title('inferred depth 1')
    ax[3,0].get_xaxis().set_ticks([])
    ax[3,0].get_yaxis().set_ticks([])

    ax[3,1].imshow(depth2[thread_val])
    ax[3,1].set_title('inferred depth 2')
    ax[3,1].get_xaxis().set_ticks([])
    ax[3,1].get_yaxis().set_ticks([])

    ax[3,2].imshow(depth3[thread_val])
    ax[3,2].set_title('inferred depth 3')
    ax[3,2].get_xaxis().set_ticks([])
    ax[3,2].get_yaxis().set_ticks([])

    # plot the actual depth1, depth2, depth3
    # | 1 _ 1 _ 1 _ x |

    gt_depth1 = outputs['X_updated']\
            ['depths_gt'][0]/cfg.DEPTHNET.depth_range
    gt_depth2 = outputs['X_updated']\
            ['depths_gt'][1]/cfg.DEPTHNET.depth_range
    gt_depth3 = outputs['X_updated']\
            ['depths_gt'][2]
    ax[4,0].imshow(gt_depth1[thread_val])
    ax[4,0].set_title('actual depth 1')
    ax[4,0].get_xaxis().set_ticks([])
    ax[4,0].get_yaxis().set_ticks([])

    ax[4,1].imshow(gt_depth2[thread_val])
    ax[4,1].set_title('actual depth 2')
    ax[4,1].get_xaxis().set_ticks([])
    ax[4,1].get_yaxis().set_ticks([])

    ax[4,2].imshow(gt_depth3[thread_val])
    ax[4,2].set_title('actual depth 3')
    ax[4,2].get_xaxis().set_ticks([])
    ax[4,2].get_yaxis().set_ticks([])

    # plot the warping weighting for each object
    # | 1 _ 1 _ 1 _ 1 |

    # plot the warping component for each object
    # | 1 _ 1 _ 1 _ 1 |

    # plot the imagination weighting for each object
    # | 1 _ 1 _ 1 _ 1 |

    # plot the attention map for each object - frame 2
    # | 1 _ 1 _ 1 _ 1 |
    map_att2 = outputs['segment_outputs']['seg_out2']\
            ['attention_maps_softmax']
    ax[7,0].imshow(map_att2[thread_val, 0])
    ax[7,0].set_title('segment map 0, frame 2')
    ax[7,0].get_xaxis().set_ticks([])
    ax[7,0].get_yaxis().set_ticks([])

    ax[7,1].imshow(map_att2[thread_val, 1])
    ax[7,1].set_title('segment map 1, frame 2')
    ax[7,1].get_xaxis().set_ticks([])
    ax[7,1].get_yaxis().set_ticks([])

    ax[7,2].imshow(map_att2[thread_val, 2])
    ax[7,2].set_title('segment map 2, frame 2')
    ax[7,2].get_xaxis().set_ticks([])
    ax[7,2].get_yaxis().set_ticks([])

    ax[7,3].imshow(map_att2[thread_val, 3])
    ax[7,3].set_title('segment map 3, frame 2')
    ax[7,3].get_xaxis().set_ticks([])
    ax[7,3].get_yaxis().set_ticks([])

    # plot the attention map for each object - frame 3
    # | 1 _ 1 _ 1 _ 1 |
    map_att3 = outputs['segment_outputs']['seg_out2']\
            ['attention_maps_softmax']
    ax[8,0].imshow(map_att3[thread_val, 0])
    ax[8,0].set_title('segment map 0, frame 2')
    ax[8,0].get_xaxis().set_ticks([])
    ax[8,0].get_yaxis().set_ticks([])

    ax[8,1].imshow(map_att3[thread_val, 1])
    ax[8,1].set_title('segment map 1, frame 2')
    ax[8,1].get_xaxis().set_ticks([])
    ax[8,1].get_yaxis().set_ticks([])

    ax[8,2].imshow(map_att3[thread_val, 2])
    ax[8,2].set_title('segment map 2, frame 2')
    ax[8,2].get_xaxis().set_ticks([])
    ax[8,2].get_yaxis().set_ticks([])

    ax[8,3].imshow(map_att3[thread_val, 3])
    ax[8,3].set_title('segment map 3, frame 2')
    ax[8,3].get_xaxis().set_ticks([])
    ax[8,3].get_yaxis().set_ticks([])

    # plot pixel shift, rgb loss pixel wise, depth loss pizel wise
    # | 1 _ 1 _ 1 _ x |
    ax[9,0].imshow(outputs['warping_outputs']['pixel_shift_dist'][thread_val])
    ax[9,0].set_title('pixel shift distance frame 2->3')
    ax[9,0].get_xaxis().set_ticks([])
    ax[9,0].get_yaxis().set_ticks([])

    ax[9,1].imshow(outputs['losses']['rgb_loss_pixels'][thread_val])
    ax[9,1].set_title('rgb val prediction loss per pixel')
    ax[9,1].get_xaxis().set_ticks([])
    ax[9,1].get_yaxis().set_ticks([])

    #NOTE: i just commented the line below out so the code would run (they are buggy)
    #NOTE: TypeError: Invalid shape (128, 128, 20) for image data
    # (20 seems too big(?))  -> 20 is batch size, other 20 mean some incorrect
    # multiplication or reshaping in depth or loss code -- TODO - low priority

    # depth_loss_pixels =outputs['losses']['depth_loss_pixels']
    # print("depth loss pixels", depth_loss_pixels.shape)
    # ax[9,2].imshow(depth_loss_pixels[thread_val]) 
    # ax[9,2].set_title('depth val prediction loss per pixel')

    ax[9,2].imshow(np.zeros((128,128)))
    ax[9,2].set_title('[BUGGY] depth val prediction loss per pixel')
    ax[9,2].get_xaxis().set_ticks([])
    ax[9,2].get_yaxis().set_ticks([])

    # plot the camera positions and object positions for view 1, 2, and 3
    # | 1 _ 1 _ 1 _ x |
    cam_loc = outputs['X_updated']['cams']
    obj_loc = outputs['X_updated']['obj_pos_gt']
    obj_pose = outputs['X_updated']['obj_pose_gt']
    # obj_loc_pred = positions["obj_loc_pred"]
    # obj_pose_pred = positions["obj_pose_pred"]

    colors = ['b', 'g', 'r', 'c', 'y', 'm', 'k']
    for j in range(cam_loc.shape[0]):
        xpt = cam_loc[j,thread_val,0]
        ypt = cam_loc[j,thread_val,1]
        angle = np.sign(cam_loc[j,thread_val,4])*np.arccos(cam_loc[j,thread_val,3])
        ax[10, 0].plot(xpt, ypt, 'o', c=colors[2], label="{0}-({1:.3f},{2:.3f}; {3:.3f})"\
                .format(j, xpt, ypt, angle))
        ax[10, 0].text(xpt*1.001, ypt*1.001, j, fontsize=10)
    ax[10, 0].set_xlim([-2.5,2.5])
    ax[10, 0].set_ylim([-2.5,2.5])
    ax[10, 0].set_aspect(1)
    leg = ax[10, 0].legend(loc='upper right', frameon=False)
    ax[10, 0].set_title("camera location")

    # TODO: there is an error from this code block — IndexError: too many indices for array: array is 3-dimensional, but 4 were indexed: line 346
    # for j in range(obj_loc.shape[0]):
    #     for k in range(obj_loc.shape[2]):
    #         xpt = obj_loc[j,thread_val,k,0]
    #         ypt = obj_loc[j,thread_val,k,1]
    #         angle = np.sign(obj_pose[j,thread_val,k,0])*np.arccos(obj_pose[j,thread_val,k,1])
    #         ax[10, 1].plot(xpt, ypt, 'o', c=colors[k], label="{0}-({1:.3f},{2:.3f}; {3:.3f})".\
    #                 format(j, xpt, ypt, angle))
    #         ax[10, 1].text(xpt*1.001, ypt*1.001, j, fontsize=10)
    # ax[10, 1].set_xlim([-2.5,2.5])
    # ax[10, 1].set_ylim([-2.5,2.5])
    # ax[10, 1].set_aspect(1)
    # leg = ax[10, 1].legend(loc='upper right', frameon=False)
    # ax[10, 1].set_title("object location")

        # TODO: plot the predicted object location
        # for j in range(obj_loc_pred.shape[0]):
            # for k in range(obj_loc_pred.shape[2]):
                # xpt = obj_loc_pred[j,i,k,0]
                # ypt = obj_loc_pred[j,i,k,1]
                # angle = obj_pose_pred[j,i,k]
                # ax[10, 2].plot(xpt, ypt, 'o', c=colors[k], label="{0}-({1:.3f},{2:.3f}; {3:.3f})".\
                        # format(j, xpt, ypt, angle))
                # ax[10, 2].text(xpt*1.001, ypt*1.001, j, fontsize=10)
        # ax[10, 2].set_xlim([-2.5,2.5])
        # ax[10, 2].set_ylim([-2.5,2.5])
        # ax[10, 2].set_aspect(1)
        # leg = ax[10, 2].legend(loc='upper right', frameon=False)
        # ax[10, 2].set_title("object location predicted")
    plt.suptitle(str(plot_title) + ' ' + str(plot_path))
    plt.savefig(plot_path)
    plt.clf()
    plt.close()

def visualize_img(frame, t, loc):
    plt.imshow(frame)
    plt.colorbar()
    plt.title(t)
    plt.savefig(loc)
    plt.clf()
    # plt.show()    

#### Old visualization stuff 
## TODO: cleanup the previous code 
def visualize_posedistribution(likelihood, s, loc, count):
    # likelihood = probabilities['likelihood'] #shape - b x slots x n_bins
    # posterior = probabilities['posterior']

    for i in range(likelihood.shape[0]):
        fig, ax = plt.subplots(likelihood.shape[1])
        for j in range(likelihood.shape[1]):
            if(likelihood.shape[1]>1):
                ax[j].bar(range(likelihood.shape[2]), likelihood[i,j])
            else:
                ax.bar(range(likelihood.shape[2]), likelihood[i,j])
        t = "probs"+str(count)+"-"+str(s[i])+"-"+str(i)
        plt.savefig(loc+t)
        plt.cla()
        plt.close(fig)

def visualize_3d(coords, c1pos, c2pos, pred_coords, loc):
    colour = []
    for x in range(64):
        for y in range(64): 
            colour.append(str(((x/63.)**2 + (y/63.)**2)*0.5))
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(2,2,1, projection = '3d')
    ax.scatter(coords[:,:, 0], coords[:, :, 1], zs = 0, c = colour)
    ax.title.set_text("coords")
    ax = fig.add_subplot(2,2,2, projection = '3d')
    ax.scatter(c1pos[:, :, 0], c1pos[:,:,1], c1pos[:,:,2], c=colour)
    ax.title.set_text("c1pos")
    ax = fig.add_subplot(2,2,3, projection = '3d')
    ax.scatter(c2pos[:, :, 0], c2pos[:,:,1], c2pos[:,:,2], c=colour)
    ax.title.set_text("c2pos")
    ax = fig.add_subplot(2,2,4, projection = '3d')
    ax.scatter(pred_coords[:,:, 0], pred_coords[:, :, 1], zs = 0, c = colour)
    ax.title.set_text("pred_coords")
    plt.show() 

def visualize_multi_img(frame1, frame2, frame_2p, d1, d2, imagine, d_weight, d2i, depth_2p, mask1,
        mask2, s, loc, count=''):
    for i in range(frame1.shape[0]):
        
        fig, ax = plt.subplots(3,3)
        
        fig.suptitle("Scene: "+str(s[i])+"-"+str(i))

        im = ax[0,0].imshow(frame1[i])
        for m in range(mask1[i].shape[0]):
            im = ax[0,0].imshow(mask1[i][m], cmap='jet', alpha=0.2)

        ax[0,0].set_title("frame1")
        
        ax[0,1].imshow(frame2[i])
        for m in range(mask2[i].shape[0]):
            ax[0,1].imshow(mask2[i][m], cmap='jet', alpha=0.2)
        ax[0,1].set_title("frame2")

        im = ax[0,2].imshow(frame_2p[i])
        for m in range(mask2[i].shape[0]):
            im = ax[0,2].imshow(mask2[i][m], cmap='jet', alpha=0.2)
        ax[0,2].set_title("frame 2 predicted")

        im = ax[1,0].imshow(d1[i])
        plt.colorbar(im, ax=ax[1,0])
        ax[1,0].set_title("depth 1")
        
        im = ax[1,1].imshow(d2[i])
        plt.colorbar(im, ax=ax[1,1])
        ax[1,1].set_title("warped depth 2")

        if(imagine.any() == None):
            im = ax[1,2].imshow(d2[i])
            ax[1,2].set_title("depth 2")
        else:
            im = ax[1,2].imshow(imagine[i])
            ax[1,2].set_title("imagined image")
#         plt.colorbar(im, ax=ax[1,2])
        
        im = ax[2,0].imshow(d_weight[i])
        plt.colorbar(im, ax=ax[2,0])
        ax[2,0].set_title("warping weight")
        
#         if not d2i is None:
        im = ax[2,1].imshow(d2i[i])
        plt.colorbar(im, ax=ax[2,1])
        ax[2,1].set_title("imagined depth 2")
        
        
        im = ax[2,2].imshow(depth_2p[i])
        plt.colorbar(im, ax=ax[2,2])
        for m in range(mask2[i].shape[0]):
            ax[2,2].imshow(mask2[i][m], cmap='jet', alpha=0.2)
        ax[2,2].set_title("final predicted depth 2")
        
        t = str(count) + "-" + str(s[i])+"-"+str(i)
        plt.savefig(loc+t)
        plt.cla()
        plt.close(fig)

def visualize_extra(rgblosspixels, depthlosspixels, shiftdistance, grad_before, grad_after,
        grad_3rd, cam_loc, obj_loc, obj_loc_pred, s, loc, count=''):
    
    for i in range(rgblosspixels.shape[0]):
        
        fig, ax = plt.subplots(3,3)
        
        fig.suptitle("Scene: "+str(s[i])+"-"+str(i))

        im = ax[0, 0].imshow(rgblosspixels[i])
        plt.colorbar(im, ax=ax[0, 0])
        ax[0, 0].set_title("rgb loss per pixel")
        
        im = ax[0, 1].imshow(depthlosspixels[i])
        plt.colorbar(im, ax=ax[0, 1])
        ax[0, 1].set_title("depth loss per pixel")

        im = ax[0, 2].imshow(shiftdistance[i])
        plt.colorbar(im, ax=ax[0, 2])
        ax[0, 2].set_title("shift distance per pixel")

        if(grad_before is None):
            grad_before = shiftdistance
            grad_after = shiftdistance
            grad_3rd = shiftdistance

        im = ax[1, 0].imshow(grad_before[i])
        plt.colorbar(im, ax=ax[1, 0])
        ax[1, 0].set_title("map grad before actv")
        
        im = ax[1, 1].imshow(grad_after[i])
        plt.colorbar(im, ax=ax[1, 1])
        ax[1, 1].set_title("map grad after actv")

        im = ax[1, 2].imshow(grad_3rd[i])
        plt.colorbar(im, ax=ax[1, 2])
        ax[1, 2].set_title("map grad ...")


        for j in range(cam_loc.shape[0]):
            xpt = cam_loc[j,i,0]
            ypt = cam_loc[j,i,1]
            ax[2,0].plot(xpt, ypt, 'bo')
            ax[2,0].text(xpt*1.001, ypt*1.001, j, fontsize=10)
            ax[2,0].set_xlim([-2.5,2.5])
            ax[2,0].set_ylim([-2.5,2.5])
        ax[2,0].set_title("camera location")

        for j in range(obj_loc.shape[0]):
            xpt = obj_loc[j,i,0]
            ypt = obj_loc[j,i,1]
            ax[2,1].plot(xpt, ypt, 'bo')
            ax[2,1].text(xpt*1.001, ypt*1.001, j, fontsize=10)
            ax[2,1].set_xlim([-2.5,2.5])
            ax[2,1].set_ylim([-2.5,2.5])
        ax[2,1].set_title("object location")

        for j in range(obj_loc_pred.shape[0]):
            xpt = obj_loc_pred[j,i,0]
            ypt = obj_loc_pred[j,i,1]
            ax[2,2].plot(xpt, ypt, 'bo')
            ax[2,2].text(xpt*1.001, ypt*1.001, j, fontsize=10)
            ax[2,2].set_xlim([-2.5,2.5])
            ax[2,2].set_ylim([-2.5,2.5])
        ax[2,2].set_title("object location predicted")

        t = str(count) + "-" + str(s[i])+"-"+str(i)
        plt.savefig(loc+t)
        plt.cla()
        plt.close(fig)

def visualize_locations(positions, s, loc, count=''):
    
    cam_loc = positions["cam_loc"]
    obj_loc = positions["obj_loc"]
    obj_pose = positions["obj_poses"]
    obj_loc_pred = positions["obj_loc_pred"]
    obj_pose_pred = positions["obj_pose_pred"]

    for i in range(cam_loc.shape[1]):
        
        fig, ax = plt.subplots(1,3, tight_layout=True, figsize=[10,3])
        fig.suptitle("Scene: "+str(s[i])+"-"+str(i))
        colors = ['b', 'g', 'r', 'c', 'y', 'm', 'k']

        for j in range(cam_loc.shape[0]):
            xpt = cam_loc[j,i,0]
            ypt = cam_loc[j,i,1]
            angle = np.sign(cam_loc[j,i,4])*np.arccos(cam_loc[j,i,3])
            ax[0].plot(xpt, ypt, 'o', c=colors[0], label="{0}-({1:.3f},{2:.3f}; {3:.3f})".format(j, xpt, ypt, angle))
            ax[0].text(xpt*1.001, ypt*1.001, j, fontsize=10)
        ax[0].set_xlim([-2.5,2.5])
        ax[0].set_ylim([-2.5,2.5])
        ax[0].set_aspect(1)
        leg = ax[0].legend(loc='upper right', frameon=False)
        ax[0].set_title("camera location")

        for j in range(obj_loc.shape[0]):
            for k in range(obj_loc.shape[2]):
                xpt = obj_loc[j,i,k,0]
                ypt = obj_loc[j,i,k,1]
                angle = np.sign(obj_pose[j,i,k,0])*np.arccos(obj_pose[j,i,k,1])
                ax[1].plot(xpt, ypt, 'o', c=colors[k], label="{0}-({1:.3f},{2:.3f}; {3:.3f})".format(j, xpt, ypt, angle))
                ax[1].text(xpt*1.001, ypt*1.001, j, fontsize=10)
        ax[1].set_xlim([-2.5,2.5])
        ax[1].set_ylim([-2.5,2.5])
        ax[1].set_aspect(1)
        leg = ax[1].legend(loc='upper right', frameon=False)
        ax[1].set_title("object location")

        for j in range(obj_loc_pred.shape[0]):
            for k in range(obj_loc_pred.shape[2]):
                xpt = obj_loc_pred[j,i,k,0]
                ypt = obj_loc_pred[j,i,k,1]
                angle = obj_pose_pred[j,i,k]
                ax[2].plot(xpt, ypt, 'o', c=colors[k], label="{0}-({1:.3f},{2:.3f}; {3:.3f})".format(j, xpt, ypt, angle))
                ax[2].text(xpt*1.001, ypt*1.001, j, fontsize=10)
        ax[2].set_xlim([-2.5,2.5])
        ax[2].set_ylim([-2.5,2.5])
        ax[2].set_aspect(1)
        leg = ax[2].legend(loc='upper right', frameon=False)
        ax[2].set_title("object location predicted")

        t = str(count) + "-" + str(s[i])+"-"+str(i)
        plt.savefig(loc+t)
        plt.cla()
        plt.close(fig)

        
def visualize_locations_traffic(positions, s, loc, count=''):
    
    cam_loc = positions["cam_loc"]
#     obj_loc = positions["obj_loc"]
#     obj_pose = positions["obj_poses"]
    obj_loc_pred = positions["obj_loc_pred"]
    obj_pose_pred = positions["obj_pose_pred"]

    for i in range(cam_loc.shape[1]):
        
        fig, ax = plt.subplots(1,3, tight_layout=True, figsize=[10,3])
        fig.suptitle("Scene: "+str(s[i])+"-"+str(i))
        colors = ['b', 'g', 'r', 'c', 'y', 'm', 'k']

        for j in range(cam_loc.shape[0]):
            xpt = cam_loc[j,i,0]
            ypt = cam_loc[j,i,1]
            angle = np.sign(cam_loc[j,i,4])*np.arccos(cam_loc[j,i,3])
            ax[0].plot(xpt, ypt, 'o', c=colors[0], label="{0}-({1:.3f},{2:.3f}; {3:.3f})".format(j, xpt, ypt, angle))
            ax[0].text(xpt*1.001, ypt*1.001, j, fontsize=10)
        ax[0].set_xlim([-2.5,2.5])
        ax[0].set_ylim([-2.5,2.5])
        ax[0].set_aspect(1)
        leg = ax[0].legend(loc='upper right', frameon=False)
        ax[0].set_title("camera location")

#         for j in range(obj_loc.shape[0]):
#             for k in range(obj_loc.shape[2]):
#                 xpt = obj_loc[j,i,k,0]
#                 ypt = obj_loc[j,i,k,1]
#                 angle = np.sign(obj_pose[j,i,k,0])*np.arccos(obj_pose[j,i,k,1])
#                 ax[1].plot(xpt, ypt, 'o', c=colors[k], label="{0}-({1:.3f},{2:.3f}; {3:.3f})".format(j, xpt, ypt, angle))
#                 ax[1].text(xpt*1.001, ypt*1.001, j, fontsize=10)
#         ax[1].set_xlim([-2.5,2.5])
#         ax[1].set_ylim([-2.5,2.5])
#         ax[1].set_aspect(1)
#         leg = ax[1].legend(loc='upper right', frameon=False)
#         ax[1].set_title("object location")

        for j in range(obj_loc_pred.shape[0]):
            for k in range(obj_loc_pred.shape[2]):
                xpt = obj_loc_pred[j,i,k,0]
                ypt = obj_loc_pred[j,i,k,1]
                angle = obj_pose_pred[j,i,k]
                ax[2].plot(xpt, ypt, 'o', c=colors[k], label="{0}-({1:.3f},{2:.3f}; {3:.3f})".format(j, xpt, ypt, angle))
                ax[2].text(xpt*1.001, ypt*1.001, j, fontsize=10)
#         ax[2].set_xlim([-2.5,2.5])
#         ax[2].set_ylim([-2.5,2.5])
        ax[2].set_aspect(1)
        leg = ax[2].legend(loc='upper right', frameon=False)
        ax[2].set_title("object location predicted")

        t = str(count) + "-" + str(s[i])+"-"+str(i)
        plt.savefig(loc+t)
        plt.cla()
        plt.close(fig)
        
def visualize_attention_maps(frame1, frame2, maps1, maps2, depth2, predicted3, s, loc, count = ''):
    for i in range(frame1.shape[0]):
        num_maps = maps1.shape[1]
        fig,ax = plt.subplots(2, num_maps+2)
        im = ax[0,0].imshow(frame1[i])
        ax[0,0].set_title("frame1")
        ax[1,0].imshow(frame2[i])
        ax[1,0].set_title("frame2")
        for j in range(num_maps):
            im = ax[0, 1+j].imshow(np.squeeze(maps1[i, j]))
            plt.colorbar(im, ax=ax[0, 1+j])
            ax[0, 1+j].set_title("frame_1 - map_{}".format(j))
            im = ax[1, 1+j].imshow(np.squeeze(maps2[i, j]))
            plt.colorbar(im, ax=ax[1, 1+j])
            ax[1, 1+j].set_title("frame_2 - map_{}".format(j))
        if not depth2 is None:
            im = ax[0, num_maps+1].imshow(np.squeeze(depth2[i]))
            plt.colorbar(im, ax=ax[0, num_maps+1])
            ax[0, num_maps+1].set_title("depth2")
            
            im = ax[1, num_maps+1].imshow(predicted3[i])
            ax[1, num_maps+1].set_title("predicted3")
            
        t = str(count) + "-" + str(s[i])+"-"+str(i)
        plt.savefig(loc+t)
        plt.cla()
        plt.close(fig)

def visualize_everything_multi(frames, maps, depths, weights, losses, gradients, positions,
        count, s, loc, intermediate=None, train_flag=True):
    b = frames[0].shape[0]
    irange = range(b)
    pool = mp.Pool(processes=16)
    pool_i = partial(visualize_everything, frames=frames, maps=maps, depths=depths, weights=weights, \
                    losses=losses, gradients=gradients, positions=positions, count=count, s=s, loc=loc, \
                     intermediate=intermediate, train=train_flag)
    pool.map(pool_i, irange)
    pool.close()
    pool.join()
    pool.terminate()
    gc.collect()

def visualize_everything(i, frames, maps, depths, weights, losses, gradients, positions,
        count, s, loc, intermediate=None, train=True):
    # print("batch element", i)
    frame1, frame2, frame3 = frames
    b = frame1.shape[0]
    maps2, maps3 = maps
    depth1, depth2, depth3 = depths
    # depthp1, depthp2, depthp3 = depthsp
    warpweight = weights["warpweight"]
    # imagine_weight = weights["imagineweight"]
    rgbloss = losses["rgbloss"]
    depthloss = losses["depthloss"]
    pixelshift = losses["pixelshift"]
    frame3p = intermediate["frame3p"]
    depth3p = intermediate["depth3p"]

    if(train):
        frame3pgrad = gradients["frame3pgrad"]
        frame3pwgrad = gradients["frame3pwgrad"]
        depth3pgrad = gradients["depth3pgrad"]
        weightgrad = gradients["weightinggrad"]
        # latentgrad = gradients["latent2grad"]
        attgrad = gradients['att_grad_0'] ## before activation
        attgrad_inseg = gradients['att_grad_inseg'] ## after activation and softmax
        attgrad_aftersfm = gradients['att_grad_aftersfm'] ## after activation and softmax inside network
    warpedframe2 = intermediate["warpedframe2"]
    imaginedframe3 = intermediate["imaginedframe3"]
    warpeddepth2 = intermediate["warpeddepth2"]
    imagineddepth3 = intermediate["imagineddepth3"]
    cam_loc = positions["cam_loc"]
    obj_loc = positions["obj_loc"]
    obj_loc_pred = positions["obj_loc_pred"]

    # for i in range(b):
    if True:
        fig, ax = plt.subplots(6, 6, tight_layout=True, figsize=(12,12))#, dpi=120)
        ax[0,0].imshow(frame1[i])
        ax[0,0].get_xaxis().set_ticks([])
        ax[0,0].get_yaxis().set_ticks([])
        ax[0,0].set_title("frame1")
        
        ax[0,1].imshow(frame2[i])
        ax[0,1].get_xaxis().set_ticks([])
        ax[0,1].get_yaxis().set_ticks([])
        ax[0,1].set_title("frame2")
        
        ax[0,2].imshow(frame3[i])
        ax[0,2].get_xaxis().set_ticks([])
        ax[0,2].get_yaxis().set_ticks([])
        ax[0,2].set_title("frame3")

        # im = ax[0,3].imshow(depth1[i], vmin=0, vmax=1.2) ## for gqn viz
        im = ax[0,3].imshow(depth1[i], vmin=0.5, vmax=6.5)
#         plt.colorbar(im, ax=ax[0,3])
        ax[0,3].set_title("depth1 GT")
        ax[0,3].get_xaxis().set_ticks([])
        ax[0,3].get_yaxis().set_ticks([])
        
        # im = ax[0,4].imshow(depth2[i], vmin=0, vmax=1.2) ## for gqn viz
        im = ax[0,4].imshow(depth2[i], vmin=0.5, vmax=6.5)
#         plt.colorbar(im, ax=ax[0,4])
        ax[0,4].set_title("depth2 GT")
        ax[0,4].get_xaxis().set_ticks([])
        ax[0,4].get_yaxis().set_ticks([])
        
        # im = ax[0,5].imshow(depth3[i], vmin=0, vmax=1.2) ## for gqn viz
        im = ax[0,5].imshow(depth3[i], vmin=0.5, vmax=6.5)
#         plt.colorbar(im, ax=ax[0,5])
        ax[0,5].set_title("depth3 GT")
        ax[0,5].get_xaxis().set_ticks([])
        ax[0,5].get_yaxis().set_ticks([])
        
        ax[1,0].imshow(frame3p[i])
        ax[1,0].set_title("frame3 predicted")
        ax[1,0].get_xaxis().set_ticks([])
        ax[1,0].get_yaxis().set_ticks([])
        
        ax[1,1].imshow(depth3p[i], vmin=0.5, vmax=6.5)
        ax[1,1].set_title("depth3 predicted")
        ax[1,1].get_xaxis().set_ticks([])
        ax[1,1].get_yaxis().set_ticks([])
        
        im = ax[1,2].imshow(warpweight[i], vmin=0, vmax=1)
        ax[1,2].set_title("warping weight")
        ax[1,2].get_xaxis().set_ticks([])
        ax[1,2].get_yaxis().set_ticks([])
        
        ax[1,3].imshow(warpedframe2[i])
        ax[1,3].set_title("warped frame2")
        ax[1,3].get_xaxis().set_ticks([])
        ax[1,3].get_yaxis().set_ticks([])
        
        im = ax[1,4].imshow(1.0-warpweight[i], vmin=0, vmax=1)
        ax[1,4].set_title("imagination weight")
        ax[1,4].get_xaxis().set_ticks([])
        ax[1,4].get_yaxis().set_ticks([])
        
        ax[1,5].imshow(imaginedframe3[i])
        ax[1,5].set_title("imagined frame3")
        ax[1,5].get_xaxis().set_ticks([])
        ax[1,5].get_yaxis().set_ticks([])
        
        im = ax[2,4].imshow(warpeddepth2[i], vmin=0.5, vmax=6.5)
        ax[2,4].set_title("depth3p warped")
        ax[2,4].get_xaxis().set_ticks([])
        ax[2,4].get_yaxis().set_ticks([])
        
        im = ax[2,5].imshow(imagineddepth3[i], vmin=0.5, vmax=6.5)
        ax[2,5].set_title("depth3p imagined")
        ax[2,5].get_xaxis().set_ticks([])
        ax[2,5].get_yaxis().set_ticks([])
        
        im = ax[2,0].imshow(rgbloss[i])
        plt.colorbar(im, ax=ax[2,0])
        ax[2,0].set_title("rgb loss values")
        ax[2,0].get_xaxis().set_ticks([])
        ax[2,0].get_yaxis().set_ticks([])
        
        im = ax[2,1].imshow(depthloss[i])
        plt.colorbar(im, ax=ax[2,1])
        ax[2,1].set_title("depth loss values")
        ax[2,1].get_xaxis().set_ticks([])
        ax[2,1].get_yaxis().set_ticks([])
        
        im = ax[2,2].imshow(pixelshift[i])
        plt.colorbar(im, ax=ax[2,2])
        ax[2,2].set_title("pixel shift values")
        ax[2,2].get_xaxis().set_ticks([])
        ax[2,2].get_yaxis().set_ticks([])
        
        
        for j in range(maps2.shape[1]):
            im = ax[3, j].imshow(maps2[i,j], vmin=0, vmax=1)
#             plt.colorbar(im, ax=ax[3,j])
            ax[3,j].set_title("attmap f2 m{}".format(j))
            ax[3,j].get_xaxis().set_ticks([])
            ax[3,j].get_yaxis().set_ticks([])
        
        for j in range(maps3.shape[1]):
            im = ax[4, j].imshow(maps3[i,j], vmin=0, vmax=1)
#             plt.colorbar(im, ax=ax[4,j])
            ax[4,j].set_title("attmap f3 m{}".format(j))
            ax[4,j].get_xaxis().set_ticks([])
            ax[4,j].get_yaxis().set_ticks([])

        if(train):
        ## gradients
            im = ax[5,0].imshow(frame3pgrad[i])
            plt.colorbar(im, ax=ax[5,0])
            ax[5,0].set_title("frame3p grad")
            ax[5,0].get_xaxis().set_ticks([])
            ax[5,0].get_yaxis().set_ticks([])
            
            im = ax[5,1].imshow(frame3pwgrad[i])
            plt.colorbar(im, ax=ax[5,1])
            ax[5,1].set_title("frame3pw grad")
            ax[5,1].get_xaxis().set_ticks([])
            ax[5,1].get_yaxis().set_ticks([])
            
            im = ax[5,2].imshow(weightgrad[i])
            plt.colorbar(im, ax=ax[5,2])
            ax[5,2].set_title("weight grad")
            ax[5,2].get_xaxis().set_ticks([])
            ax[5,2].get_yaxis().set_ticks([])
            
            im = ax[5,3].imshow(attgrad[i])
            plt.colorbar(im, ax=ax[5,3])
            ax[5,3].set_title("attmap grad")
            ax[5,3].get_xaxis().set_ticks([])
            ax[5,3].get_yaxis().set_ticks([])
            
            im = ax[5,4].imshow(attgrad_aftersfm[i])
            plt.colorbar(im, ax=ax[5,4])
            ax[5,4].set_title("attmap grad aftersfm")
            ax[5,4].get_xaxis().set_ticks([])
            ax[5,4].get_yaxis().set_ticks([])
            
            im = ax[5,5].imshow(attgrad_aftersfm[i])
            plt.colorbar(im, ax=ax[5,5])
            ax[5,5].set_title("attmap grad afterseg")
            ax[5,5].get_xaxis().set_ticks([])
            ax[5,5].get_yaxis().set_ticks([])
            
        t = str(count) + "-" + str(s[i])+"-"+str(i)
        plt.savefig(loc+t)
        plt.cla()
        plt.close(fig)

#### external code: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
def plot_grad_flow(path, named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        # print("name of param", n)
        if(p.requires_grad) and ("bias" not in n):
            # print("has grad")
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            if("main_net.lin2.0" in n):
                layers.append("latent ids")
                ave_grads.append(p.grad[:, 39:103].abs().mean())
                max_grads.append(p.grad[:, 39:103].abs().max())
                layers.append("latent pose")
                ave_grads.append(p.grad[:, 3:39].abs().mean())
                max_grads.append(p.grad[:, 3:39].abs().max())
                layers.append("latent location")
                ave_grads.append(p.grad[:, 0:3].abs().mean())
                max_grads.append(p.grad[:, 0:3].abs().max())

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(path, bbox_inches="tight")

def plot_img(img, location):
    ## img size: hxwx1 or hxwx3
    fig = plt.figure(figsize=[6,6], dpi=160, frameon=False)
    plt.imshow(img, interpolation='nearest')
    plt.axis('off')
    plt.savefig(location)
    plt.cla()
    plt.close()

def plot_img_with_heatmap(img, location):
    ## img size: hxwx1 
    fig = plt.figure(figsize=[6,6], dpi=160, frameon=False)
    ax = fig.add_subplot(111)
    ax.imshow(img, interpolation='nearest', aspect='equal', cmap=plt.get_cmap('gray'))
    # plt.colorbar()
    ax.set_axis_off()
    plt.savefig(location)
    plt.cla()
    plt.close()

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def plot_depth_with_heatmap(img, location):
    ## img size: hxwx1 
    fig = plt.figure(figsize=[6,6], dpi=160, frameon=False)
    ax = fig.add_subplot(111)
    im = ax.imshow(np.log(img), interpolation='nearest', aspect='equal', cmap=plt.get_cmap('gray'))
    # plt.colorbar(im, ax)
    ax.set_axis_off()
    # forceAspect(ax, 1)
    plt.savefig(location)
    plt.cla()
    plt.close()

def plot_context_matrix(mat, loc):
    fig,ax = plt.subplots()
    plt.matshow(mat, cmap=plt.cm.Blues)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            c = mat[i,j]
            plt.text(i,j,"{:.3f}".format(c), size=8, va='center', ha='center')
    plt.axis('off')
    plt.savefig(loc)
    plt.cla()
    plt.close(fig)

def visualize_figures_new(outputs, cfg, params):
    X_upd = outputs['X_updated']
    seg = outputs['segment_outputs']
    depth = outputs['depth_outputs']
    warp = outputs['warping_outputs']
    imag = outputs['imag_outputs']
    predicted = outputs['predicted_outputs']

    frames = X_upd['frames']
    depths = depth['depth_smoothed']
    depths_gt = X_upd['depths_gt']
    seg_out1 = seg['seg_out1']['attention_maps_softmax']
    seg_out2 = seg['seg_out2']['attention_maps_softmax']
    seg_out3 = seg['seg_out3']['attention_maps_softmax']
    maps = np.stack([seg_out1, seg_out2, seg_out3], 1)
    colored_inferred_masks = label_maps_to_colour(seg_out3)

    output_loc =  "/home/tushar/temp/" + params['experiment_name']+\
                    "_"+ params['run_time']+"/test_viz/"

    if not os.path.exists(output_loc):
        os.makedirs(output_loc)

    for b in range(frames.shape[1]):
        im = colored_inferred_masks[b] * 255
        im = Image.fromarray(im.astype(np.uint8))
        im.save(output_loc + "POP_{}.png".format(b))

        im = frames[2, b].transpose(1,2,0) * 255
        im = Image.fromarray(im.astype(np.uint8))
        im.save(output_loc + "input_{}.png".format(b))


def visualize_figures_traffic(frames, maps, depths, weights, losses, gradients,\
        positions, count, idx, loc, intermediate):

    for i in range(idx.shape[0]):
        foldername = loc + "batch_{}_{}_{}/".format(count, i, idx[i])
        if(not os.path.exists(foldername)):
            os.makedirs(foldername)

        frame1, frame2, frame3 = frames
        ##visualize image1
        filename = foldername + "image1.png"
        plot_img(frame1[i], filename)
        ##visualize image2
        filename = foldername + "image2.png"
        plot_img(frame2[i], filename)
        ##visualize image3
        filename = foldername + "image3.png"
        plot_img(frame3[i], filename)

        depth1, depth2, depth3 = depths
        ##visualize depth1
        filename = foldername + "depth1.png"
        plot_depth_with_heatmap(depth1[i], filename)
        ##visualize depth2
        filename = foldername + "depth2.png"
        plot_depth_with_heatmap(depth2[i], filename)
        ##visualize depth3
        filename = foldername + "depth3.png"
        plot_depth_with_heatmap(depth3[i], filename)

        depthgt1, depthgt2, depthgt3 = intermediate['depth_gt']
        ##visualize depth1
        filename = foldername + "depth1gt.png"
        plot_depth_with_heatmap(depthgt1[i], filename)
        ##visualize depth2
        filename = foldername + "depth2gt.png"
        plot_depth_with_heatmap(depthgt2[i], filename)
        ##visualize depth3
        filename = foldername + "depth3gt.png"
        plot_depth_with_heatmap(depthgt3[i], filename)

        maps2, maps3 = maps
#         attgt1, attgt2, attgt3 = intermediate['attgt']
        warpweight = weights["warpweight"]
        warpedframe2 = intermediate["warpedframe2"]
        imaginedframe3 = intermediate["imaginedframe3"]

        ##visualize warped image
        filename = foldername + "warpedframe2.png"
        plot_img(warpedframe2[i], filename)
        ##visualize imagined image
        filename = foldername + "imaginedframe3.png"
        plot_img(imaginedframe3[i], filename)

        frame3p = intermediate["frame3p"]
        ##visualize final predicted image
        filename = foldername + "predictedframe3.png"
        plot_img(frame3p[i], filename)

        ##visualize warping weight
        filename = foldername + "warpingweight.png"
        plot_img_with_heatmap(warpweight[i], filename)

        ##visualize imagination weight
        filename = foldername + "imaginationweight.png"
        plot_img_with_heatmap((1.0-warpweight[i]), filename)

        ##visualize attention map parts of the image
        filename = foldername + "attentionframe2_slot{}.png"
        for j in range(maps2.shape[1]):
            plot_img(frame2[i]*maps2[i,j,:,:,None] + (1 - maps2[i,j,:,:,None]), filename.format(j))
        
        filename = foldername + "attentionframe3_slot{}.png"
        for j in range(maps3.shape[1]):
            plot_img(frame3[i]*maps3[i,j,:,:,None] + (1 - maps3[i,j,:,:,None]), filename.format(j))
            
        ##visualize gt attention map parts of the image
#         filename = foldername + "attentionframe2gt_slot{}.png"
#         for j in range(attgt2.shape[1]):
#             plot_img(frame2[i]*attgt2[i,j,:,:,None] + (1 - attgt2[i,j,:,:,None]), filename.format(j))
        
#         filename = foldername + "attentionframe3gt_slot{}.png"
#         for j in range(attgt3.shape[1]):
#             plot_img(frame3[i]*attgt3[i,j,:,:,None] + (1 - attgt3[i,j,:,:,None]), filename.format(j))
