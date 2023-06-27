from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
import json
import panoptic_decoder
import networks_lite
import time
from thop import clever_format
from thop import profile
from bj import *
from torchvision.utils import save_image

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

def profile_once(encoder, decoder, x):
    x_e = x[0, :, :, :].unsqueeze(0)
    x_d = encoder(x_e)
    flops_e, params_e = profile(encoder, inputs=(x_e, ), verbose=False)
    flops_d, params_d = profile(decoder, inputs=(x_d, ), verbose=False)

    flops, params = clever_format([flops_e + flops_d, params_e + params_d], "%.3f")
    flops_e, params_e = clever_format([flops_e, params_e], "%.3f")
    flops_d, params_d = clever_format([flops_d, params_d], "%.3f")

    return flops, params, flops_e, params_e, flops_d, params_d


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def get_reprojected_image(opt):
    """Evaluates a pretrained model using a specified test set
    """
    if opt.ddad:
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 200
    else:
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    if opt.debug:
        filenames = filenames[:16]
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path, map_location=f'cuda:{opt.gpu_num}')

    if opt.ddad:
        dataset=datasets.DGPDataset(data_path=opt.json_path, split='val',
                                    height=opt.height, width=opt.width,
                                    frame_idxs=opt.frame_ids, num_scales=4, datum_names=['lidar','CAMERA_01'],
                                    back_context=1, forward_context=1)
    else:
        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                            encoder_dict['height'], encoder_dict['width'],
                                            opt.frame_ids, 4, is_train=False)
    dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)
    
    device = torch.device("cpu" if opt.no_cuda else f"cuda:{opt.gpu_num}")
    if opt.lite:
        if opt.parallel:
            encoder=networks_lite.LiteMono_parallel(maxim=opt.maxim,
                                                    block_size=tuple(opt.block_size),
                                                    grid_size=tuple(opt.grid_size),
                                                    residual=opt.res,
                                                    global_block_type=[opt.global_block_type for i in range(3)])
        else:
            encoder = networks_lite.LiteMono(opt,
                                                maxim=opt.maxim,
                                                block_size=tuple(opt.block_size),
                                                grid_size=tuple(opt.grid_size),
                                                residual=opt.res,
                                                global_block_type=[opt.global_block_type for i in range(3)],
                                                SE=opt.mab_se)
    else:
        encoder = networks.ResnetEncoder(opt.num_layers, False)

    if opt.panoptic_decoder:
        with open(opt.panoptic_option_pth) as f:
            panoptic_opt=json.load(f)

        depth_decoder=panoptic_decoder.SinglePanopticDeepLabDecoder_bj(
            **panoptic_opt
        )
    elif opt.lite:
        depth_decoder=networks_lite.DepthDecoder(encoder.num_ch_enc, [0,1,2])

    else:
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    model_dict_dec = depth_decoder.state_dict()
    decoder_dict = torch.load(decoder_path, map_location=f'cuda:{opt.gpu_num}')
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in model_dict_dec})

    encoder.to(device)
    encoder.eval()
    depth_decoder.to(device)
    depth_decoder.eval()

    opt.use_pose_net = True
    if opt.use_pose_net:
        if opt. lite:
            pose_encoder = networks_lite.ResnetEncoder(opt.num_layers, False, 2)
            pose_decoder = networks_lite.PoseDecoder(pose_encoder.num_ch_enc, 
                                                        num_input_features=1, 
                                                        num_frames_to_predict_for=2)
        else:
            pose_encoder = networks.ResnetEncoder(opt.num_layers, False, 2)
            pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 
                                                num_input_features=1, 
                                                num_frames_to_predict_for=2)
    # pose_encoder_dict = pose_encoder.state_dict()
    # pose_decoder_dict = pose_decoder.state_dict()

    pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")

    pose_encoder_dict = torch.load(pose_encoder_path, map_location=f'cuda:{opt.gpu_num}')
    pose_decoder_dict = torch.load(pose_decoder_path, map_location=f'cuda:{opt.gpu_num}')

    pose_encoder.load_state_dict(pose_encoder_dict)
    pose_decoder.load_state_dict(pose_decoder_dict)

    pose_encoder.to(device)
    pose_encoder.eval()
    pose_decoder.to(device)
    pose_decoder.eval()

    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # input_color = data[("color", 0, 0)].to(device)
            for key, ipt in data.items():
                data[key] = ipt.to(device)
            input_color = data[("color", 0, 0)]

            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
            
            t1 = time_sync()
            if opt.lite:
                output,_ = depth_decoder(encoder(input_color))
                output.update(predict_poses(opt, pose_encoder, pose_decoder, data))
            else:
                output = depth_decoder(encoder(input_color))
                output.update(predict_poses(opt, pose_encoder, pose_decoder, data))
            t2 = time_sync()

            generate_images_pred(opt, data, output)

            recon_images = []
            recon_images.append(data[("color", 0, 0)][0])
            for frame_id in opt.frame_ids[1:]:           
                recon_images.append(output[("color", frame_id, 0)][0])
            image_cat = torch.cat(recon_images, 1)
            save_image(image_cat,  f'./visualizations/img_{i}.png')
            

    


if __name__ == "__main__":
    options = MonodepthOptions()
    get_reprojected_image(options.parse())
