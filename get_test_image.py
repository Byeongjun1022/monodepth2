from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils import readlines
from options import MonodepthOptions
import datasets
import PIL.Image as pil
from torchvision.utils import save_image

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")




def get_test_image(opt):
    output_pth = '/mnt_2/Results/Color'

    if not os.path.exists(output_pth):
        print('Since output path does not exist, make output folder')
        os.makedirs(output_pth)

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                       192, 640,
                                       [0], 4, is_train=False)
    
    for i in range(len(filenames)):

        input_color=dataset[i][("color", 0, 0)]
        save_image(input_color, os.path.join(output_pth,f'color_{i}.jpg'))
    



if __name__ == "__main__":
    options = MonodepthOptions()
    get_test_image(options.parse())
