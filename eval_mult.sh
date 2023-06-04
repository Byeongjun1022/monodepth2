#!/bin/bash

#for ((var=14 ; var < 20 ; var++));
#do
#  python evaluate_depth.py --load_weights_folder  ~/tmp/mono_model_panoptic/models/weights_$var --eval_mono --panoptic_decoder
#done

for ((var=21 ; var < 22 ; var++));
do
  # CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder  ~/tmp/lite_rm_dilation_8_2023-04-28\ 15:25:08/models/weights_$var --eval_mono --lite  --global_block_type MAB --gpu_num 0
#  CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /mnt_2/tmp/lite_mab_no_residual_2023-05-27\ 23:03:39/models/weights_$var --eval_mono --lite  --global_block_type MAB --gpu_num 0
#  CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /mnt_2/tmp/lite_mab_no_residual_2x_2023-05-29\ 16:57:40/models/weights_$var --eval_mono --lite  --global_block_type MAB --gpu_num 0
# CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /mnt_2/tmp/lite_maxim_2x_2023-05-31\ 01:55:07/models/weights_$var --eval_mono --lite --maxim --global_block_type MAB --gpu_num 0 --save_pred_disps
# CUDA_VISIBLE_DEVICES=1 python evaluate_depth.py --load_weights_folder /mnt_2/tmp/lite_maxim_wdconv_2023-06-02\ 20:24:45/models/weights_$var --eval_mono --lite --global_block_type MAXIM --gpu_num 0 --save_pred_disps
   CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /mnt_2/tmp/lite_mab_se_2x_2023-06-01\ 16:20:38/models/weights_$var --eval_mono --lite --global_block_type MAB --gpu_num 0 --mab_se #--save_pred_disps
  # CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /mnt_2/tmp/lite_4/models/weights_$var --eval_mono --lite --global_block_type LGFI --gpu_num 0 #--save_pred_disps
done