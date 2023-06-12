#!/bin/bash

#for ((var=14 ; var < 20 ; var++));
#do
#  python evaluate_depth.py --load_weights_folder  ~/tmp/mono_model_panoptic/models/weights_$var --eval_mono --panoptic_decoder
#done

for ((var=16 ; var < 30 ; var++));
do
  # CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder  ~/tmp/lite_rm_dilation_8_2023-04-28\ 15:25:08/models/weights_$var --eval_mono --lite  --global_block_type MAB --gpu_num 0
#  CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /mnt_2/tmp/lite_mab_no_residual_2023-05-27\ 23:03:39/models/weights_$var --eval_mono --lite  --global_block_type MAB --gpu_num 0
#  CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /mnt_2/tmp/lite_mab_no_residual_2x_2023-05-29\ 16:57:40/models/weights_$var --eval_mono --lite  --global_block_type MAB --gpu_num 0 --save_pred_disps
# CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /mnt_2/tmp/lite_maxim_2x_2023-05-31\ 01:55:07/models/weights_$var --eval_mono --lite --maxim --global_block_type MAB --gpu_num 0 --save_pred_disps
# CUDA_VISIBLE_DEVICES=1 python evaluate_depth.py --load_weights_folder /mnt_2/tmp/lite_maxim_wdconv_2023-06-02\ 20:24:45/models/weights_$var --eval_mono --lite --global_block_type MAXIM --gpu_num 0 --save_pred_disps
# CUDA_VISIBLE_DEVICES=1 python evaluate_depth.py --load_weights_folder /mnt_2/tmp/lite_max_se_2023-06-05\ 14:27:26/models/weights_$var --eval_mono --lite --global_block_type Max --gpu_num 0 --mab_se #--save_pred_disps
#   CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /mnt_2/tmp/lite_mab_adamw_2023-06-07\ 10:42:31/models/weights_$var --eval_mono --lite --global_block_type MAB --gpu_num 0 --mab_se --save_pred_disps
  # CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /mnt_2/tmp/lite_4/models/weights_$var --eval_mono --lite --global_block_type LGFI --gpu_num 0 #--save_pred_disps
#   CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /mnt_2/tmp/lite_max_wnhead_w_2023-06-0813:15:18/models/weights_$var --eval_mono --lite --global_block_type Max --gpu_num 0 --save_pred_disps
   CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --load_weights_folder /mnt_2/tmp/lite_edge_2023-06-1123:07:09/models/weights_$var --eval_mono --lite --global_block_type Edge --gpu_num 0 #--save_pred_disps
done

