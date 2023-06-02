#!/bin/bash

#for ((var=14 ; var < 20 ; var++));
#do
#  python evaluate_depth.py --load_weights_folder  ~/tmp/mono_model_panoptic/models/weights_$var --eval_mono --panoptic_decoder
#done

for ((var=15; var < 20 ; var++));
do
#  python evaluate_depth.py --load_weights_folder  ~/tmp/lite_lgfi_se_nores_2023-04-05\ 19:56:19/models/weights_$var --eval_mono --lite
#  python evaluate_depth.py --load_weights_folder  ~/tmp/lite_parallel_2023-04-10\ 23:29:50/models/weights_$var --eval_mono --lite --parallel
#  python evaluate_depth.py --load_weights_folder  ~/tmp/lite_lgfi_se_2023-04-04\ 23:29:58/models/weights_28 --eval_mono --lite --global_block_type LGFI_SE --res --save_pred_disps
 python evaluate_depth.py --load_weights_folder  ~/tmp/lite_4/models/weights_$var --eval_mono --lite --global_block_type LGFI --save_pred_disps
#  python evaluate_depth.py --load_weights_folder  ~/tmp/lite_mab_no_residual_3/models/weights_$var --eval_mono --lite --global_block_type MAB
  # python evaluate_depth.py --load_weights_folder  ~/tmp/lite_mab_no_residual_2023-05-31\ 01:34:53/models/weights_$var --eval_mono --lite --global_block_type MAB
#  python evaluate_depth.py --load_weights_folder  ~/tmp/lite_mab_no_residual_2023-05-29\ 00:23:23/models/weights_$var --eval_mono --lite --global_block_type MAB
done
