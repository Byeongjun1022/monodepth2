#!/bin/bash

#for ((var=14 ; var < 20 ; var++));
#do
#  python evaluate_depth.py --load_weights_folder  ~/tmp/mono_model_panoptic/models/weights_$var --eval_mono --panoptic_decoder
#done

for ((var=15 ; var < 21 ; var++));
do
  python evaluate_depth.py --load_weights_folder  ~/tmp/lite_mab_no_residual_4_blockngrid8x8/models/weights_$var --eval_mono --lite  --block_size 4 4 --grid_size 4 4
done
