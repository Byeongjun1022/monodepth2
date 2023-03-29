#!/bin/bash

#for ((var=14 ; var < 20 ; var++));
#do
#  python evaluate_depth.py --load_weights_folder  ~/tmp/mono_model_panoptic/models/weights_$var --eval_mono --panoptic_decoder
#done

for ((var=5 ; var < 6 ; var++));
do
  python evaluate_depth.py --load_weights_folder  ~/tmp/lite_mab_maxim/models/weights_$var --eval_mono --lite --maxim
done
