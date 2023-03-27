# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions
import count_number

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    if opts.count:
        enc_parameter_num = count_number.count_parameters(trainer.models["encoder"])
        dec_parameter_num = count_number.count_parameters(trainer.models["depth"])
        print(f'encoder parameters:{enc_parameter_num}')
        print(f'decoder parameters:{dec_parameter_num}')
    else:
        trainer.train()
