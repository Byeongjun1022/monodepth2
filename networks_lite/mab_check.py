
import torch
from torch import nn
from maxim_pytorch import ResidualSplitHeadMultiAxisGmlpLayer
class MAB(nn.Module):
    def __init__(self, num_channels, residual, block_size=(4, 4), grid_size=(4, 4), **kwargs):
        super().__init__()
        self.block_size = block_size
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.residual = residual

        self.mab = ResidualSplitHeadMultiAxisGmlpLayer(self.block_size, self.grid_size, self.num_channels)
        if kwargs is not None:
            if kwargs['SE'] == True:
                self.se = True


    def forward(self, x):
        input_ = x

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.mab(x)
        x = x.permute(0, 3, 1, 2)

        if self.residual:
            x = input_ + x

        return x
    
if __name__ == '__main__':
    mab = MAB(64, True, SE = True)
    print('hi')
