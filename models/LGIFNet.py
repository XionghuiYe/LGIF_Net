import torch
from einops import repeat
from torch import nn
from torchsummary import summary

from models.LGI_Former import LGBlock


class LGIFNet(nn.Module):
    def __init__(self, depth=12, drop_path_rate=0.):
        super().__init__()
        # channel 1
        self.channel_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=3, kernel_size=(5, 5), padding="same"
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
        )
        # channel 2
        self.channel_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=5, kernel_size=(5, 5), padding="same"
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
        )
        # channel 3
        self.channel_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=8, kernel_size=(5, 5), padding="same"
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3)),
        )
        # interpretation
        self.interpretation = nn.Sequential(
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.LazyLinear(out_features=1666),
            nn.ReLU(),
            nn.LazyLinear(out_features=400),
            nn.ReLU(),
            nn.LazyLinear(out_features=1),
        )
        self.lg_region_tokens = nn.Parameter(torch.zeros(1, 196))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            LGBlock(dim=196, num_heads=4, drop_path=dpr[i])
            for i in range(depth)])

    def forward(self, inputs):
        inputs_1 = inputs[:, 0, :, :]
        inputs_1 = inputs_1.unsqueeze(1)
        inputs_2 = inputs[:, 1, :, :]
        inputs_2 = inputs_2.unsqueeze(1)
        inputs_3 = inputs[:, 2, :, :]
        inputs_3 = inputs_3.unsqueeze(1)
        # channel 1
        channel_1 = self.channel_1(inputs_1)
        # channel 2
        channel_2 = self.channel_2(inputs_2)
        # channel 3
        channel_3 = self.channel_3(inputs_3)
        # merge
        merged = torch.cat((channel_1, channel_2, channel_3), 1)
        x = torch.flatten(merged,2)
        b = x.size(0)
        region_tokens = repeat(self.lg_region_tokens, 'n c -> b n c', b=b)
        x = torch.cat([region_tokens, x], dim=1)  # (b, nt*nh*nw, 1+thw, c)
        for blk in self.blocks:
            x = blk(x)
        # x = x[:, 1:]
        # x = x[:, 0]
        # interpretation
        outputs = self.interpretation(x)
        return outputs


if __name__ == '__main__':

    model = LGIFNet()
    model.to('cuda')
    summary(model,(3,42,42))