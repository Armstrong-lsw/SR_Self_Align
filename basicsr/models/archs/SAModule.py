from torch import nn


class SALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SALayer, self).__init__()
        self.sam = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction, channel, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.sam(x)
        return x * y