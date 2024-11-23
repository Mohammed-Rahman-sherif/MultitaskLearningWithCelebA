from torch import nn
from collections import OrderedDict

class MTArchitecture(nn.Module):
    def __init__(self):
        super(MTArchitecture, self).__init__()
        self.shared_layer = nn.Sequential(OrderedDict([
            ("conv_1", nn.Conv2d(in_channels=3, out_channels=6,
                                 kernel_size=3, stride=1, padding=1)),
            ("bn1", nn.BatchNorm2d(6)),
            ("re_1", nn.ReLU()),
            ("pl_1", nn.MaxPool2d(kernel_size=2, stride=2)),

            ("conv_2", nn.Conv2d(in_channels=6, out_channels=16,
                                 kernel_size=3, stride=1, padding=1)),
            ("bn_2", nn.BatchNorm2d(16)),
            ("re_2", nn.ReLU()),
            ("pl_2", nn.MaxPool2d(kernel_size=2, stride=2)),

            ("conv_3", nn.Conv2d(in_channels=16, out_channels=32,
                                 kernel_size=3, stride=1, padding=1)),
            ("bn_3", nn.BatchNorm2d(32)),
            ("re_3", nn.ReLU()),
            ("pl_3", nn.AdaptiveAvgPool2d((8, 8))),
        ]))

        self.heads = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ("fc_1", nn.Linear(in_features=32*8*8, out_features=32)),
                ("bn_fc_1", nn.BatchNorm1d(32)),
                ("re_fc_1", nn.ReLU()),
                ("dr_fc_1", nn.Dropout(0.5)),
                ("fc_out", nn.Linear(in_features=32, out_features=1)),
                ("sig_out", nn.Sigmoid())
            ])) for _ in range(40)  # 40 heads for 40 attributes
        ])

    def forward(self, X):
        X = self.shared_layer(X)
        X = X.view(X.size(0), -1)
        return [head(X) for head in self.heads]