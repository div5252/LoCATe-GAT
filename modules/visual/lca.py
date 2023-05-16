import torch
import torch.nn as nn
from clip.model import QuickGELU
from modules.visual.cbam import CBAM

class LCA(nn.Module):
    def __init__(self, LCA_drops, embed_dim=512):
        super(LCA, self).__init__()
        self.embed_dim = embed_dim

        if embed_dim == 512:        
            self.branch1 = nn.Sequential(
                nn.Conv2d(2, 8, kernel_size=(1,1)),
                QuickGELU(),
                nn.Conv2d(8, 64, kernel_size=(3,3), dilation=1),
                QuickGELU(),
                CBAM(64),
                nn.Conv2d(64, 256, kernel_size=(1,1)),
                QuickGELU(),
                nn.Dropout(LCA_drops[0]),
                nn.AdaptiveAvgPool2d((1, 1))
            )

            self.branch2 = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=(1,1)),
                QuickGELU(),
                nn.Conv2d(16, 128, kernel_size=(3,3), dilation=2),
                QuickGELU(),
                CBAM(128),
                nn.Conv2d(128, 192, kernel_size=(1,1)),
                QuickGELU(),
                nn.Dropout(LCA_drops[1]),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

            self.branch3 = nn.Sequential(
                nn.Conv2d(2, 4, kernel_size=(1,1)),
                QuickGELU(),
                nn.Conv2d(4, 16, kernel_size=(3,3), dilation=4),
                QuickGELU(),
                CBAM(16),
                nn.Conv2d(16, 64, kernel_size=(1,1)),
                QuickGELU(),
                nn.Dropout(LCA_drops[2]),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        elif embed_dim == 768:
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=(1,1)),
                QuickGELU(),
                nn.Conv2d(8, 64, kernel_size=(3,3), dilation=1),
                QuickGELU(),
                CBAM(64),
                nn.Conv2d(64, 512, kernel_size=(1,1)),
                QuickGELU(),
                nn.Dropout(LCA_drops[0]),
                nn.AdaptiveAvgPool2d((1, 1))
            )

            self.branch2 = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=(1,1)),
                QuickGELU(),
                nn.Conv2d(8, 32, kernel_size=(3,3), dilation=2),
                QuickGELU(),
                CBAM(32),
                nn.Conv2d(32, 128, kernel_size=(1,1)),
                QuickGELU(),
                nn.Dropout(LCA_drops[1]),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

            self.branch3 = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=(1,1)),
                QuickGELU(),
                nn.Conv2d(8, 32, kernel_size=(3,3), dilation=4),
                QuickGELU(),
                CBAM(32),
                nn.Conv2d(32, 128, kernel_size=(1,1)),
                QuickGELU(),
                nn.Dropout(LCA_drops[2]),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        elif embed_dim == 1024:
            self.branch1 = nn.Sequential(
                nn.Conv2d(4, 8, kernel_size=(1,1)),
                QuickGELU(),
                nn.Conv2d(8, 64, kernel_size=(3,3), dilation=1),
                QuickGELU(),
                CBAM(64),
                nn.Conv2d(64, 512, kernel_size=(1,1)),
                QuickGELU(),
                nn.Dropout(LCA_drops[0]),
                nn.AdaptiveAvgPool2d((1, 1))
            )

            self.branch2 = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=(1,1)),
                QuickGELU(),
                nn.Conv2d(16, 128, kernel_size=(3,3), dilation=2),
                QuickGELU(),
                CBAM(128),
                nn.Conv2d(128, 384, kernel_size=(1,1)),
                QuickGELU(),
                nn.Dropout(LCA_drops[1]),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

            self.branch3 = nn.Sequential(
                nn.Conv2d(4, 8, kernel_size=(1,1)),
                QuickGELU(),
                nn.Conv2d(8, 16, kernel_size=(3,3), dilation=4),
                QuickGELU(),
                CBAM(16),
                nn.Conv2d(16, 128, kernel_size=(1,1)),
                QuickGELU(),
                nn.Dropout(LCA_drops[2]),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        else:
            exit("No LCA for embedding dimension = " + str(embed_dim))


    def forward(self, x):
        bs, t, _ = x.shape
        if self.embed_dim == 512:
            x = x.reshape(bs, t, 2, 16, 16)
            x = x.reshape(bs*t, 2, 16, 16)
        elif self.embed_dim == 768:
            x = x.reshape(bs, t, 3, 16, 16)
            x = x.reshape(bs*t, 3, 16, 16)
        elif self.embed_dim == 1024:
            x = x.reshape(bs, t, 4, 16, 16)
            x = x.reshape(bs*t, 4, 16, 16)

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x1 = torch.squeeze(x1)
        x2 = torch.squeeze(x2)
        x3 = torch.squeeze(x3)
        out = torch.cat((x1, x2, x3), dim=1)
        out = out.reshape(bs, t, self.embed_dim)
        return out

# 2 branches
class LCA_branch_2(nn.Module):
    def __init__(self, LCA_drops, embed_dim=512):
        super(LCA_branch_2, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=(1,1)),
            QuickGELU(),
            nn.Conv2d(8, 64, kernel_size=(3,3), dilation=2),
            QuickGELU(),
            CBAM(64),
            nn.Conv2d(64, 256, kernel_size=(1,1)),
            QuickGELU(),
            nn.Dropout(LCA_drops[1]),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=(1,1)),
            QuickGELU(),
            nn.Conv2d(8, 64, kernel_size=(3,3), dilation=4),
            QuickGELU(),
            CBAM(64),
            nn.Conv2d(64, 256, kernel_size=(1,1)),
            QuickGELU(),
            nn.Dropout(LCA_drops[2]),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        bs, t, _ = x.shape
        x = x.reshape(bs, t, 2, 16, 16)
        x = x.reshape(bs*t, 2, 16, 16)

        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x1 = torch.squeeze(x1)
        x2 = torch.squeeze(x2)
        out = torch.cat((x1, x2), dim=1)
        out = out.reshape(bs, t, 512)
        return out

# 1 branch   
class LCA_branch_1(nn.Module):
    def __init__(self, LCA_drops, embed_dim=512):
        super(LCA_branch_1, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=(1,1)),
            QuickGELU(),
            nn.Conv2d(8, 64, kernel_size=(3,3), dilation=4),
            QuickGELU(),
            CBAM(64),
            nn.Conv2d(64, 512, kernel_size=(1,1)),
            QuickGELU(),
            nn.Dropout(LCA_drops[2]),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        bs, t, _ = x.shape
        x = x.reshape(bs, t, 2, 16, 16)
        x = x.reshape(bs*t, 2, 16, 16)

        x1 = self.branch1(x)
        x1 = torch.squeeze(x1)

        x1 = x1.reshape(bs, t, 512)
        return x1