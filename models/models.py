import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F


class SWN_GCN_layer(nn.Module):
    '''
    Implementation of WN-GCN
    '''
    def __init__(self, args, pointconv_cfg, reverse=False):
        super(SWN_GCN_layer, self).__init__()
        self.reverse = reverse
        i = 0
        if reverse:
            i = -1
        self.aggregate = self.weighted_aggregation(pointconv_cfg[i], pointconv_cfg[i], 3)
        self.vertexfeat = self.make_layer(args, pointconv_cfg)


    def forward(self, x):
        if self.reverse:
            x = self.vertexfeat(x)
            x = self.aggregate(x)
            return x
        x = self.aggregate(x)
        x = self.vertexfeat(x)

        return x

    def weighted_aggregation(self, in_channel, out_channel, kernel_size):


        alpha = torch.nn.Parameter(torch.tensor(1/9), requires_grad=False).to('cuda')
        beta = torch.nn.Parameter(torch.tensor(1/9), requires_grad=True).to('cuda')

        ## Fixing alpha is done in train.py by blocking the gradients over them
        w = torch.tensor([[alpha.clone(),  alpha.clone(), alpha.clone()],
                          [alpha.clone(),  beta, alpha.clone()],
                          [alpha.clone(),  alpha.clone(), alpha.clone()]])
        w = w.view(1, 1, kernel_size, kernel_size).repeat(out_channel, 1, 1, 1)
        w = w.to('cuda')
        layer = torch.nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=1, bias=False, groups=out_channel)
        layer.weight = torch.nn.Parameter(w, requires_grad=True)
        return layer


    def make_layer(self, args, pointconv_cfg):
        layer = []
        for i, v in enumerate(pointconv_cfg):
            if i == 0:
                in_channels = v
            else:
                try:
                    layer += [torch.nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=1, bias=True), torch.nn.BatchNorm2d(v), torch.nn.ReLU()]
                except (TypeError):
                    print("debug! come to GraphNetBlock ")
                    pdb.set_trace()

                in_channels = v
        return torch.nn.Sequential(*layer)



class CosFaceClassifier(nn.Module):
    def __init__(self, cfgs, normalize=True):
        super(CosFaceClassifier, self).__init__()
        self.fc1 = nn.Linear(cfgs[0], cfgs[1])
        self.relu = nn.ReLU()
        self.normalize=normalize
        if normalize:
            self.fc2 = nn.utils.weight_norm(nn.Linear(cfgs[1], cfgs[2]))
        else:
            self.fc2 = nn.Linear(cfgs[1], cfgs[2])

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        out = self.fc2(x)
        return out


class SWN_GCN(nn.Module):
    '''
    Weighted nearest-Neighbor Graph Convolutional Network Module
    '''

    def __init__(self, args, cfgs, cfgs_cls, cosface=True, reverse=False):

        super(SWN_GCN, self).__init__()
        self.encoder = self.make_layer(args, cfgs, reverse)
        self.GAP = torch.nn.AdaptiveAvgPool2d((1, 1))
        if cosface:
            self.classifier = CosFaceClassifier(cfgs_cls)
        else:
            self.classifier = nn.Sequential(nn.Linear(cfgs_cls[0], cfgs_cls[1]),
                                            nn.Linear(cfgs_cls[1], cfgs_cls[2]))

    def forward(self, x):
        equi_feat = self.encoder(x)
        inv_feat = self.GAP(equi_feat)
        x = inv_feat.view(inv_feat.shape[0], -1)
        x = self.classifier(x)
        return x, equi_feat, inv_feat

    def make_layer(self, args, cfgs, reverse):
        layer=[]
        for v in cfgs:
            if isinstance(v, list):
                layer += [SWN_GCN_layer(args, v, reverse)]
        return torch.nn.Sequential(*layer)
