import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class MLP(nn.Module):
    """
        Project or Predict head
    """
    def __init__(self, 
                 in_dim, 
                 out_dim=256, 
                 hidden_dim=4096, 
                 use_bn=True, 
                 nlayers=2, 
                 norm_last_layer=True,
                 act_layer=nn.ReLU):
        super().__init__()
        self.norm_last_layer = norm_last_layer
        nlayers = max(1, nlayers)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_layer())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(act_layer())
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = nn.Sequential(*layers)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x = self.mlp(x)
        if self.norm_last_layer:
            x = nn.functional.normalize(x, dim=-1, p=2)
        return x