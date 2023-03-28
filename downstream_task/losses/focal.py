import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self,  gamma=2, alpha=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logit, target):

        logpt = -self.ce(logit, target)
        pt = torch.exp(logpt)
        if self.alpha is not None:
            logpt *= self.alpha
        loss = -((1 - pt) ** self.gamma) * logpt

        return loss
    

    
class SoftTargetFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, logit, target):
        logpt = F.log_softmax(logit, dim=1)
        pt  = torch.exp(logpt)
        if self.alpha is not None:
            logpt *= self.alpha
        logit = -((1 - pt) ** self.gamma) * logpt
        
        return torch.sum(logit * target, dim=1).mean()