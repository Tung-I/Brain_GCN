import torch
import torch.nn as nn
import numpy as np


class MyBCELoss(nn.Module):

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.FloatTensor(weight).cuda()

    def forward(self, output, target):
        #print(target.size())
        target = torch.zeros_like(output).scatter_(1, target, 1)
        #print(output.size())
        #print(target.size())
        loss = nn.BCELoss(weight=self.weight)
        return loss(output, target)

class DiceLoss(nn.Module):
    """The Dice loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target, segments):
        """
        Args:
            output (torch.Tensor) (V, F): V is the num of vertex, F is the num of features.
            target (torch.LongTensor) (N, N): The N*N image label.
            segments (torch.LongTensor) (N, N): The output of SLIC superpixel
        Returns:
            loss (torch.Tensor) (0): The dice loss.
        """

        n_class = output.size(1)

        output_img = torch.zeros_like(target).cuda()
        output = torch.argmax(output, 1)

        mask0 = torch.zeros_like(target)
        mask1 = torch.ones_like(target)
        
        # output(V, F) -> output_img(N, N)
        n_range = output.size(0) if output.size(0)<(segments.max()+1) else segments.max()+1
        for i in range(n_range):
            output_img += torch.where(segments==i, output[i]*mask1, mask0)

        ######
        to_save = output_img.clone().detach()
        to_save = np.asarray(to_save.cpu())
        np.save('/home/tony/Documents/output_img.npy', to_save)
        #####

        # Get the one-hot encoding of both the target and output_img
        template = torch.zeros(n_class, target.size(0), target.size(1)).cuda()
        target = torch.unsqueeze(target, 0)
        output_img = torch.unsqueeze(output_img, 0)
        target = torch.zeros_like(template).scatter_(0, target, 1)
        output_img = torch.zeros_like(template).scatter_(0, output_img, 1)
        
        output_img.requires_grad = True

        # Calculate the dice loss.
        reduced_dims = list(range(1, target.dim())) # (C, N, N) --> (C)
        intersection = 2.0 * (output_img * target).sum(reduced_dims)
        union = (output_img ** 2).sum(reduced_dims) + (target ** 2).sum(reduced_dims)
        score = intersection / (union + 1e-10)
        #print(1 - score.mean())
        return 1 - score.mean()
