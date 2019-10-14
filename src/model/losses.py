import torch
import torch.nn as nn
import numpy as np


class DiceLoss(nn.Module):
    """The Dice loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target, segments):
        """
        Args:
            output (torch.Tensor) (V, F): The model output.
            target (torch.LongTensor) (N, N): The data target.
            segments (torch.LongTensor) (N, N)
        Returns:
            loss (torch.Tensor) (0): The dice loss.
        """

        n_class = output.size(1)

        output_img = torch.zeros_like(target)
        output = torch.argmax(output, 1)

        mask0 = torch.zeros_like(target)
        mask1 = torch.ones_like(target)

        for i in range(segments.max() + 1):
            output_img += torch.where(segments==i, output[i]*mask1, mask0)


        # Get the one-hot encoding of the ground truth label. (C, N, N)
        target = torch.unsqueeze(target, 0)
        output_img = torch.unsqueeze(output_img, 0)
        target = torch.zeros(n_class, target.size(0), target.size(1)).scatter_(0, target, 1)
        output_img = torch.zeros(n_class, target.size(0), target.size(1)).scatter_(0, output_img, 1)

        # Calculate the dice loss.
        reduced_dims = list(range(1, target.dim())) # (C, N, N) --> (C)
        intersection = 2.0 * (output_img * target).sum(reduced_dims)
        union = (output_img ** 2).sum(reduced_dims) + (target ** 2).sum(reduced_dims)
        score = intersection / (union + 1e-10)
        return 1 - score.mean()
