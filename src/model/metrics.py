import torch
import torch.nn as nn
import numpy as np
from skimage.morphology import label


class Dice(nn.Module):
    """The Dice score.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.
        Returns:
            metric (torch.Tensor) (C): The dice scores for each class.
        """
        # Get the one-hot encoding of the prediction and the ground truth label.
        pred = output.argmax(dim=1, keepdim=True)
        pred = torch.zeros_like(output).scatter_(1, pred, 1)
        target = torch.zeros_like(output).scatter_(1, target, 1)

        # Calculate the dice score.
        reduced_dims = list(range(2, output.dim())) # (N, C, *) --> (N, C)
        intersection = 2.0 * (pred * target).sum(reduced_dims)
        union = pred.sum(reduced_dims) + target.sum(reduced_dims)
        score = intersection / (union + 1e-10)
        return score.mean(dim=0)


class Accuracy(nn.Module):
    """The accuracy for the classification task.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C): The model output.
            target (torch.LongTensor) (N): The data target.
        Returns:
            metric (torch.Tensor) (0): The accuracy.
        """
        pred = torch.argmax(output, dim=1)
        return (pred == target).float().mean()

        
class FalseNegativeSize(nn.Module):
    """The false negative target size.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor) (N, C, *): The model output.
            target (torch.LongTensor) (N, 1, *): The data target.
        Returns:
            metric (torch.Tensor) (C): The average false negative size for each class.
        """
        scores = []
        # Get the one-hot encoding of the prediction and the ground truth label.
        pred = output.argmax(dim=1, keepdim=True)
        pred = torch.zeros_like(output).scatter_(1, pred, 1)
        target = torch.zeros_like(output).scatter_(1, target, 1)
        
        # Calculate the score for each class
        for i in range(1, output.shape[1]):
            label_target = label(target[:, i].squeeze(dim=0).cpu().numpy(), connectivity=output.dim()-2)
            label_target_list = np.unique(label_target)[1:]
            _pred = pred[:, i].squeeze(dim=0).cpu().numpy()
            
            score = []
            for target_id in label_target_list:
                if (np.sum((_pred == 1) * (label_target == target_id)) == 0):
                    score.append(np.sum(label_target == target_id) / 1000.0)
            scores.append(score)
        return scores