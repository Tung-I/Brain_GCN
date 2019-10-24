import torch
import torch.nn as nn
import numpy as np
from skimage.morphology import label


class Dice(nn.Module):
    """The Dice score.
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

        pred = torch.argmax(output, 1, keepdim=True)

        # target = target.unsqueeze(0)
        # Get the one-hot encoding of the ground truth label.
        target = torch.zeros_like(output).scatter_(1, target, 1)
        pred = torch.zeros_like(output).scatter_(1, pred, 1)

        # Calculate the dice loss.

        intersection = 2.0 * (pred * target).sum(0)
        union = (pred ** 2).sum(0) + (target ** 2).sum(0)
        score = intersection / (union + 1e-10)
        return score


# class Dice(nn.Module):
#     """The Dice score.
#     """
#     def __init__(self):
#         super().__init__()


#     def forward(self, output, target, segments):
#         """
#         Args:
#             output (torch.Tensor) (V, F): The model output.
#             target (torch.LongTensor) (N, N): The data target.
#             segments (torch.LongTensor) (N, N)
#         Returns:
#             loss (torch.Tensor) (0): The dice loss.
#         """

#         n_class = output.size(1)

#         output_img = torch.zeros_like(target).cuda()
#         output = torch.argmax(output, 1)

#         mask0 = torch.zeros_like(target)
#         mask1 = torch.ones_like(target)

#         n_range = output.size(0) if output.size(0)<(segments.max()+1) else segments.max()+1
#         for i in range(n_range):
#             output_img += torch.where(segments==i, output[i]*mask1, mask0)


#         # Get the one-hot encoding of the ground truth label. (C, N, N)
#         template = torch.zeros(n_class, target.size(0), target.size(1)).cuda()
#         target = torch.unsqueeze(target, 0)
#         output_img = torch.unsqueeze(output_img, 0)
#         target = torch.zeros_like(template).scatter_(0, target, 1)
#         output_img = torch.zeros_like(template).scatter_(0, output_img, 1)

#         # Calculate the dice loss.
#         reduced_dims = list(range(1, target.dim())) # (C, N, N) --> (C)
#         intersection = 2.0 * (output_img * target).sum(reduced_dims)
#         union = (output_img ** 2).sum(reduced_dims) + (target ** 2).sum(reduced_dims)
#         score = intersection / (union + 1e-10)
#         return score


class F1Score(nn.Module):
    """The accuracy for the classification task.
    """
    def __init__(self):
        super().__init__()
        self.TP = 0
        self.TN = 0
        self.Fp = 0
        self.FN = 0

    def _reset(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

    def forward(self, output, target, segments):
        """
        Args:
            output (torch.Tensor) (N, C): The model output.
            target (torch.LongTensor) (N): The data target.
        Returns:
            metric (torch.Tensor) (0): The accuracy.
        """
        pred = torch.argmax(output, dim=1)
        pre_mask = torch.zeros_like(output).scatter_(1, pred.cpu().view(-1, 1), 1.)
        tar_mask = torch.zeros_like(output).scatter_(1, target.data.cpu().view(-1, 1), 1.)
        self.TP += (pre_mask[:, 1]*tar_mask[:, 1]).float().sum()
        self.FP += (pre_mask[:, 1]*tar_mask[:, 0]).float().sum()
        self.FN += (pre_mask[:, 0]*tar_mask[:, 1]).float().sum()
        self.TN += (pre_mask[:, 0]*tar_mask[:, 0]).float().sum()
        precision = self.TP/((self.TP+self.FP) + 1e-10) 
        recall = self.TP/((self.TP+self.FN) + 1e-10)
        F1 = 2*precision*recall / ((precision+recall) + 1e-10)


        return (pred == target).float().mean()



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
