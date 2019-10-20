import torch
from src.runner.trainers.base_trainer import BaseTrainer


class BrainTrainer(BaseTrainer):
    """The KiTS trainer for segmentation task.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_inputs_targets(self, batch):
        """Specify the data input and target.
        Args:
            batch (dict): A batch of data.
        Returns:
            input (torch.Tensor): The data input.
            target (torch.LongTensor): The data target.
        """
        f = torch.squeeze(batch['features'], 0)
        #a = torch.squeeze(batch['adj_arr'], 0)
        l = torch.squeeze(batch['label'], 0)
        s = torch.squeeze(batch['segments'], 0)
        t = batch['tao']
        #return batch['features'], batch['adj_arr'], batch['label'], batch['segments']
        #return f, a, l, s
        return f, l, s, t

    def _compute_losses(self, output, target, segments):
        """Compute the losses.
        Args:
            output (torch.Tensor): The model output.
            target (torch.LongTensor): The data target.
        Returns:
            losses (list of torch.Tensor): The computed losses.
        """
        losses = [loss(output, target, segments) for loss in self.loss_fns]
        return losses

    def _compute_metrics(self, output, target, segments):
        """Compute the metrics.
        Args:
             output (torch.Tensor): The model output.
             target (torch.LongTensor): The data target.
        Returns:
            metrics (list of torch.Tensor): The computed metrics.
        """
        metrics = [metric(output, target, segments) for metric in self.metric_fns]
        return metrics

    def _init_log(self):
        """Initialize the log.
        Returns:
            log (dict): The initialized log.
        """
        log = {}
        log['Loss'] = 0
        for loss in self.loss_fns:
            log[loss.__class__.__name__] = 0
        for metric in self.metric_fns:
            if metric.__class__.__name__ == 'Dice':
                log['Dice'] = 0
                for i in range(self.net.n_class):
                    log[f'Dice_{i}'] = 0
            else:
                log[metric.__class__.__name__] = 0
        return log

    def _update_log(self, log, batch_size, loss, losses, metrics):
        """Update the log.
        Args:
            log (dict): The log to be updated.
            batch_size (int): The batch size.
            loss (torch.Tensor): The weighted sum of the computed losses.
            losses (list of torch.Tensor): The computed losses.
            metrics (list of torch.Tensor): The computed metrics.
        """
        log['Loss'] += loss.item() * batch_size
        for loss, _loss in zip(self.loss_fns, losses):
            log[loss.__class__.__name__] += _loss.item() * batch_size
        for metric, _metric in zip(self.metric_fns, metrics):
            if metric.__class__.__name__ == 'Dice':
                log['Dice'] += _metric.mean().item() * batch_size
                for i, class_score in enumerate(_metric):
                    log[f'Dice_{i}'] += class_score.item() * batch_size
            else:
                log[metric.__class__.__name__] += _metric.item() * batch_size
