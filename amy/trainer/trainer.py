from typing import Callable, Dict, Optional, Union, Tuple, List
from collections import defaultdict
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import monai

import sklearn.metrics as skm

# from torchvision.utils import make_grid
# from base import BaseTrainer
# from utils import inf_loop, MetricTracker

from .base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self,
                 model: torch.nn.Module,
                 loss_function: Callable,
                 metric_ftns: Dict[str, Callable],
                 optimizer: torch.optim,
                 lr_scheduler: torch.optim.lr_scheduler,
                 config: dict,
                 project: str,
                 data_loader: torch.utils.data.dataloader,
                 valid_data_loader: torch.utils.data.dataloader = None,
                 seed: int = None,
                 device: str = None,
                 tags: Optional[List[str]] = None,
                 log_step: int = None,
                 mixed_precision: bool = False,
                 entity: str = "amyloid",
                 classification: bool = False,
                 crop_foreground: bool = False,
                 ):

        super().__init__(model=model,
                         loss_function=loss_function,
                         metric_ftns=metric_ftns,
                         optimizer=optimizer,
                         config=config,
                         lr_scheduler=lr_scheduler,
                         seed=seed,
                         device=device,
                         project=project,
                         tags=tags,
                         entity=entity,
                         )

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader

        self.classification = classification

        self.crop_foreground = monai.transforms.CropForeground(k_divisible=16, return_coords=True, mode='constant') if crop_foreground else None
        self.mixed_precision = mixed_precision

        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        self.inputs_pr_iteration = int(config['inputs_pr_iteration'])

        self.batch_size = data_loader.batch_size
        self.len_epoch = len(data_loader) if not self.iterative else self.inputs_pr_iteration
        self.log_step = int(self.len_epoch/4) if not isinstance(log_step, int) else int(log_step/self.batch_size)

        if self.classification:
            self.balanced_accuracy = list()
            self.auc = list()
            self.best_auc = 0.0
            self.best_balanced_accuracy = 0.0
            self.min_validation_loss = 0

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        losses = defaultdict(list)

        for batch_idx, (data, target) in enumerate(self.data_loader):
            if self.crop_foreground is not None:
                with torch.no_grad():
                    _, low, high = self.crop_foreground(torch.sum(torch.abs(data.to(self.device)), dim=0).to("cpu"))
                    data = data[:, :, low[0]:high[0], low[1]:high[1], low[2]:high[2]]

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()


            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    out = self.model(data)
                    loss = self._loss(out, target)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model(data)
                loss = self._loss(out, target)

                loss.backward()
                self.optimizer.step()

            loss = loss.item()  # Detach loss from comp graph and moves it to the cpu
            losses['loss'].append(loss)

            if batch_idx % self.log_step == 0:
                print('Train {}: {} {} Loss: {:.6f}'.format(
                    'Epoch' if not self.iterative else 'Iteration',
                    epoch,
                    self._progress(batch_idx),
                    loss))

            if batch_idx >= self.inputs_pr_iteration and self.iterative:
                break

        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        losses['loss_func'] = str(self.loss_function)

        return {"loss": np.mean(losses["loss"])}

    def _loss(self, out: Union[torch.Tensor, Tuple[torch.Tensor]], target: torch.Tensor):

        if isinstance(out, (list, tuple)):
            output, auxiliary = out

            loss = self.loss_function(output, target)
            auxiliary = auxiliary if isinstance(auxiliary, list) else [auxiliary]
            for aux in auxiliary:
                loss += 0.33*self.loss_function(aux, target)

            return loss
        output = out

        if len(output.shape) == 6:
            loss = self.loss_function(output[:, 0], target)  # The actual prediction not deep supervision
            ssum = 1.
            for t in range(1, output.shape[1]):
                loss += self.loss_function(output[:, t], target)/(2.0**t)
                ssum += 1/(2.0**t)
            loss /= ssum
            if torch.isnan(loss):
                return torch.tensor(0.0).to(self.device)
            return loss

        loss = self.loss_function(output, target)
        if torch.isnan(loss):
                return torch.tensor(0.0).to(self.device)
        return loss

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        if self.valid_data_loader is None:
            return None

        self.model.eval()
        metrics = defaultdict(list)

        tot_cases = sum([len(i) for i in self.valid_data_loader.values()])
        comb = list()
        balanced_comb = list()

        for key, valid_data_loader in self.valid_data_loader.items():

            if self.classification:
                preds = list()
                targets = list()

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(valid_data_loader):
                    if self.crop_foreground is not None:
                        with torch.no_grad():
                            _, low, high = self.crop_foreground(torch.sum(torch.abs(data.to(self.device)), dim=0).to("cpu"))
                            data = data[:, :, low[0]:high[0], low[1]:high[1], low[2]:high[2]]

                    data, target = data.to(self.device), target.to(self.device)

                    out = self.model(data)
                    loss = self.loss_function(out, target)
                    metrics['{}_val_loss'.format(key)].append(loss.item())
                    metrics["val_loss"].append(loss.item())

                    if self.classification:
                        with torch.no_grad():
                            out2 = torch.sigmoid(out)
                            preds.append(out2.cpu().numpy())
                            targets.append(target.cpu().numpy())

                    for key_2, metric in self.metric_ftns.items():
                        if self.metrics_is_dict:
                            metrics[key_2].append(metric(out.cpu(), target.cpu()).item())
                        else:
                            metrics[key_2].append(metric(out, target).item())

            if self.classification:
                preds = np.concatenate(preds, axis=0)
                targets = np.concatenate(targets, axis=0)
                metrics["{}_balanced_accuracy".format(key)].append(skm.balanced_accuracy_score(targets, preds > 0.5))
                metrics["{}_auc".format(key)].append(skm.roc_auc_score(targets, preds))

                metrics["{}_accuracy_0.3".format(key)] = np.mean((preds > 0.3) == targets)
                metrics["{}_accuracy_0.5".format(key)] = np.mean((preds > 0.5) == targets)
                metrics["{}_accuracy_0.7".format(key)] = np.mean((preds > 0.7) == targets)

                metrics["balanced_accuracy"].append((targets, preds))
                comb.append(skm.roc_auc_score(targets, preds)*len(valid_data_loader) / tot_cases)
                balanced_comb.append(skm.balanced_accuracy_score(targets, preds > 0.5)*len(valid_data_loader) / tot_cases)

        if self.classification:
            # calculate balanced accuracy
            targets, preds = np.concatenate([i[0] for i in metrics["balanced_accuracy"]], axis=0), np.concatenate([i[1] for i in metrics["balanced_accuracy"]], axis=0)
            metrics["balanced_accuracy"] = skm.balanced_accuracy_score(targets, preds > 0.5)
            metrics["auc"] = skm.roc_auc_score(targets, preds)
            metrics["weighted_auc"] = np.sum(comb)
            metrics["weighted_balanced_accuracy"] = np.sum(balanced_comb)

        metric_dict = dict()
        for key, item in metrics.items():
            metric_dict[key] = np.mean(metrics[key])

        if self.classification:
            self.auc.append(metric_dict["weighted_auc"])
            self.balanced_accuracy.append(metric_dict["weighted_balanced_accuracy"])
            if epoch > 3:
                # check if the avareage balanced accuracy of the last 3 epochs is better than the best
                if np.mean(self.balanced_accuracy[-3:]) > self.best_balanced_accuracy:
                    self.best_balanced_accuracy = np.mean(self.balanced_accuracy[-3:])
                    self.save_checkpoint(epoch, best=True, name="best_balanced_accuracy")

        return metric_dict

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx
            total = self.data_loader.n_samples
        elif hasattr(self.data_loader, 'batch_size'):
            current = batch_idx
            total = self.len_epoch
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
