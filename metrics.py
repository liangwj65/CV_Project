# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import numpy as np
import torch
import torchmetrics
import torchmetrics.functional.classification as class_metrics


class Metrics:
    def __init__(self,
                 metrics: tuple = ('auc', 'ap', 'iou', 'precision', 'recall', 'f1', 'accuracy', 'f1-best'),
                 threshold: float = 0.5,
                 average=None,
                 return_on_update: bool = True,
                 sync_on_compute: bool = False,
                 approximate_auc: bool = False):

        self.metrics: tuple[str] = metrics
        self.threshold: float = threshold
        self.average = average

        # Map of metrics' methods implemented internally to their names.
        self.builtin_metric_functions = {'iou': calculate_iou,
                                         'precision': calculate_precision,
                                         'recall': calculate_recall,
                                         'f1': calculate_f1,
                                         'accuracy': calculate_accuracy}

        # Internal state is a Confusion Matrix and a ROC calculator.
        self.confusion_matrix: torchmetrics.classification.BinaryConfusionMatrix = \
            torchmetrics.classification.BinaryConfusionMatrix(threshold=self.threshold,
                                                              sync_on_compute=sync_on_compute)
        self.auroc: Optional[torchmetrics.classification.BinaryAUROC] = None
        self.ap: Optional[torchmetrics.classification.BinaryAveragePrecision] = None
        self.mean_f1best: Optional[torchmetrics.aggregation.MeanMetric] = \
            torchmetrics.aggregation.MeanMetric(sync_on_compute=sync_on_compute)
        if "auc" in self.metrics:
            thresholds: Optional[int] = 100 if approximate_auc else None
            self.auroc = torchmetrics.classification.BinaryAUROC(sync_on_compute=sync_on_compute,
                                                                 thresholds=thresholds)
        if "ap" in self.metrics:
            thresholds: Optional[int] = 100 if approximate_auc else None
            self.ap = torchmetrics.classification.BinaryAveragePrecision(
                sync_on_compute=sync_on_compute, thresholds=thresholds
            )

    def get_metric_names(self) -> tuple[str]:
        return self.metrics

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> dict[str, np.ndarray]:
        targets = torch.where(targets > self.threshold, 1.0, 0.0)
        targets = targets.long()

        res: dict[str, np.ndarray] = {}

        if len(self.metrics) > 0:
            conf_matrix: torch.Tensor = self.confusion_matrix(preds, targets)
            if 'auc' in self.metrics:
                auc: torch.Tensor = self.auroc(preds, targets)
            if 'ap' in self.metrics:
                ap: torch.Tensor = self.ap(preds, targets)

            for metric in self.metrics:
                if metric == 'auc':
                    res[metric] = auc.numpy()
                elif metric == 'ap':
                    res[metric] = ap.numpy()
                elif metric == 'f1-best':
                    if torch.max(targets).numpy().item() > 0:
                        batched_bestf1: torch.Tensor = calculate_f1best(preds, targets)
                    else:
                        batched_bestf1: torch.Tensor = calculate_f1best(-preds+1, -targets+1)
                    self.mean_f1best.update(batched_bestf1)
                    res[metric] = torch.mean(batched_bestf1).detach().numpy()
                else:
                    res[metric] = self.builtin_metric_functions[metric](
                        conf_matrix, average=self.average
                    ).numpy()

        return res

    def compute(self) -> dict[str, np.ndarray]:
        res: dict[str, np.ndarray] = {}

        if len(self.metrics) > 0:
            conf_matrix: torch.Tensor = self.confusion_matrix.compute()
            if 'auc' in self.metrics:
                auc: torch.Tensor = self.auroc.compute()
            if 'ap' in self.metrics:
                ap: torch.Tensor = self.ap.compute()

            for metric in self.metrics:
                if metric == 'auc':
                    res[metric] = auc.numpy()
                elif metric == 'ap':
                    res[metric] = ap.numpy()
                elif metric == 'f1-best':
                    res[metric] = self.mean_f1best.compute().detach().numpy()
                else:
                    res[metric] = self.builtin_metric_functions[metric](
                        conf_matrix, average=self.average
                    ).numpy()

        return res

    def reset(self) -> None:
        if self.confusion_matrix:
            self.confusion_matrix.reset()
        if self.auroc:
            self.auroc.reset()
        if self.ap:
            self.ap.reset()
        if self.mean_f1best:
            self.mean_f1best.reset()


def calculate_accuracy(conf_matrix: torch.tensor, average=None) -> torch.tensor:
    return conf_matrix.diag().sum() / conf_matrix.sum()


def calculate_iou(conf_matrix: torch.tensor, average=None) -> torch.tensor:
    true_positive = torch.diag(conf_matrix)
    false_positive = conf_matrix.sum(0) - true_positive
    false_negative = conf_matrix.sum(1) - true_positive
    iou = true_positive / (true_positive + false_positive + false_negative)
    if average == 'macro':
        return iou.mean()
    else:
        return iou


def calculate_precision(conf_matrix: torch.tensor, average=None) -> torch.tensor:
    true_positive = torch.diag(conf_matrix)
    false_positive = conf_matrix.sum(0) - true_positive
    precision = true_positive / (true_positive + false_positive)
    precision = torch.where(torch.isnan(precision), torch.ones_like(precision), precision)
    if average == 'macro':
        return precision.mean()
    else:
        return precision


def calculate_recall(conf_matrix: torch.tensor, average=None) -> torch.tensor:
    true_positive = torch.diag(conf_matrix)
    false_negative = conf_matrix.sum(1) - true_positive
    recall = true_positive / (true_positive + false_negative)
    if average == 'macro':
        return recall.mean()
    else:
        return recall


def calculate_f1(conf_matrix: torch.tensor, average=None) -> torch.tensor:
    true_positive = torch.diag(conf_matrix)
    false_negative = conf_matrix.sum(1) - true_positive
    false_positive = conf_matrix.sum(0) - true_positive
    precision = true_positive / (true_positive + false_positive)
    precision = torch.where(torch.isnan(precision), torch.ones_like(precision), precision)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    if average == 'macro':
        return f1.mean()
    else:
        return f1


def calculate_f1best(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert preds.size(dim=0) == target.size(dim=0)
    batched_bestf1: torch.Tensor = torch.zeros((preds.size(dim=0),))
    for i in range(preds.size(dim=0)):
        precision, recall, _ = class_metrics.binary_precision_recall_curve(preds, target)
        precision = torch.where(torch.isnan(precision), torch.ones_like(precision), precision)
        f1: torch.Tensor = 2 * recall * precision / (recall + precision)
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
        batched_bestf1[i] = torch.max(f1)
    return batched_bestf1

