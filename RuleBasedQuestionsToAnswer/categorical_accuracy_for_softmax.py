from typing import Optional

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric

from collections import Counter

@Metric.register("categorical_accuracy_for_softmax")
class CategoricalAccuracySoftmax(Metric):
    """
    Categorical Top-K accuracy. Assumes integer labels, with
    each item to be classified having a single correct class.
    Tie break enables equal distribution of scores among the
    classes with same maximum predicted scores.
    """
    def __init__(self) -> None:
        self.correct_count = 0.
        self.total_count = 0.
        self.instances_count = 0.
        self.total_reciprocal_rank = 0.
        self.total_position = 0.
        self.all_pos = list()

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        """
        predictions, gold_labels = self.unwrap_to_tensors(predictions, gold_labels)


        predictions = predictions.view(-1)
        # print("predictions shape", predictions.shape)
        gold_labels = gold_labels.view(-1).long()
        # print("gold_labels", gold_labels.shape)
        top_k = torch.zeros_like(predictions)
        # print("top_k", top_k.shape)
        # print("predictions", predictions)
        # print("Predictions index", predictions.max(-1)[1].unsqueeze(-1))
        top_k.scatter_(0,predictions.max(-1)[1].unsqueeze(-1),1)
        correct = top_k.eq(gold_labels.float()).float()
        # print("TOP k", top_k)
        # print("Gold labels", gold_labels.float())
        # print("correct_count", correct.sum())
        # print("total count", predictions.shape[0])
        self.total_count += predictions.shape[0]
        self.correct_count += correct.sum()
        # Also compute mean reciprocal rank
        sorted_predictions, sorting_indices = torch.sort(predictions, descending=True)
        sorted_gold_labels = gold_labels[sorting_indices]
        pos = 1.0
        for label in sorted_gold_labels:
            if label == 1:
                break
            pos += 1.0
        # print("Predictions", predictions)
        # print("Sorted predictions", sorted_predictions)
        # print("gold_labels", gold_labels)
        # print("Sorted labels", sorted_gold_labels)
        # print("pos", pos)
        self.instances_count += 1.0
        self.total_reciprocal_rank += 1.0/pos
        self.total_position += pos
        self.all_pos.append(pos)

        # exit()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        if self.instances_count > 1e-12:
            # instead of accuracy we will calculate MRR and return that instead
            mrr = float(self.total_reciprocal_rank)/float(self.instances_count)
        else:
            mrr = 0.0
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return mrr
        # return accuracy

    @overrides
    def reset(self):
        # print("Reset")
        # print("instances_count", self.instances_count)
        # print("total_reciprocal_rank", self.total_reciprocal_rank)
        # print("mrr", float(self.total_reciprocal_rank)/float(self.instances_count))
        # print("avg position", float(self.total_position)/float(self.instances_count))
        # print("all pos", Counter(self.all_pos))
        self.correct_count = 0.0
        self.total_count = 0.0
        self.instances_count = 0.0
        self.total_reciprocal_rank = 0.0
        self.total_position = 0.0