from collections import Counter
import logging
import typing
from typing import Dict, List, Optional

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

import numpy
from overrides import overrides
import torch

from contexteval.common.util import is_empty_metric

logger = logging.getLogger(__name__)


@Model.register("word_conditional_majority_selective_tagger")
class WordConditionalMajoritySelectiveTagger(Model):
    """
    The ``WordConditionalSelectiveMajorityTagger`` takes a token as input and
    predicts a tag by outputting the token's most frequent tag in the train set.
    If a word has not been seen in the train set, the overall most frequent tag
    is returned.

    Token-tag statistics are only updated when this ``Model`` is set to train mode
    (e.g., with ``model.train()``). As a result, only one epoch of "training" is
    necessary to fit the entire train set.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to return a logit / class probabilities tensor
        of the proper shape.
    calculate_per_label_f1 : ``bool``, optional (default=``False``)
        Calculate per-label f1 metrics during training. This is only necessary if you
        want to use a label's f1 score as the validation metric; to get per-label F1
        scores of test data, use the `error-analysis` command. This is recommended to
        be set to False when the label space is large, as it will significantly
        slow the model.
    calculate_span_f1 : ``bool``, optional (default=``False``)
        Calculate span-level f1 metrics.
    label_encoding : ``str``, optional (default=``None``)
        Label encoding to use when calculating span f1. Valid options are
        "BIO", "BIOUL", "IOB1". Required if ``calculate_span_f1`` is true.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 calculate_per_label_f1: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(WordConditionalMajoritySelectiveTagger, self).__init__(vocab, regularizer)

        self._num_classes = self.vocab.get_vocab_size("labels")
        self._total_label_counts: typing.Counter[str] = Counter()
        self._token_label_counts: Dict[str, typing.Counter[str]] = {}

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }

        self.calculate_per_label_f1 = calculate_per_label_f1

        label_metric_name = "label_{}" if self.calculate_per_label_f1 else "_label_{}"
        for label_name, label_index in self.vocab._token_to_index["labels"].items():
            self.metrics[label_metric_name.format(label_name)] = F1Measure(positive_label=label_index)

        # Whether to run in error analysis mode or not, see commands.error_analysis
        self.error_analysis = False
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                label_indices: torch.LongTensor,
                raw_tokens: List[List[str]],
                labels: torch.LongTensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        label_indices : torch.LongTensor
            A LongTensor of shape (batch_size, max_num_adpositions) with the tokens
            to predict a label for for each element (sentence) in the batch.
        raw_tokens : List[List[str]], optional (default = None)
            A batch of lists with the raw token strings. Used to compute
            token_representations, if either are None.
        labels : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels
            of shape ``(batch_size, num_label_indices)``.

        Retpurns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, tag_vocab_size)`` representing
            a distribution of the tag classes per word. We set a 1 to the majority tag.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, tag_vocab_size)`` representing
            a distribution of the tag classes per word. We set a 1 to the majority tag.
        loss : torch.FloatTensor, optional
            There is no "loss" in this model, since there are no parameters to optimize.
            We still return a scalar loss for compatibility with AllenNLP, and set this to 0.
        """
        label_indices = label_indices.long()
        batch_size = label_indices.size(0)
        max_num_labels = label_indices.size(1)
        mask = self._get_text_mask_from_label_indices(label_indices)

        if labels is not None and self.training:
            # We want to update our counts
            if batch_size != labels.size(0):
                raise ValueError("token_index and label have differing number of sequences: "
                                 "{} and {} (respectively)".format(
                                     label_indices.size(0), labels.size(0)))
            for token_sequence, sequence_label_indices, sequence_labels in zip(
                    raw_tokens, label_indices, labels):
                for sequence_token_index, sequence_label in zip(
                        sequence_label_indices, sequence_labels):
                    token = token_sequence[sequence_token_index]
                    self._total_label_counts[sequence_label.item()] += 1
                    if token not in self._token_label_counts:
                        self._token_label_counts[token] = Counter()
                    self._token_label_counts[token][sequence_label.item()] += 1

        # Get the predicted labels for each token represented by token_index.
        predicted_labels = []
        for token_sequence, sequence_label_indices in zip(raw_tokens, label_indices):
            sequence_predictions = []
            for sequence_token_index in sequence_label_indices:
                token = token_sequence[sequence_token_index]
                sequence_predictions.append(self._get_token_label(token))
            predicted_labels.append(sequence_predictions)

        logits_and_class_probabilities = torch.zeros(
            batch_size, max_num_labels, self._num_classes)
        for sequence_logits, sequence_predictions in zip(
                logits_and_class_probabilities, predicted_labels):
            for token_logits, pred_label in zip(
                    sequence_logits, sequence_predictions):
                # token_logits shape: (num_classes,)
                token_logits[pred_label] = 1.0

        output_dict = {
            "logits": logits_and_class_probabilities,
            "class_probabilities": logits_and_class_probabilities
        }

        if labels is not None:
            for name, metric in self.metrics.items():
                # When not running in error analysis mode, skip
                # metrics that start with "_"
                if not self.error_analysis and name.startswith("_"):
                    continue
                metric(logits_and_class_probabilities, labels, mask.float())
            loss = torch.zeros(1, requires_grad=True)
            # Hack to make the loss not a leaf variable
            hack = torch.zeros(1)
            output_dict["loss"] = loss + hack
        return output_dict

    def _get_token_label(self, token: str):
        if token in self._token_label_counts:
            # In-vocabulary token, return the token's most common tag
            return self._token_label_counts[token].most_common(1)[0][0]
        else:
            # OOV token, return most common tag.
            return self._total_label_counts.most_common(1)[0][0]

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"labels"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        all_predictions = all_predictions.cpu().data.numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]

        all_labels = []
        for predictions in predictions_list:
            argmax_indices = numpy.argmax(predictions, axis=-1)
            labels = [self.vocab.get_token_from_index(x, namespace="labels")
                      for x in argmax_indices]
            all_labels.append(labels)
        output_dict['labels'] = all_labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # When not running in error analysis mode, skip
        # metrics that start with "_".
        metrics_to_return = {}
        for name, metric in self.metrics.items():
            # If not error analysis mode and label starts with _,
            # skip this metric
            if not self.error_analysis and name.startswith("_"):
                continue
            # Skip empty metrics.
            if is_empty_metric(metric):
                continue
            # Get the value of the metric
            if isinstance(metric, CategoricalAccuracy):
                metrics_to_return[name] = metric.get_metric(reset)
            elif isinstance(metric, F1Measure):
                precision, recall, f1 = metric.get_metric(reset)
                metrics_to_return[name + "_precision"] = precision
                metrics_to_return[name + "_recall"] = recall
                metrics_to_return[name + "_f1"] = f1
            else:
                raise ValueError("metric {} not supported yet".format(type(metric)))
        return metrics_to_return

    @overrides
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        pytorch_created_state = super(WordConditionalMajoritySelectiveTagger,
                                      self).state_dict()
        pytorch_created_state["_total_label_counts"] = self._total_label_counts
        pytorch_created_state["_token_label_counts"] = self._token_label_counts
        return pytorch_created_state

    @overrides
    def load_state_dict(self, state_dict, strict=True):
        _total_label_counts = state_dict.pop("_total_label_counts")
        _token_label_counts = state_dict.pop("_token_label_counts")
        super(WordConditionalMajoritySelectiveTagger, self).load_state_dict(state_dict, strict=strict)
        self._total_label_counts = _total_label_counts
        self._token_label_counts = _token_label_counts

    def _get_text_mask_from_label_indices(self, label_indices):
        """
        Extract the text mask given padded token indices.

        Parameters
        ----------
        label_indices : torch.LongTensor, optional (default = None)
            A LongTensor of shape (batch_size, 2) with the token pairs
            to predict a label for for each element in the batch. Padding is denoted
            by negative elements, since using 0 doesn't work (as 0 could be a
            valid element, referring to the 0th index).

        Returns
        -------
        mask : torch.LongTensor
            A mask of shape (batch_size, max_num_arcs) with 1 in positions where
            there is label to predict and 0 in places with padding.
        """
        mask = (label_indices >= 0).long()
        assert mask.dim() == 2
        return mask
