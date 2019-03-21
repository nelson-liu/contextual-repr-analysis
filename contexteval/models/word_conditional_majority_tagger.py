from collections import Counter
import logging
import typing
from typing import Dict, List, Optional

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_mask_from_sequence_lengths
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, SpanBasedF1Measure

import numpy
from overrides import overrides
import torch

from contexteval.common.util import is_empty_metric

logger = logging.getLogger(__name__)


@Model.register("word_conditional_majority_tagger")
class WordConditionalMajorityTagger(Model):
    """
    The ``WordConditionalMajorityTagger`` takes a series of tokens as input and predicts tags
    for each token by outputting the token's most frequent tag in the train set. If a word has not
    been seen in the train set, the overall most frequent tag is returned.

    Token-tag statistics are only updated when this ``Model`` is set to train mode
    (e.g., with ``model.train()``). As a result, only one epoch of "training" is necessary to
    fit the entire train set.

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
                 calculate_span_f1: bool = False,
                 label_encoding: Optional[str] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(WordConditionalMajorityTagger, self).__init__(vocab, regularizer)

        self._num_classes = self.vocab.get_vocab_size("labels")
        self._total_label_counts: typing.Counter[str] = Counter()
        self._token_label_counts: Dict[str, typing.Counter[str]] = {}

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }

        self.calculate_per_label_f1 = calculate_per_label_f1
        self.calculate_span_f1 = calculate_span_f1
        if label_encoding and label_encoding not in ["BIO", "BIOUL", "IOB1"]:
            raise ConfigurationError("If not None, label encoding must be one of BIO, BIOUL, "
                                     "or IOB1. Got {}".format(label_encoding))
        self.label_encoding = label_encoding

        label_metric_name = "label_{}" if self.calculate_per_label_f1 else "_label_{}"
        for label_name, label_index in self.vocab._token_to_index["labels"].items():
            self.metrics[label_metric_name.format(label_name)] = F1Measure(positive_label=label_index)

        if self.calculate_span_f1:
            if not self.label_encoding:
                raise ConfigurationError("label_encoding must be provided when "
                                         "calculating_span_f1 is true.")
            else:
                # Set up span-based F1 measure
                self.metrics["span_based_f1"] = SpanBasedF1Measure(self.vocab,
                                                                   tag_namespace="labels",
                                                                   label_encoding=self.label_encoding)

        # Whether to run in error analysis mode or not, see commands.error_analysis
        self.error_analysis = False
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                raw_tokens: List[List[Optional[str]]],
                labels: torch.LongTensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        raw_tokens : List[List[Optional[str]]]
            A batch of lists with the raw token strings.
        labels : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.

        Retpurns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word. We set a 1 to the majority tag.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word. We set a 1 to the majority tag.
        loss : torch.FloatTensor, optional
            There is no "loss" in this model, since there are no parameters to optimize.
            We still return a scalar loss for compatibility with AllenNLP, and set this to 0.
        """
        lengths = []
        max_length = -1
        for token_sequence in raw_tokens:
            # find the first "None".
            # TODO (nfliu): This assumes that when we see the first padding token,
            # all subsequent tokens are padding. It would be nice to relax this assumption.
            try:
                length = token_sequence.index(None)
            except ValueError:
                # not found, so there is no padding here
                length = len(token_sequence)
            if length > max_length:
                max_length = length
            lengths.append(length)

        lengths = torch.LongTensor(lengths)
        mask = get_mask_from_sequence_lengths(lengths, max_length)

        batch_size, seq_len = mask.size()
        if labels is not None and self.training:
            # We want to update our counts
            if len(raw_tokens) != labels.size(0):
                raise ValueError("raw_tokens and labels have differing sequence lengths: "
                                 "{} and {} (respectively)".format(len(raw_tokens), labels.size(0)))

            for token_sequence, label_sequence in zip(raw_tokens, labels):
                for token, label in zip(token_sequence, label_sequence):
                    if token is None:
                        # This token and label are padding, so we ignore
                        continue
                    self._total_label_counts[label.item()] += 1
                    if token not in self._token_label_counts:
                        self._token_label_counts[token] = Counter()
                    self._token_label_counts[token][label.item()] += 1

        # Get the predicted labels for each token.
        predicted_labels = []
        for token_sequence in raw_tokens:
            token_sequence_labels = [self._get_token_label(token) for token in
                                     token_sequence if token is not None]
            predicted_labels.append(token_sequence_labels)

        logits_and_class_probabilities = torch.zeros(
            batch_size, seq_len, self._num_classes)
        for token_sequence_logits, token_sequence_labels in zip(
                logits_and_class_probabilities, predicted_labels):
            # token_sequence_logits shape: (seq_len, num_classes)
            # token_sequence_labels is a list of int (len: seq_len)
            for token_logits, label in zip(token_sequence_logits, token_sequence_labels):
                # token_logits: (num_classes)
                token_logits[label] = 1.0
        output_dict = {
            "logits": logits_and_class_probabilities,
            "class_probabilities": logits_and_class_probabilities}

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
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        all_predictions = all_predictions.cpu().data.numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions in predictions_list:
            argmax_indices = numpy.argmax(predictions, axis=-1)
            tags = [self.vocab.get_token_from_index(x, namespace="labels")
                    for x in argmax_indices]
            all_tags.append(tags)
        output_dict['tags'] = all_tags
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
            elif isinstance(metric, SpanBasedF1Measure):
                metric_dict = metric.get_metric(reset=reset)
                for name, value in metric_dict.items():
                    if "overall" in name:
                        metrics_to_return[name] = value
            else:
                raise ValueError("metric {} not supported yet".format(type(metric)))
        return metrics_to_return

    @overrides
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        pytorch_created_state = super(WordConditionalMajorityTagger, self).state_dict()
        pytorch_created_state["_total_label_counts"] = self._total_label_counts
        pytorch_created_state["_token_label_counts"] = self._token_label_counts
        return pytorch_created_state

    @overrides
    def load_state_dict(self, state_dict, strict=True):
        _total_label_counts = state_dict.pop("_total_label_counts")
        _token_label_counts = state_dict.pop("_token_label_counts")
        super(WordConditionalMajorityTagger, self).load_state_dict(state_dict, strict=strict)
        self._total_label_counts = _total_label_counts
        self._token_label_counts = _token_label_counts
