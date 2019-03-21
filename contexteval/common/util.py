from typing import Any, List
import torch
from allennlp.nn.util import get_mask_from_sequence_lengths
from allennlp.training.metrics.metric import Metric
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, SpanBasedF1Measure
from contexteval.training.metrics import Perplexity


def get_text_mask_from_representations(
        token_representations: torch.FloatTensor) -> torch.LongTensor:
    """
    Takes the dictionary of tensors produced by a ``ListField[MetadataField]``
    and returns a mask with 0 where the tokens are padding, and 1 otherwise.

    Parameters
    ----------
    token_representations : torch.FloatTensor
        A padded tensor of shape (batch_size, seq_len, representation_dim),
        with the represenatations of the tokens.
    """
    mask = (torch.sum(token_representations, dim=-1) != 0).long()
    assert mask.dim() == 2
    return mask


def is_empty_metric(metric: Metric) -> bool:
    if isinstance(metric, CategoricalAccuracy):
        return metric.total_count == 0
    if isinstance(metric, F1Measure):
        return get_item(metric._true_positives +
                        metric._false_negatives) == 0
    if isinstance(metric, SpanBasedF1Measure):
        return (len(metric._true_positives.keys()) +
                len(metric._false_positives.keys()) +
                len(metric._false_negatives.keys())) == 0
    if isinstance(metric, Perplexity):
        return metric._total_num_instances == 0
    raise ValueError("metric {} not supported yet".format(type(metric)))


def get_item(value: Any):
    if hasattr(value, 'item'):
        val = value.item()
    else:
        val = value
    return val


def pad_contextualizer_output(seqs: List[torch.Tensor]):
    """
    Takes the output of a contextualizer, a list (of length batch_size)
    of Tensors with shape (seq_len, repr_dim), and produces a padded
    Tensor with these possibly-variable length items of shape
    (batch_size, seq_len, repr_dim)

    Returns
    -------
    padded_representations: torch.FloatTensor
        FloatTensor of shape (batch_size, seq_len, repr_dim) with 0 padding.
    mask: torch.FloatTensor
        A (batch_size, max_length) mask with 1's in positions without padding
        and 0's in positions with padding.
    """
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    mask = get_mask_from_sequence_lengths(seqs[0].new_tensor(lengths), max_len)
    return torch.stack(
        [torch.cat([s, s.new_zeros(max_len - len_, s.size(-1))], dim=0) for
         s, len_ in zip(seqs, lengths)]), mask
