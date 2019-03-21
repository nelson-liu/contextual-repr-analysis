import logging
from typing import Dict, List, Optional, Union

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.models.model import Model
from allennlp.models import load_archive
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import (
    combine_tensors,
    get_combined_dim,
    get_device_of,
    get_range_vector,
    sequence_cross_entropy_with_logits)
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

from contexteval.contextualizers import Contextualizer
from contexteval.common.util import (
    is_empty_metric,
    get_text_mask_from_representations,
    pad_contextualizer_output)

logger = logging.getLogger(__name__)


@Model.register("pairwise_tagger")
class PairwiseTagger(Model):
    """
    This ``PairwiseTagger`` takes two vectors as input and combines them in some way.
    The combined vector is then run through a configurable feed-forward layer to decode
    a label for the two inputs.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    token_representation_dim : ``int``
        The dimension of a single token representation.
    encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        An optional ``Seq2SeqEncoder`` to use to additionally pass the contextualized
        input through before feeding it into the decoder. If ``None``, a
        PassThroughEncoder is used.
    decoder : ``FeedForward`` or ``str``, optional (default=None)
        The decoder used to produce the labels from the combined inputs. If None or "linear",
        a linear model is built to map from the combination vector to the label space. If "mlp",
        a multilayer perceptron with 1024 hidden units and a ReLU activation is used.
    combination : ``str``, optional (default="x,y,x*y")
        If the two vectors are ``x`` and ``y``, we allow the following kinds of combinations: ``x``,
        ``y``, ``x*y``, ``x+y``, ``x-y``, ``x/y``, where each of those binary operations is performed
        elementwise.  You can list as many combinations as you want, comma separated.  For example, you
        might give ``x,y,x*y`` as the ``combination`` parameter to this class.
    contextualizer : ``Contextualizer``, optional (default=None)
        If token_representations are not provided by the dataset loader, we fall back
        to this contextualizer for calculating representations of the input tokens.
    calculate_per_label_f1 : ``bool``, optional (default=``False``)
        Calculate per-label f1 metrics during training. This is only necessary if you
        want to use a label's f1 score as the validation metric; to get per-label F1
        scores of test data, use the `error-analysis` command. This is recommended to
        be set to False when the label space is large, as it will significantly slow the model.
    loss_average: ``str``, optional (default=``"batch"``)
        When calculating the final loss, whether to average across the batch or across
        each individual instance. The former makes more sense for tagging tasks, whereas
        the latter makes more sense when a batch is semantically meaningless and just
        a way of grouping instances together. This must be one of ``"batch"`` or
        ``"token"``.
    pretrained_file: ``str``, optional (default=``None``)
        Path to a serialized model archive to transfer the contextualizer weights
        and/or encoder weights from. See also ``transfer_contextualizer_from_pretrained_file``
        and ``transfer_encoder_from_pretrained_file``. This key is automatically
        deleted upon serialization of the model.
    transfer_contextualizer_from_pretrained_file: ``bool``, optional (default=False)
        If ``pretrained_file`` is provided and this is True, we will attempt to
        load the contextualizer weights from the ``pretrained_file``.
    transfer_encoder_from_pretrained_file: ``bool``, optional (default=False)
        If ``pretrained_file`` is provided and this is True, we will attempt to
        load the encoder weights from the ``pretrained_file``.
    freeze_encoder: ``bool``, optional (default=False)
        If ``True``, the ``encoder`` is not updated during training
        (requires_grad=False). If ``False``, the ``encoder`` is updated during
        training (requires_grad=True).
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 token_representation_dim: int,
                 encoder: Optional[Seq2SeqEncoder] = None,
                 decoder: Optional[Union[FeedForward, str]] = None,
                 combination: str = 'x,y,x*y',
                 contextualizer: Optional[Contextualizer] = None,
                 calculate_per_label_f1: bool = False,
                 loss_average: str = "batch",
                 pretrained_file: Optional[str] = None,
                 transfer_contextualizer_from_pretrained_file: bool = False,
                 transfer_encoder_from_pretrained_file: bool = False,
                 freeze_encoder: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(PairwiseTagger, self).__init__(vocab, regularizer)

        self._num_classes = self.vocab.get_vocab_size("labels")
        self._token_representation_dim = token_representation_dim
        self._contextualizer = contextualizer
        if encoder is None:
            encoder = PassThroughEncoder(input_dim=self._token_representation_dim)
        self._encoder = encoder

        # Load the contextualizer and encoder weights from the
        # pretrained_file if applicable
        if pretrained_file:
            archive = None
            if self._contextualizer and transfer_contextualizer_from_pretrained_file:
                logger.info("Attempting to load contextualizer weights from "
                            "pretrained_file at {}".format(pretrained_file))
                archive = load_archive(cached_path(pretrained_file))
                contextualizer_state = archive.model._contextualizer.state_dict()
                contextualizer_layer_num = self._contextualizer._layer_num
                self._contextualizer.load_state_dict(contextualizer_state)
                if contextualizer_layer_num is not None:
                    logger.info("Setting layer num to {}".format(
                        contextualizer_layer_num))
                    self._contextualizer.set_layer_num(contextualizer_layer_num)
                else:
                    self._contextualizer.reset_layer_num()
                logger.info("Successfully loaded contextualizer weights!")
            if transfer_encoder_from_pretrained_file:
                logger.info("Attempting to load encoder weights from "
                            "pretrained_file at {}".format(pretrained_file))
                if archive is None:
                    archive = load_archive(cached_path(pretrained_file))
                encoder_state = archive.model._encoder.state_dict()
                self._encoder.load_state_dict(encoder_state)
                logger.info("Successfully loaded encoder weights!")

        self._freeze_encoder = freeze_encoder
        for parameter in self._encoder.parameters():
            # If freeze is true, requires_grad should be false and vice versa.
            parameter.requires_grad_(not self._freeze_encoder)

        self._combination = combination
        self._combined_input_dim = get_combined_dim(combination,
                                                    [self._encoder.get_output_dim(),
                                                     self._encoder.get_output_dim()])

        if decoder is None or decoder == "linear":
            # Create the default decoder (logistic regression) if it is not provided.
            decoder = FeedForward.from_params(Params(
                {"input_dim": self._combined_input_dim,
                 "num_layers": 1,
                 "hidden_dims": self._num_classes,
                 "activations": "linear"}))
            logger.info("No decoder provided to model, using default "
                        "decoder: {}".format(decoder))
        elif decoder == "mlp":
            # Create the MLP decoder
            decoder = FeedForward.from_params(Params(
                {"input_dim": self._combined_input_dim,
                 "num_layers": 2,
                 "hidden_dims": [1024, self._num_classes],
                 "activations": ["relu", "linear"]}))
            logger.info("Using MLP decoder: {}".format(decoder))
        self._decoder = decoder

        check_dimensions_match(self._token_representation_dim, self._encoder.get_input_dim(),
                               "token representation dim", "encoder input dim")
        check_dimensions_match(self._combined_input_dim, self._decoder.get_input_dim(),
                               "combined input dim", "decoder input dim")
        check_dimensions_match(self._decoder.get_output_dim(), self._num_classes,
                               "decoder output dim", "number of classes")

        if loss_average not in {"batch", "token"}:
            raise ConfigurationError("loss_average is {}, expected one of batch "
                                     "or token".format(loss_average))
        self.loss_average = loss_average
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
        logger.info("Applying initializer...")
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                arc_indices: torch.LongTensor,
                token_representations: torch.FloatTensor = None,
                raw_tokens: List[List[str]] = None,
                labels: torch.LongTensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        If ``token_representations`` is provided, ``tokens`` is not required. If
        ``token_representations`` is ``None``, then ``tokens`` is required.

        Parameters
        ----------
        arc_indices : torch.LongTensor
            A LongTensor of shape (batch_size, max_num_arcs, 2) with the token pairs
            to predict a label for for each element in the batch.
        token_representations : torch.FloatTensor, optional (default = None)
            A tensor of shape (batch_size, sequence_length, representation_dim) with
            the represenatation of the first token. If None, we use a contextualizer
            within this model to produce the token representation.
        raw_tokens : List[List[str]], optional (default = None)
            A batch of lists with the raw token strings. Used to compute
            token_representations, if either are None.
        labels : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels
            of shape ``(batch_size, num_arc_indices)``.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, max_num_arcs, num_classes)`` representing
            unnormalized log probabilities of the classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, max_num_arcs, num_classes)`` representing
            a distribution of the tag classes.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimized.
        """
        # Convert to LongTensor
        # TODO: add PR to ArrayField to preserve array types.
        arc_indices = arc_indices.long()
        if token_representations is None:
            if self._contextualizer is None:
                raise ConfigurationError(
                    "token_representations not provided as input to the model, and no "
                    "contextualizer was specified. Either add a contextualizer to your "
                    "dataset reader (preferred if your contextualizer is frozen) or to "
                    "this model (if you wish to train your contextualizer).")
            if raw_tokens is None:
                raise ValueError("Input raw_tokens is ``None`` --- make sure to set "
                                 "include_raw_tokens in the DatasetReader to True.")
            if arc_indices is None:
                raise ValueError("Did not recieve arc_indices as input, needed "
                                 "if the contextualizer is within the model.")
            # Convert contextualizer output into a tensor
            # Shape: (batch_size, max_seq_len, representation_dim)
            token_representations, _ = pad_contextualizer_output(
                self._contextualizer(raw_tokens))

        # Move token representations to the same device as the
        # module (CPU or CUDA). TODO(nfliu): This only works if the module
        # is on one device.
        device = next(self._decoder._linear_layers[0].parameters()).device
        token_representations = token_representations.to(device)
        text_mask = get_text_mask_from_representations(token_representations)
        text_mask = text_mask.to(device)
        label_mask = self._get_label_mask_from_arc_indices(arc_indices)
        label_mask = label_mask.to(device)

        # Encode the token representations
        encoded_token_representations = self._encoder(token_representations, text_mask)

        batch_size = arc_indices.size(0)

        # Index into the encoded_token_representations to get two tensors corresponding
        # to the children and parent of the arcs. Each of these tensors is of shape
        # (batch_size, num_arc_indices, representation_dim)
        first_arc_indices = arc_indices[:, :, 0]
        range_vector = get_range_vector(
            batch_size, get_device_of(first_arc_indices)).unsqueeze(1)
        first_token_representations = encoded_token_representations[
            range_vector, first_arc_indices]
        first_token_representations = first_token_representations.contiguous()

        second_arc_indices = arc_indices[:, :, 1]
        range_vector = get_range_vector(
            batch_size, get_device_of(second_arc_indices)).unsqueeze(1)
        second_token_representations = encoded_token_representations[
            range_vector, second_arc_indices]
        second_token_representations = second_token_representations.contiguous()

        # Take the batch and produce two tensors fit for combining
        # Shape: (batch_size, num_arc_indices, combined_representation_dim)
        combined_tensor = combine_tensors(
            self._combination,
            [first_token_representations, second_token_representations])

        # Decode out a label from the combined tensor.
        # Shape: (batch_size, num_arc_indices, num_classes)
        logits = self._decoder(combined_tensor)
        class_probabilities = F.softmax(logits, dim=-1)
        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
        if labels is not None:
            loss = sequence_cross_entropy_with_logits(logits, labels, label_mask,
                                                      average=self.loss_average)
            for name, metric in self.metrics.items():
                # When not running in error analysis mode, skip
                # metrics that start with "_"
                if not self.error_analysis and name.startswith("_"):
                    continue
                metric(logits, labels, label_mask.float())
            output_dict["loss"] = loss
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
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

    def _get_label_mask_from_arc_indices(self, arc_indices):
        """
        Extract the label mask given padded token indices.

        Parameters
        ----------
        arc_indices : torch.LongTensor, optional (default = None)
            A LongTensor of shape (batch_size, max_num_arcs, 2) with the token pairs
            to predict a label for for each element in the batch. Padding is denoted
            by [0, 0] elements.

        Returns
        -------
        mask : torch.LongTensor
            A mask of shape (batch_size, max_num_arcs) with 1 in positions where
            there is an arc to predict and 0 in positions with padding.
        """
        mask = (torch.sum(arc_indices, dim=-1) != 0).long()
        assert mask.dim() == 2
        return mask

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'PairwiseTagger':
        token_representation_dim = params.pop_int("token_representation_dim")

        encoder = params.pop("encoder", None)
        if encoder is not None:
            encoder = Seq2SeqEncoder.from_params(encoder)
        decoder = params.pop("decoder", None)
        if decoder is not None and not isinstance(decoder, str):
            decoder = FeedForward.from_params(decoder)
        combination = params.pop("combination", "x,y,x*y")
        contextualizer = params.pop('contextualizer', None)
        if contextualizer:
            contextualizer = Contextualizer.from_params(contextualizer)
        calculate_per_label_f1 = params.pop_bool("calculate_per_label_f1", False)
        loss_average = params.pop("loss_average", "batch")
        pretrained_file = params.pop("pretrained_file", None)
        transfer_contextualizer_from_pretrained_file = params.pop_bool(
            "transfer_contextualizer_from_pretrained_file", False)
        transfer_encoder_from_pretrained_file = params.pop_bool(
            "transfer_encoder_from_pretrained_file", False)
        freeze_encoder = params.pop_bool("freeze_encoder", False)

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab,
                   token_representation_dim=token_representation_dim,
                   encoder=encoder,
                   decoder=decoder,
                   combination=combination,
                   contextualizer=contextualizer,
                   calculate_per_label_f1=calculate_per_label_f1,
                   loss_average=loss_average,
                   pretrained_file=pretrained_file,
                   transfer_contextualizer_from_pretrained_file=transfer_contextualizer_from_pretrained_file,
                   transfer_encoder_from_pretrained_file=transfer_encoder_from_pretrained_file,
                   freeze_encoder=freeze_encoder,
                   initializer=initializer,
                   regularizer=regularizer)
