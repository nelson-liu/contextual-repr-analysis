import logging
from typing import Dict, List, Optional, Union

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data import Vocabulary
from allennlp.modules import ConditionalRandomField, FeedForward, Seq2SeqEncoder, TimeDistributed
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.models.model import Model
from allennlp.models import load_archive
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, F1Measure, SpanBasedF1Measure

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from contexteval.common.util import (
    is_empty_metric,
    get_text_mask_from_representations,
    pad_contextualizer_output)
from contexteval.contextualizers import Contextualizer
from contexteval.training.metrics import Perplexity

logger = logging.getLogger(__name__)


@Model.register("tagger")
class Tagger(Model):
    """
    This ``Tagger`` takes a series of vectors (representations for tokens in a sequence)
    as input and predicts tags for each token.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    token_representation_dim : ``int``, required
        The dimension of a single token representation.
    encoder : ``Seq2SeqEncoder``, optional (default=``None``)
        An optional ``Seq2SeqEncoder`` to use to additionally pass the contextualized
        input through before feeding it into the decoder. If ``None``, a
        PassThroughEncoder is used.
    decoder : ``FeedForward`` or ``str``, optional (default=None)
        The decoder used to produce the labels from the combined inputs. If None or "linear",
        a linear model is built to map from the combination vector to the label space. If "mlp",
        a multilayer perceptron with 1024 hidden units and a ReLU activation is used.
    use_crf : ``bool``, optional (default=False)
        If True, a CRF is used on top of the decoder outputs.
    include_start_end_transitions : ``bool``, optional (default=``True``)
        Whether to include start and end transition parameters in the CRF.
        Ignored if ``use_crf`` is ``False``.
    constrain_crf_decoding : ``bool``, optional (default=``False``)
        If ``True``, the CRF is constrained at decoding time to
        produce valid sequences of tags. If this is ``True``, then
        ``label_encoding`` is required. Ignored if ``use_crf`` is ``False``.
    label_encoding : ``str``, optional (default=``None``)
        Label encoding to use when calculating span f1 and constraining
        the CRF at decoding time . Valid options are "BIO", "BIOUL", "IOB1".
        Required if ``calculate_span_f1`` or ``constrain_crf_decoding`` is true.
    contextualizer : ``Contextualizer``, optional (default=None)
        If token_representations are not provided by the dataset loader, we fall back
        to this contextualizer for calculating representations of the input tokens.
    calculate_per_label_f1 : ``bool``, optional (default=``False``)
        Calculate per-label f1 metrics during training. This is only necessary if you
        want to use a label's f1 score as the validation metric; to get per-label F1
        scores of test data, use the `error-analysis` command. This is recommended to
        be set to False when the label space is large, as it will significantly slow the model.
    calculate_span_f1 : ``bool``, optional (default=``False``)
        Calculate span-level f1 metrics during training.
    calculate_perplexity: ``bool``, optional (default=``False``)
        Calculate the perplexity during training.
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
                 use_crf: bool = False,
                 constrain_crf_decoding: bool = False,
                 include_start_end_transitions: bool = True,
                 label_encoding: Optional[str] = None,
                 contextualizer: Optional[Contextualizer] = None,
                 calculate_per_label_f1: bool = False,
                 calculate_span_f1: bool = False,
                 calculate_perplexity: bool = False,
                 loss_average: str = "batch",
                 pretrained_file: Optional[str] = None,
                 transfer_contextualizer_from_pretrained_file: bool = False,
                 transfer_encoder_from_pretrained_file: bool = False,
                 freeze_encoder: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(Tagger, self).__init__(vocab, regularizer)

        self._num_classes = self.vocab.get_vocab_size("labels")
        self._token_representation_dim = token_representation_dim
        self._contextualizer = contextualizer
        if encoder is None:
            encoder = PassThroughEncoder(input_dim=token_representation_dim)
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
                logger.info("contextualizer_layer_num {}".format(contextualizer_layer_num))
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

        if decoder is None or decoder == "linear":
            # Create the default decoder (logistic regression) if it is not provided.
            decoder = FeedForward.from_params(Params(
                {"input_dim": self._encoder.get_output_dim(),
                 "num_layers": 1,
                 "hidden_dims": self._num_classes,
                 "activations": "linear"}))
            logger.info("No decoder provided to model, using default "
                        "decoder: {}".format(decoder))
        elif decoder == "mlp":
            # Create the MLP decoder
            decoder = FeedForward.from_params(Params(
                {"input_dim": self._encoder.get_output_dim(),
                 "num_layers": 2,
                 "hidden_dims": [1024, self._num_classes],
                 "activations": ["relu", "linear"]}))
            logger.info("Using MLP decoder: {}".format(decoder))

        self._decoder = TimeDistributed(decoder)
        self._use_crf = use_crf
        self._constrain_crf_decoding = constrain_crf_decoding
        self._crf = None
        if use_crf:
            logger.info("Using CRF on top of decoder outputs")
            if constrain_crf_decoding:
                if label_encoding is None:
                    raise ConfigurationError(
                        "constrain_crf_decoding is True, but "
                        "label_encoding was not provided. label_encoding "
                        "must be provided.")
                logger.info("Constraining CRF decoding with label "
                            "encoding {}".format(label_encoding))
                labels = self.vocab.get_index_to_token_vocabulary("labels")
                constraints = allowed_transitions(label_encoding, labels)
            else:
                constraints = None
            self._crf = ConditionalRandomField(
                self._num_classes, constraints,
                include_start_end_transitions=include_start_end_transitions)

        check_dimensions_match(self._token_representation_dim, self._encoder.get_input_dim(),
                               "dimensionality of token representation", "encoder input dim")
        check_dimensions_match(self._encoder.get_output_dim(), self._decoder._module.get_input_dim(),
                               "encoder output dim", "decoder input dim")
        check_dimensions_match(self._decoder._module.get_output_dim(), self._num_classes,
                               "decoder output dim", "number of classes")
        if loss_average not in {"batch", "token"}:
            raise ConfigurationError("loss_average is {}, expected one of batch "
                                     "or token".format(loss_average))
        self.loss_average = loss_average
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3)
        }

        self.calculate_perplexity = calculate_perplexity
        if calculate_perplexity:
            self.metrics["perplexity"] = Perplexity()

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
        logger.info("Applying initializer...")
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                token_representations: torch.FloatTensor = None,
                raw_tokens: List[List[str]] = None,
                labels: torch.LongTensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        token_representations : torch.FloatTensor, optional (default = None)
            A padded tensor of shape (batch_size, seq_len, representation_dim),
            with the represenatations of the tokens. If None, we use a contextualizer within
            this model to produce the token representation.
        raw_tokens : List[List[str]], optional (default = None)
            A batch of lists with the raw token strings. Used to compute token_representations
            if it is None.
        labels : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels of shape
            ``(batch_size, num_tokens)``.

        Retpurns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        if token_representations is None:
            if self._contextualizer is None:
                raise ConfigurationError(
                    "token_representations not provided as input to the model, and no "
                    "contextualizer was specified. Either add a contextualizer to your "
                    "dataset reader (preferred if your contextualizer is frozen) or to "
                    "this model (if you wish to train your contextualizer).")
            if raw_tokens is None:
                raise ValueError("Input raw_tokens is ``None`` and token representations "
                                 "were not provided!")
            token_representations, mask = pad_contextualizer_output(
                self._contextualizer(raw_tokens))
            # Move token representations to the same device as the
            # module (CPU or CUDA). TODO(nfliu): This only works if the module
            # is on one device.
            device = next(self._decoder._module._linear_layers[0].parameters()).device
            token_representations = token_representations.to(device)
            mask = mask.to(device)
        else:
            mask = get_text_mask_from_representations(token_representations)

        batch_size, sequence_length = mask.size()

        # Encode the token representations.
        encoded_token_representations = self._encoder(token_representations, mask)

        logits = self._decoder(encoded_token_representations)

        output_dict = {}
        # Run CRF if provided and calculate class_probabilities
        if self._crf:
            best_paths = self._crf.viterbi_tags(logits, mask)
            # Just get the tags and ignore the score.
            predicted_tags = [x for x, y in best_paths]
            # Add tags to output dict
            output_dict["tags"] = predicted_tags
            # Get the class probabilities from the viterbi tags
            class_probabilities = logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1
        else:
            reshaped_log_probs = logits.view(-1, self._num_classes)
            class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
                [batch_size, sequence_length, self._num_classes])

        output_dict["logits"] = logits
        output_dict["mask"] = mask
        output_dict["class_probabilities"] = class_probabilities

        if labels is not None:
            if self._crf:
                # Add negative log-likelihood as loss
                log_likelihood = self._crf(logits, labels, mask)
                loss = -log_likelihood
            else:
                loss = sequence_cross_entropy_with_logits(logits, labels, mask,
                                                          average=self.loss_average)

            for name, metric in self.metrics.items():
                # When not running in error analysis mode, skip
                # metrics that start with "_"
                if not self.error_analysis and name.startswith("_"):
                    continue
                if name == "perplexity":
                    # Perplexity metric API is a bit different from the others.
                    metric(loss, mask.float().sum())
                else:
                    metric(class_probabilities, labels, mask.float())
            output_dict["loss"] = loss
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a ``"tags"`` key to the dictionary with the result.
        """
        if "tags" in output_dict:
            # Tags were already calculated from CRF during forward()
            output_dict["tags"] = [
                [self.vocab.get_token_from_index(tag, namespace="labels")
                 for tag in instance_tags]
                for instance_tags in output_dict["tags"]]
        else:
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
            if isinstance(metric, (CategoricalAccuracy, Perplexity)):
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

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'Tagger':
        token_representation_dim = params.pop_int("token_representation_dim")

        encoder = params.pop("encoder", None)
        if encoder is not None:
            encoder = Seq2SeqEncoder.from_params(encoder)
        decoder = params.pop("decoder", None)
        if decoder is not None and not isinstance(decoder, str):
            decoder = FeedForward.from_params(decoder)

        use_crf = params.pop_bool("use_crf", False)
        constrain_crf_decoding = params.pop_bool("constrain_crf_decoding", False)
        include_start_end_transitions = params.pop_bool("include_start_end_transitions", True)

        contextualizer = params.pop('contextualizer', None)
        if contextualizer:
            contextualizer = Contextualizer.from_params(contextualizer)
        calculate_per_label_f1 = params.pop_bool("calculate_per_label_f1", False)
        calculate_span_f1 = params.pop_bool("calculate_span_f1", False)
        calculate_perplexity = params.pop_bool("calculate_perplexity", False)
        loss_average = params.pop("loss_average", "batch")
        label_encoding = params.pop_choice("label_encoding", [None, "BIO", "BIOUL", "IOB1"],
                                           default_to_first_choice=True)

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
                   use_crf=use_crf,
                   constrain_crf_decoding=constrain_crf_decoding,
                   include_start_end_transitions=include_start_end_transitions,
                   label_encoding=label_encoding,
                   contextualizer=contextualizer,
                   calculate_per_label_f1=calculate_per_label_f1,
                   calculate_span_f1=calculate_span_f1,
                   calculate_perplexity=calculate_perplexity,
                   loss_average=loss_average,
                   pretrained_file=pretrained_file,
                   transfer_contextualizer_from_pretrained_file=transfer_contextualizer_from_pretrained_file,
                   transfer_encoder_from_pretrained_file=transfer_encoder_from_pretrained_file,
                   freeze_encoder=freeze_encoder,
                   initializer=initializer,
                   regularizer=regularizer)
