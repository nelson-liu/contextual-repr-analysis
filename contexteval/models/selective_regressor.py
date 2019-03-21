import logging
from typing import Dict, List, Optional, Union

from overrides import overrides
import torch

from allennlp.common import Params
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder
from allennlp.modules.seq2seq_encoders import PassThroughEncoder
from allennlp.models.model import Model
from allennlp.models import load_archive
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_device_of, get_range_vector
from allennlp.training.metrics import MeanAbsoluteError, PearsonCorrelation

from contexteval.contextualizers import Contextualizer
from contexteval.common.util import (
    get_text_mask_from_representations,
    pad_contextualizer_output)

logger = logging.getLogger(__name__)


@Model.register("selective_regressor")
class SelectiveRegressor(Model):
    """
    The ``SelectiveRegressor`` takes a vector as input and runs it through a
    configurable feed-forward layer to predict a real-valued output.

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
    contextualizer : ``Contextualizer``, optional (default=None)
        If token_representations are not provided by the dataset loader, we fall back
        to this contextualizer for calculating representations of the input tokens.
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
                 contextualizer: Optional[Contextualizer] = None,
                 pretrained_file: Optional[str] = None,
                 transfer_contextualizer_from_pretrained_file: bool = False,
                 transfer_encoder_from_pretrained_file: bool = False,
                 freeze_encoder: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SelectiveRegressor, self).__init__(vocab, regularizer)

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

        if decoder is None or decoder == "linear":
            # Create the default decoder (logistic regression) if it is not provided.
            decoder = FeedForward.from_params(Params(
                {"input_dim": self._encoder.get_output_dim(),
                 "num_layers": 1,
                 "hidden_dims": 1,
                 "activations": "linear"}))
            logger.info("No decoder provided to model, using default "
                        "decoder: {}".format(decoder))
        elif decoder == "mlp":
            # Create the MLP decoder
            decoder = FeedForward.from_params(Params(
                {"input_dim": self._encoder.get_output_dim(),
                 "num_layers": 2,
                 "hidden_dims": [1024, 1],
                 "activations": ["relu", "linear"]}))
            logger.info("Using MLP decoder: {}".format(decoder))
        self._decoder = decoder

        check_dimensions_match(self._token_representation_dim, self._encoder.get_input_dim(),
                               "token representation dim", "encoder input dim")
        check_dimensions_match(self._encoder.get_output_dim(), self._decoder.get_input_dim(),
                               "encoder output dim", "decoder input dim")
        check_dimensions_match(self._decoder.get_output_dim(), 1,
                               "decoder output dim", "1, since we're predicting a real value")
        # SmoothL1Loss as described in "Neural Models of Factuality" (NAACL 2018)
        self.loss = torch.nn.SmoothL1Loss(reduction="none")
        self.metrics = {
            "mae": MeanAbsoluteError(),
            "pearson_r": PearsonCorrelation()
        }

        # Whether to run in error analysis mode or not, see commands.error_analysis
        self.error_analysis = False
        logger.info("Applying initializer...")
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                label_indices: torch.LongTensor,
                token_representations: torch.FloatTensor = None,
                raw_tokens: List[List[str]] = None,
                labels: torch.FloatTensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        If ``token_representations`` is provided, ``tokens`` is not required. If
        ``token_representations`` is ``None``, then ``tokens`` is required.

        Parameters
        ----------
        label_indices : torch.LongTensor
            A LongTensor of shape (batch_size, max_num_labels) with the tokens
            to predict a real-valued label for for each element (sentence) in the batch.
        token_representations : torch.FloatTensor, optional (default = None)
            A tensor of shape (batch_size, sequence_length, representation_dim) with
            the representation of the first token. If None, we use a contextualizer
            within this model to produce the token representation.
        raw_tokens : List[List[str]], optional (default = None)
            A batch of lists with the raw token strings. Used to compute
            token_representations, if either are None.
        labels : torch.FloatTensor, optional (default = None)
            A torch tensor representing the real-valued gold labels
            of shape ``(batch_size, num_label_indices)``.

        Returns
        -------
        An output dictionary consisting of:
        predictions : torch.FloatTensor
            A tensor of shape ``(batch_size, num_label_indices)`` representing
            the real-valued predictions for each label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimized.
        """
        # Convert to LongTensor
        label_indices = label_indices.long()
        if token_representations is None:
            if self._contextualizer is None:
                raise ConfigurationError(
                    "token_representation not provided as input to the model, and no "
                    "contextualizer was specified. Either add a contextualizer to your "
                    "dataset reader (preferred if your contextualizer is frozen) or to "
                    "this model (if you wish to train your contextualizer).")
            if raw_tokens is None:
                raise ValueError("Input raw_tokens is ``None`` --- make sure to set "
                                 "include_raw_tokens in the DatasetReader to True.")
            if label_indices is None:
                raise ValueError("Did not recieve any token indices, needed "
                                 "if the contextualizer is within the model.")
            # Convert contextualizer output into a tensor
            # Shape: (batch_size, max_seq_len, representation_dim)
            token_representations, _ = pad_contextualizer_output(
                self._contextualizer(raw_tokens))

        # Move token representation to the same device as the
        # module (CPU or CUDA). TODO(nfliu): This only works if the module
        # is on one device.
        device = next(self._decoder._linear_layers[0].parameters()).device
        token_representations = token_representations.to(device)
        text_mask = get_text_mask_from_representations(token_representations)
        text_mask = text_mask.to(device)
        label_mask = self._get_label_mask_from_label_indices(label_indices)
        label_mask = label_mask.to(device)

        # Mask out the -1 padding in the label_indices, since that doesn't
        # work with indexing. Note that we can't 0 pad because 0 is actually
        # a valid label index, so we pad with -1 just for the purposes of
        # proper mask calculation and then convert to 0-padding by applying
        # the mask.
        masked_label_indices = label_indices * label_mask

        # Encode the token representation.
        encoded_token_representations = self._encoder(token_representations,
                                                      text_mask)

        batch_size = masked_label_indices.size(0)
        # Index into the encoded_token_representations to get tensors corresponding
        # to the representations of the tokens to predict labels for.
        # Shape: (batch_size, num_label_indices, representation_dim)
        range_vector = get_range_vector(
            batch_size, get_device_of(masked_label_indices)).unsqueeze(1)
        selected_token_representations = encoded_token_representations[
            range_vector, masked_label_indices]
        selected_token_representations = selected_token_representations.contiguous()

        # Decode out a label from the token representation
        # Shape: (batch_size, num_label_indices, 1)
        predictions = self._decoder(selected_token_representations)
        # Shape: (batch_size, num_label_indices)
        predictions = predictions.squeeze(-1)
        output_dict = {"predictions": predictions}
        if labels is not None:
            # Calculate masked SmoothL1Loss
            # Shape: (batch_size, num_label_indices)
            per_prediction_loss = self.loss(predictions, labels)
            # Mask the per_prediction_loss and take the masked average
            masked_per_prediction_loss = per_prediction_loss * label_mask.float()
            total_masked_per_prediction_loss = masked_per_prediction_loss.sum()
            num_predictions = label_mask.float().sum()
            for name, metric in self.metrics.items():
                # When not running in error analysis mode, skip
                # metrics that start with "_"
                if not self.error_analysis and name.startswith("_"):
                    continue
                metric(predictions, labels, label_mask.float())
            output_dict["loss"] = total_masked_per_prediction_loss / num_predictions
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
            metrics_to_return[name] = metric.get_metric(reset)
        return metrics_to_return

    def _get_label_mask_from_label_indices(self, label_indices):
        """
        Extract the labels mask given padded token indices.

        Parameters
        ----------
        label_indices : torch.LongTensor, optional (default = None)
            A LongTensor of shape (batch_size, max_num_labels) with the token indices
            to predict a label for for each element in the batch. Padding is denoted
            by negative elements, since using 0 doesn't work (as 0 could be a
            valid element, referring to the 0th index).

        Returns
        -------
        mask : torch.LongTensor
            A mask of shape (batch_size, max_num_labels) with 1 in positions where
            there is label to predict and 0 in places with padding.
        """
        mask = (label_indices >= 0).long()
        assert mask.dim() == 2
        return mask

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SelectiveRegressor':
        token_representation_dim = params.pop_int("token_representation_dim")

        encoder = params.pop("encoder", None)
        if encoder is not None:
            encoder = Seq2SeqEncoder.from_params(encoder)
        decoder = params.pop("decoder", None)
        if decoder is not None and not isinstance(decoder, str):
            decoder = FeedForward.from_params(decoder)
        contextualizer = params.pop('contextualizer', None)
        if contextualizer:
            contextualizer = Contextualizer.from_params(contextualizer)

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
                   contextualizer=contextualizer,
                   pretrained_file=pretrained_file,
                   transfer_contextualizer_from_pretrained_file=transfer_contextualizer_from_pretrained_file,
                   transfer_encoder_from_pretrained_file=transfer_encoder_from_pretrained_file,
                   freeze_encoder=freeze_encoder,
                   initializer=initializer,
                   regularizer=regularizer)
