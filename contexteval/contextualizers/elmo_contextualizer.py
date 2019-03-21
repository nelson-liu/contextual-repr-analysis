import logging
from typing import List

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of
from allennlp.modules.elmo import Elmo, batch_to_ids
import torch
from torch.nn import ParameterList, Parameter


from contexteval.contextualizers import Contextualizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Contextualizer.register("elmo_contextualizer")
class ElmoContextualizer(Contextualizer):
    """
    This contextualizer uses the ``allennlp.modules.Elmo`` module
    to contextualize input sequences. In particular, the representations
    are not computed ahead of time and a scalar mixtures of the layers is used.

    Parameters
    ----------
    elmo: Elmo
        The Elmo object to use in contextualization.
    batch_size: int
        The batch size to use for contextualization.
        this should equal the batch size of the model.
    layer_num: int, optional (default=``None``)
        If not None, we do not learn a scalar weighting. Instead,
        the scalar weights are set and frozen to an array where the
        weights for each layer is 0 and the weight for the ``layer_num``
        layer is 50. When this is not None, ``freeze_scalar_mix``
        is implicitly true.
    freeze_scalar_mix: bool, optional (default=``False``)
        If True, the scalar mix is frozen and not trained. This only
        really makes sense to use if the ElmoContextualizer weights are
        loaded from another model.
    first_half_only: bool, optional (default=``False``)
        Whether to return only the first half of the word representation.
        For example, for the original ELMo embeddings, this would return
        only the first 512 elements of each word vectors (vector[:512]).
        Since the backward components of the model will never be updated
        with this, we set requires_grad to false for the
        _elmo_lstm.backward_layer parameters.
    second_half_only: bool, optional (default=``False``)
        Whether to return only the second half of the word representation.
        For example, for the original ELMo embeddings, this would return
        only the last 512 elements of each word vectors (vector[512:]).
    """
    def __init__(self,
                 elmo: Elmo,
                 batch_size: int,
                 layer_num: int = None,
                 freeze_scalar_mix: bool = False,
                 first_half_only: bool = False,
                 second_half_only: bool = False) -> None:
        super().__init__()
        self._batch_size = batch_size
        if len(elmo._scalar_mixes) != 1:
            raise ConfigurationError(
                "Input ELMo module must only return 1 layer of "
                "representations, got one that returns {} layers".format(
                    len(elmo._scalar_mixes)))
        self._elmo = elmo
        self._layer_num = layer_num
        if self._layer_num is not None:
            self.set_layer_num(self._layer_num)

        self._freeze_scalar_mix = freeze_scalar_mix
        if freeze_scalar_mix or self._layer_num is not None:
            # Turn off grad if freeze_scalar_mix is True or we are
            # using a particular layer of ELMo.
            for name, scalar_mix_param in self._elmo._scalar_mixes[0].named_parameters():
                logger.info("Freezing {} parameter in scalar mix".format(name))
                scalar_mix_param.requires_grad_(False)

        self._first_half_only = first_half_only
        self._second_half_only = second_half_only
        if self._first_half_only is True and self._second_half_only is True:
            raise ConfigurationError("first_half_only and second_half_only "
                                     "cannot both be true.")
        if self._first_half_only:
            for parameter_name, parameter in self._elmo._elmo_lstm.named_parameters():
                if "backward_layer_" in parameter_name:
                    logger.info("first_half_only is True, freezing {}".format(parameter_name))
                    parameter.requires_grad_(False)
        if self._second_half_only:
            for parameter_name, parameter in self._elmo._elmo_lstm.named_parameters():
                if "forward_layer_" in parameter_name:
                    logger.info("second_half_only is True, freezing {}".format(parameter_name))
                    parameter.requires_grad_(False)

    def set_layer_num(self, layer_num: int):
        """
        Given an int referring to the layer of representations to use,
        set the scalar mix to this.
        """
        num_elmo_layers = self._elmo._elmo_lstm.num_layers
        scalar_mix_parameters = [Parameter(torch.FloatTensor([0.0]),
                                           requires_grad=False)
                                 for i in range(num_elmo_layers)]
        # Use indexing to set the 50.0 parameter to enable negative indices
        # to passed to the layer_num argument.
        scalar_mix_parameters[layer_num] = Parameter(torch.FloatTensor([50.0]),
                                                     requires_grad=False)
        scalar_mix_parameters = ParameterList(scalar_mix_parameters)
        self._elmo._scalar_mixes[0].scalar_parameters = scalar_mix_parameters
        self._elmo._scalar_mixes[0].gamma = Parameter(torch.FloatTensor([1.0]),
                                                      requires_grad=False)

    def reset_layer_num(self):
        num_elmo_layers = self._elmo._elmo_lstm.num_layers
        scalar_mix_parameters = [Parameter(torch.FloatTensor([0.0]))
                                 for i in range(num_elmo_layers)]
        scalar_mix_parameters = ParameterList(scalar_mix_parameters)
        self._elmo._scalar_mixes[0].scalar_parameters = scalar_mix_parameters
        self._elmo._scalar_mixes[0].gamma = Parameter(torch.FloatTensor([1.0]))

    def forward(self, sentences: List[List[str]]) -> torch.FloatTensor:
        """
        Parameters
        ----------
        sentences: List[List[str]]
            A batch of sentences. len(sentences) is the batch size, and each sentence
            itself is a list of strings (the constituent words). If the batch is padded,
            the expected padding token in the Python ``None``.

        Returns
        -------
        representations: List[FloatTensor]
            A list with the contextualized representations of all words in an
            input sentence. Each inner FloatTensor is of shape (seq_len, repr_dim),
            and an outer List is used to store the representations for each input
            sentence. **The individual FloatTensosrs should _not_ have any padding.**
        """
        all_input_representations = []

        # Take batch_size groups of the input sentences
        for batch_sentences in lazy_groups_of(iter(sentences), self._batch_size):
            # Remove the padding "None"s in the input, if they exist.
            unpadded_batch_sentences = [
                [word for word in batch_sentence if word is not None] for
                batch_sentence in batch_sentences]

            # Convert the batch of sentences to character IDs
            character_ids = batch_to_ids(unpadded_batch_sentences)
            device = next(self._elmo.parameters()).device
            character_ids = character_ids.to(device)

            # Contextualize the character IDs
            embeddings = self._elmo(character_ids)
            if len(embeddings["elmo_representations"]) != 1:
                raise ValueError("Output of ELMo returned more than "
                                 "1 layer of representations.")
            # Tensor of size (batch_size, max_seq_len, representation_dim)
            batch_representations = embeddings["elmo_representations"][0]
            # Lengths of each element in the batch
            batch_sequence_lengths = torch.sum(embeddings["mask"], dim=1)
            # Shave off the padding for API consistency / masking downstream
            for sequence_length, padded_representation in zip(
                    batch_sequence_lengths, batch_representations):
                unpadded_representation = padded_representation[:sequence_length]
                if self._first_half_only:
                    unpadded_representation = torch.chunk(
                        unpadded_representation, chunks=2, dim=-1)[0]
                elif self._second_half_only:
                    unpadded_representation = torch.chunk(
                        unpadded_representation, chunks=2, dim=-1)[1]
                all_input_representations.append(unpadded_representation)

        return all_input_representations
