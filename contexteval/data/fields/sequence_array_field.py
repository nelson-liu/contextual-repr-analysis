from allennlp.data.fields import ArrayField, SequenceField
import numpy
from overrides import overrides


class SequenceArrayField(ArrayField, SequenceField):
    """
    A class representing an array with sequential semantics. The input sequence_dim
    represents the length of the sequence. The array could have arbitrary
    dimensions. A batch of these arrays are padded to the max dimension length in the
    batch for each dimension.
    """
    def __init__(self,
                 array: numpy.ndarray,
                 padding_value: int = 0,
                 sequence_dim: int = 0) -> None:
        self.array = array
        self.padding_value = padding_value
        self.sequence_dim = sequence_dim

    @overrides
    def sequence_length(self) -> int:
        return self.array.shape[self.sequence_dim]
