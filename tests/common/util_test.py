import numpy as np
from numpy.testing import assert_allclose
import torch

from contexteval.common.custom_test_case import CustomTestCase
from contexteval.common.util import (
    get_text_mask_from_representations,
    pad_contextualizer_output)


class TestUtil(CustomTestCase):
    def test_get_text_mask_from_representations(self):
        token_representations = torch.FloatTensor([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0, 0], [0, 0]],
            [[0.1, 0.2], [0.3, 0.4], [0, 0], [0, 0], [0, 0]],
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.1, 0.1], [0.1, 0.1]],
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.2, 0.3], [0, 0]]])
        mask = get_text_mask_from_representations(token_representations)
        assert_allclose(mask.cpu().numpy(), np.array([[1, 1, 1, 0, 0],
                                                      [1, 1, 0, 0, 0],
                                                      [1, 1, 1, 1, 1],
                                                      [1, 1, 1, 1, 0]]))

    def test_pad_contextualizer_output(self):
        contextualizer_output = [
            torch.Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            torch.Tensor([[0.1, 0.2], [0.3, 0.4]]),
            torch.Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]),
            torch.Tensor([[0.1, 0.2]])
        ]
        padded_output, mask = pad_contextualizer_output(contextualizer_output)
        assert_allclose(padded_output.cpu().numpy(),
                        np.array([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0, 0]],
                                  [[0.1, 0.2], [0.3, 0.4], [0, 0], [0, 0]],
                                  [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
                                  [[0.1, 0.2], [0, 0], [0, 0], [0, 0]]]))
        assert_allclose(mask.cpu().numpy(), np.array([[1, 1, 1, 0],
                                                      [1, 1, 0, 0],
                                                      [1, 1, 1, 1],
                                                      [1, 0, 0, 0]]))
