from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
import numpy as np
from numpy.testing import assert_allclose
import pytest
import torch

from contexteval.contextualizers import Contextualizer
from contexteval.common.custom_test_case import CustomTestCase
from contexteval.common.util import pad_contextualizer_output


class TestElmoContextualizer(CustomTestCase):
    model_paths = CustomTestCase.FIXTURES_ROOT / "contextualizers" / "elmo"
    sentence_1 = ("The cryptocurrency space is now figuring out to have "
                  "the highest search on Google globally .".split(" "))
    sentence_2 = "Bitcoin alone has a sixty percent share of global search .".split(" ")
    sentence_3 = "ELMo has had trouble handling sentences / with forward slashes .".split(" ")

    def test_elmo_contextualizer_normal(self):
        weights_path = self.model_paths / "lm_weights.hdf5"
        options_path = self.model_paths / "options.json"
        rep_dim = 32
        num_sentences = 3

        # Test the first layer (index 0)
        params = Params({
            "type": "elmo_contextualizer",
            "batch_size": 2,
            "elmo": {
                "options_file": options_path,
                "weight_file": weights_path,
                "dropout": 0.0,
                "num_output_representations": 1
            }
        })
        elmo_contextualizer = Contextualizer.from_params(params)
        representations = elmo_contextualizer([
            self.sentence_1, self.sentence_2, self.sentence_3])
        assert len(representations) == num_sentences

        first_sentence_representation = representations[0]
        seq_len = 16
        assert first_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            torch.sum(first_sentence_representation, dim=-1).detach().cpu().numpy(),
            np.array([-5.4501357, 0.57151437, -1.9986794, -1.9020741, -1.6883984,
                      0.46092677, -2.0832047, -2.045756, -2.660774, -5.4992304,
                      -3.6687968, -3.4485395, -1.9255438, -0.92559034, -1.7234659,
                      -4.93639]),
            rtol=1e-5)

        second_sentence_representation = representations[1]
        seq_len = 11
        assert second_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            torch.sum(second_sentence_representation, dim=-1).detach().cpu().numpy(),
            np.array([-0.51167095, -0.61811006, -2.8013024, -3.7508147, -1.6987357,
                      -1.1114583, -3.6302583, -3.3409853, -1.3613609, -3.6760461,
                      -5.137144]),
            rtol=1e-5)
        third_sentence_representation = representations[2]
        seq_len = 11
        assert third_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            torch.sum(third_sentence_representation, dim=-1).detach().cpu().numpy(),
            np.array([-1.5057361, -2.6824353, -4.1259403, -3.4485295, -1.3296673,
                      -4.5548496, -6.077871, -3.4515395, -3.8405519, -4.3518186,
                      -4.8782477]),
            rtol=1e-5)

    def test_elmo_contextualizer_raises_error_2_output_reps(self):
        weights_path = self.model_paths / "lm_weights.hdf5"
        options_path = self.model_paths / "options.json"

        params = Params({
            "type": "elmo_contextualizer",
            "batch_size": 2,
            "elmo": {
                "options_file": options_path,
                "weight_file": weights_path,
                "dropout": 0.0,
                "num_output_representations": 2
            }
        })
        with pytest.raises(ConfigurationError):
            Contextualizer.from_params(params)

    def test_elmo_contextualizer_with_grad(self):
        weights_path = self.model_paths / "lm_weights.hdf5"
        options_path = self.model_paths / "options.json"

        params = Params({
            "type": "elmo_contextualizer",
            "batch_size": 2,
            "elmo": {
                "options_file": options_path,
                "weight_file": weights_path,
                "dropout": 0.0,
                "num_output_representations": 1,
                "requires_grad": True,
            }
        })
        elmo_contextualizer = Contextualizer.from_params(params)
        unpadded_representations = elmo_contextualizer([
            self.sentence_1, self.sentence_2, self.sentence_3])
        token_representations, mask = pad_contextualizer_output(
            unpadded_representations)
        loss = token_representations.sum()
        loss.backward()
        elmo_grads = [param.grad for name, param in
                      elmo_contextualizer.named_parameters() if '_elmo_lstm' in name]
        assert all([grad is not None for grad in elmo_grads])

    def test_elmo_contextualizer_without_grad(self):
        weights_path = self.model_paths / "lm_weights.hdf5"
        options_path = self.model_paths / "options.json"

        params = Params({
            "type": "elmo_contextualizer",
            "batch_size": 2,
            "elmo": {
                "options_file": options_path,
                "weight_file": weights_path,
                "dropout": 0.0,
                "num_output_representations": 1,
                "requires_grad": False,
            }
        })
        elmo_contextualizer = Contextualizer.from_params(params)
        unpadded_representations = elmo_contextualizer([
            self.sentence_1, self.sentence_2, self.sentence_3])
        token_representations, mask = pad_contextualizer_output(
            unpadded_representations)
        loss = token_representations.sum()
        loss.backward()
        elmo_grads = [param.grad for name, param in
                      elmo_contextualizer.named_parameters() if '_elmo_lstm' in name]
        assert all([grad is None for grad in elmo_grads])

    def test_elmo_contextualizer_with_layer_num_and_grad(self):
        weights_path = self.model_paths / "lm_weights.hdf5"
        options_path = self.model_paths / "options.json"

        params = Params({
            "type": "elmo_contextualizer",
            "batch_size": 2,
            "layer_num": 1,
            "elmo": {
                "options_file": options_path,
                "weight_file": weights_path,
                "dropout": 0.0,
                "num_output_representations": 1,
                "requires_grad": True
            }
        })
        elmo_contextualizer = Contextualizer.from_params(params)
        unpadded_representations = elmo_contextualizer([
            self.sentence_1, self.sentence_2, self.sentence_3])
        token_representations, mask = pad_contextualizer_output(
            unpadded_representations)
        loss = token_representations.sum()
        loss.backward()
        for name, param in elmo_contextualizer.named_parameters():
            if "scalar_mix" in name:
                assert param.grad is None, "Parameter {} should not have grad.".format(name)
            else:
                assert param.grad is not None, "Parameter {} should have grad.".format(name)

    def test_elmo_contextualizer_with_layer_num_and_without_grad(self):
        weights_path = self.model_paths / "lm_weights.hdf5"
        options_path = self.model_paths / "options.json"

        params = Params({
            "type": "elmo_contextualizer",
            "batch_size": 2,
            "layer_num": 1,
            "elmo": {
                "options_file": options_path,
                "weight_file": weights_path,
                "dropout": 0.0,
                "num_output_representations": 1,
                "requires_grad": False
            }
        })
        elmo_contextualizer = Contextualizer.from_params(params)
        unpadded_representations = elmo_contextualizer([
            self.sentence_1, self.sentence_2, self.sentence_3])
        token_representations, mask = pad_contextualizer_output(
            unpadded_representations)
        loss = token_representations.sum()
        # Nothing in the contextualizer is requires_grad=True, so this
        # should be requires_grad=False and grad_fn should be None
        assert loss.grad_fn is None
        assert loss.requires_grad is False

    def test_elmo_contextualizer_with_grad_frozen_scalar_mix(self):
        weights_path = self.model_paths / "lm_weights.hdf5"
        options_path = self.model_paths / "options.json"

        params = Params({
            "type": "elmo_contextualizer",
            "batch_size": 2,
            "layer_num": 1,
            "freeze_scalar_mix": True,
            "elmo": {
                "options_file": options_path,
                "weight_file": weights_path,
                "dropout": 0.0,
                "num_output_representations": 1,
                "requires_grad": True,
            }
        })
        elmo_contextualizer = Contextualizer.from_params(params)
        unpadded_representations = elmo_contextualizer([
            self.sentence_1, self.sentence_2, self.sentence_3])
        token_representations, mask = pad_contextualizer_output(
            unpadded_representations)
        loss = token_representations.sum()
        loss.backward()
        for name, param in elmo_contextualizer.named_parameters():
            if "scalar_mix" in name:
                assert param.grad is None, "Parameter {} should not have grad.".format(name)
            else:
                assert param.grad is not None, "Parameter {} should have grad.".format(name)

    def test_elmo_contextualizer_without_grad_frozen_scalar_mix(self):
        weights_path = self.model_paths / "lm_weights.hdf5"
        options_path = self.model_paths / "options.json"

        params = Params({
            "type": "elmo_contextualizer",
            "batch_size": 2,
            "layer_num": 1,
            "freeze_scalar_mix": True,
            "elmo": {
                "options_file": options_path,
                "weight_file": weights_path,
                "dropout": 0.0,
                "num_output_representations": 1,
                "requires_grad": False,
            }
        })
        elmo_contextualizer = Contextualizer.from_params(params)
        unpadded_representations = elmo_contextualizer([
            self.sentence_1, self.sentence_2, self.sentence_3])
        token_representations, mask = pad_contextualizer_output(
            unpadded_representations)
        loss = token_representations.sum()
        # Nothing in the contextualizer is requires_grad=True, so this
        # should be requires_grad=False and grad_fn should be None
        assert loss.grad_fn is None
        assert loss.requires_grad is False
