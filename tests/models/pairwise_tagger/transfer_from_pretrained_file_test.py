from flaky import flaky
import numpy
from numpy.testing import assert_allclose
from allennlp.common import Params

from contexteval.common.model_test_case import ModelTestCase
from contexteval.models import PairwiseTagger


class TestPairwiseTaggerTransferFromPretrainedFile(ModelTestCase):
    def setUp(self):
        super(TestPairwiseTaggerTransferFromPretrainedFile, self).setUp()

    @flaky
    def test_tagger_transfer_encoder(self):
        self.set_up_model(
            self.FIXTURES_ROOT / 'tasks' / 'syntactic_dependency_arc_classification' /
            'experiment_transfer_encoder_contextualizer.json',
            self.FIXTURES_ROOT / 'data' / 'syntactic_dependency' / 'ptb.conllu')
        old_model, _ = self.ensure_model_can_train_save_and_load(self.param_file)

        params_for_loading = Params({
            "token_representation_dim": 32,
            "encoder": {
                "type": "lstm",
                "input_size": 32,
                "hidden_size": 10,
                "bidirectional": True,
                "num_layers": 2
            },
            "decoder": "mlp",
            "pretrained_file": (self.TEST_DIR / "save_and_load_test" /
                                "model.tar.gz"),
            "transfer_encoder_from_pretrained_file": True,
            "contextualizer": {
                "type": "elmo_contextualizer",
                "batch_size": 80,
                "elmo": {
                    "weight_file": (self.FIXTURES_ROOT / "contextualizers" /
                                    "elmo" / "lm_weights.hdf5"),
                    "options_file": (self.FIXTURES_ROOT / "contextualizers" /
                                     "elmo" / "options.json"),
                    "requires_grad": True,
                    "num_output_representations": 1,
                    "dropout": 0.0
                }
            }
        })
        # Build the model from params
        new_model = PairwiseTagger.from_params(vocab=self.vocab, params=params_for_loading)

        # Compare the new_model and self.model and assert that encoder
        # parameters are the same, but contextualizer params differ.
        old_model_encoder_params = dict(old_model._encoder.named_parameters())
        new_model_encoder_params = dict(new_model._encoder.named_parameters())
        for key in old_model_encoder_params:
            # Scalar mix parameters will not match
            if "scalar_mix" in key:
                continue
            assert_allclose(old_model_encoder_params[key].detach().cpu().numpy(),
                            new_model_encoder_params[key].detach().cpu().numpy(),
                            err_msg=key)

        old_model_contextualizer_params = dict(
            old_model._contextualizer.named_parameters())
        new_model_contextualizer_params = dict(
            new_model._contextualizer.named_parameters())
        for key in old_model_contextualizer_params:
            old_model_param = old_model_contextualizer_params[key].detach().cpu().numpy()
            new_model_param = new_model_contextualizer_params[key].detach().cpu().numpy()
            if numpy.allclose(old_model_param, new_model_param):
                raise ValueError(
                    "old_model_contextualizer {} should not equal "
                    "new_model_contextualizer {}, but got {} and {}".format(
                        key, key, old_model_param, new_model_param))

    @flaky
    def test_tagger_transfer_contextualizer(self):
        self.set_up_model(
            self.FIXTURES_ROOT / 'tasks' / 'syntactic_dependency_arc_classification' /
            'experiment_transfer_encoder_contextualizer.json',
            self.FIXTURES_ROOT / 'data' / 'syntactic_dependency' / 'ptb.conllu')
        old_model, _ = self.ensure_model_can_train_save_and_load(self.param_file)

        params_for_loading = Params({
            "token_representation_dim": 32,
            "encoder": {
                "type": "lstm",
                "input_size": 32,
                "hidden_size": 10,
                "bidirectional": True,
                "num_layers": 2
            },
            "decoder": "mlp",
            "pretrained_file": (self.TEST_DIR / "save_and_load_test" /
                                "model.tar.gz"),
            "transfer_contextualizer_from_pretrained_file": True,
            "contextualizer": {
                "type": "elmo_contextualizer",
                "batch_size": 80,
                "elmo": {
                    "weight_file": (self.FIXTURES_ROOT / "contextualizers" /
                                    "elmo" / "lm_weights.hdf5"),
                    "options_file": (self.FIXTURES_ROOT / "contextualizers" /
                                     "elmo" / "options.json"),
                    "requires_grad": True,
                    "num_output_representations": 1,
                    "dropout": 0.0
                }
            }
        })
        # Build the model from params
        new_model = PairwiseTagger.from_params(vocab=self.vocab, params=params_for_loading)

        # Compare the new_model and self.model and assert that contextualizer
        # parameters are the same, but encoder params differ.
        old_model_contextualizer_params = dict(old_model._contextualizer.named_parameters())
        new_model_contextualizer_params = dict(new_model._contextualizer.named_parameters())
        for key in old_model_contextualizer_params:
            # Scalar mix parameters will not match
            if "scalar_mix" in key:
                continue
            assert_allclose(
                old_model_contextualizer_params[key].detach().cpu().numpy(),
                new_model_contextualizer_params[key].detach().cpu().numpy(),
                err_msg=key)

        old_model_encoder_params = dict(
            old_model._encoder.named_parameters())
        new_model_encoder_params = dict(
            new_model._encoder.named_parameters())
        for key in old_model_encoder_params:
            old_model_param = old_model_encoder_params[key].detach().cpu().numpy()
            new_model_param = new_model_encoder_params[key].detach().cpu().numpy()
            if numpy.allclose(old_model_param, new_model_param):
                raise ValueError(
                    "old_model_encoder {} should not equal "
                    "new_model_encoder {}, but got {} and {}".format(
                        key, key, old_model_param, new_model_param))

    @flaky
    def test_tagger_transfer_encoder_and_contextualizer(self):
        self.set_up_model(
            self.FIXTURES_ROOT / 'tasks' / 'syntactic_dependency_arc_classification' /
            'experiment_transfer_encoder_contextualizer.json',
            self.FIXTURES_ROOT / 'data' / 'syntactic_dependency' / 'ptb.conllu')
        old_model, _ = self.ensure_model_can_train_save_and_load(self.param_file)

        params_for_loading = Params({
            "token_representation_dim": 32,
            "encoder": {
                "type": "lstm",
                "input_size": 32,
                "hidden_size": 10,
                "bidirectional": True,
                "num_layers": 2
            },
            "decoder": "mlp",
            "pretrained_file": (self.TEST_DIR / "save_and_load_test" /
                                "model.tar.gz"),
            "transfer_contextualizer_from_pretrained_file": True,
            "transfer_encoder_from_pretrained_file": True,
            "freeze_encoder": True,
            "contextualizer": {
                "type": "elmo_contextualizer",
                "batch_size": 80,
                "elmo": {
                    "weight_file": (self.FIXTURES_ROOT / "contextualizers" /
                                    "elmo" / "lm_weights.hdf5"),
                    "options_file": (self.FIXTURES_ROOT / "contextualizers" /
                                     "elmo" / "options.json"),
                    "requires_grad": True,
                    "num_output_representations": 1,
                    "dropout": 0.0
                }
            }
        })
        # Build the model from params
        new_model = PairwiseTagger.from_params(vocab=self.vocab, params=params_for_loading)

        # Compare the new_model and self.model and assert that contextualizer
        # and the encoder parameters are the same.
        old_model_contextualizer_params = dict(old_model._contextualizer.named_parameters())
        new_model_contextualizer_params = dict(new_model._contextualizer.named_parameters())
        for key in old_model_contextualizer_params:
            # Scalar mix parameters will not match
            if "scalar_mix" in key:
                continue
            assert_allclose(
                old_model_contextualizer_params[key].detach().cpu().numpy(),
                new_model_contextualizer_params[key].detach().cpu().numpy(),
                err_msg=key)

        old_model_encoder_params = dict(old_model._encoder.named_parameters())
        new_model_encoder_params = dict(new_model._encoder.named_parameters())
        for key in old_model_encoder_params:
            # Scalar mix parameters will not match
            if "scalar_mix" in key:
                continue
            assert_allclose(
                old_model_encoder_params[key].detach().cpu().numpy(),
                new_model_encoder_params[key].detach().cpu().numpy(),
                err_msg=key)

        # Check that old_model encoder params are not frozen but new_model
        # encoder params are frozen.
        for name, parameter in old_model_encoder_params.items():
            assert parameter.requires_grad
        for name, parameter in new_model_encoder_params.items():
            assert not parameter.requires_grad
