from flaky import flaky
import numpy
from numpy.testing import assert_allclose
from allennlp.common import Params

from contexteval.common.model_test_case import ModelTestCase
from contexteval.models import TaggerWithLM


class TestTaggerWithLanguageModelTransferFromPretrainedFile(ModelTestCase):
    def setUp(self):
        super(TestTaggerWithLanguageModelTransferFromPretrainedFile, self).setUp()

    @flaky
    def test_tagger_transfer_encoder(self):
        self.set_up_model(
            self.FIXTURES_ROOT / 'tasks' / 'ccg_supertagging' /
            'experiment_transfer_encoder_language_model.json',
            self.FIXTURES_ROOT / 'data' / 'ccg' / 'ccgbank.txt')
        old_model, _ = self.ensure_model_can_train_save_and_load(self.param_file)

        params_for_loading = Params({
            "language_model": {
                "options_file": "tests/fixtures/language_models/trained_gated_cnn/options.json",
                "weight_file": "tests/fixtures/language_models/trained_gated_cnn/model_state_epoch_0.th",
                "add_bos_eos": True
            },
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
            "transfer_encoder_from_pretrained_file": True
        })
        # Build the model from params
        new_model = TaggerWithLM.from_params(
            vocab=self.vocab, params=params_for_loading)

        # Compare the new_model and self.model and assert that encoder
        # parameters are the same, but LM params differ.
        old_model_encoder_params = dict(old_model._encoder.named_parameters())
        new_model_encoder_params = dict(new_model._encoder.named_parameters())
        for key in old_model_encoder_params:
            assert_allclose(old_model_encoder_params[key].detach().cpu().numpy(),
                            new_model_encoder_params[key].detach().cpu().numpy(),
                            err_msg=key)

        old_language_model_params = dict(old_model._language_model.named_parameters())
        new_language_model_params = dict(new_model._language_model.named_parameters())
        for key in old_language_model_params:
            old_model_param = old_language_model_params[key].detach().cpu().numpy()
            new_model_param = new_language_model_params[key].detach().cpu().numpy()
            if numpy.allclose(old_model_param, new_model_param):
                raise ValueError(
                    "old_language_model {} should not equal "
                    "new_language_model {}, but got {} and {}".format(
                        key, key, old_model_param, new_model_param))

    @flaky
    def test_tagger_transfer_language_model(self):
        self.set_up_model(
            self.FIXTURES_ROOT / 'tasks' / 'ccg_supertagging' /
            'experiment_transfer_encoder_language_model.json',
            self.FIXTURES_ROOT / 'data' / 'ccg' / 'ccgbank.txt')
        old_model, _ = self.ensure_model_can_train_save_and_load(self.param_file)

        params_for_loading = Params({
            "language_model": {
                "options_file": "tests/fixtures/language_models/trained_gated_cnn/options.json",
                "weight_file": "tests/fixtures/language_models/trained_gated_cnn/model_state_epoch_0.th",
                "add_bos_eos": True
            },
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
            "transfer_language_model_from_pretrained_file": True,
        })
        # Build the model from params
        new_model = TaggerWithLM.from_params(
            vocab=self.vocab, params=params_for_loading)

        # Compare the new_model and self.model and assert that LM
        # parameters are the same, but encoder params differ.
        old_language_model_params = dict(old_model._language_model.named_parameters())
        new_language_model_params = dict(new_model._language_model.named_parameters())
        for key in old_language_model_params:
            assert_allclose(
                old_language_model_params[key].detach().cpu().numpy(),
                new_language_model_params[key].detach().cpu().numpy(),
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
    def test_tagger_transfer_encoder_and_language_model(self):
        self.set_up_model(
            self.FIXTURES_ROOT / 'tasks' / 'ccg_supertagging' /
            'experiment_transfer_encoder_language_model.json',
            self.FIXTURES_ROOT / 'data' / 'ccg' / 'ccgbank.txt')
        old_model, _ = self.ensure_model_can_train_save_and_load(self.param_file)

        params_for_loading = Params({
            "language_model": {
                "options_file": "tests/fixtures/language_models/trained_gated_cnn/options.json",
                "weight_file": "tests/fixtures/language_models/trained_gated_cnn/model_state_epoch_0.th",
                "add_bos_eos": True
            },
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
            "transfer_language_model_from_pretrained_file": True,
            "transfer_encoder_from_pretrained_file": True
        })
        # Build the model from params
        new_model = TaggerWithLM.from_params(
            vocab=self.vocab, params=params_for_loading)

        # Compare the new_model and self.model and assert that LM
        # and the encoder parameters are the same.
        old_language_model_params = dict(old_model._language_model.named_parameters())
        new_language_model_params = dict(new_model._language_model.named_parameters())
        for key in new_language_model_params:
            assert_allclose(
                old_language_model_params[key].detach().cpu().numpy(),
                new_language_model_params[key].detach().cpu().numpy(),
                err_msg=key)

        old_model_encoder_params = dict(old_model._encoder.named_parameters())
        new_model_encoder_params = dict(new_model._encoder.named_parameters())
        for key in old_model_encoder_params:
            assert_allclose(
                old_model_encoder_params[key].detach().cpu().numpy(),
                new_model_encoder_params[key].detach().cpu().numpy(),
                err_msg=key)

    @flaky
    def test_tagger_transfer_and_freeze_language_model(self):
        self.set_up_model(
            self.FIXTURES_ROOT / 'tasks' / 'ccg_supertagging' /
            'experiment_transfer_encoder_language_model.json',
            self.FIXTURES_ROOT / 'data' / 'ccg' / 'ccgbank.txt')
        old_model, _ = self.ensure_model_can_train_save_and_load(self.param_file)

        params_for_loading = Params({
            "language_model": {
                "options_file": "tests/fixtures/language_models/trained_gated_cnn/options.json",
                "weight_file": "tests/fixtures/language_models/trained_gated_cnn/model_state_epoch_0.th",
                "add_bos_eos": True
            },
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
            "transfer_language_model_from_pretrained_file": True,
            "transfer_encoder_from_pretrained_file": True,
            "freeze_language_model": True,
            "freeze_encoder": True
        })
        # Build the model from params
        new_model = TaggerWithLM.from_params(
            vocab=self.vocab, params=params_for_loading)

        # Compare the new_model and self.model and assert that LM
        # and the encoder parameters are the same.

        old_language_model_params = dict(old_model._language_model.named_parameters())
        new_language_model_params = dict(new_model._language_model.named_parameters())
        for key in new_language_model_params:
            assert_allclose(
                old_language_model_params[key].detach().cpu().numpy(),
                new_language_model_params[key].detach().cpu().numpy(),
                err_msg=key)

        # Check that old_model LM params are unfrozen while
        # new_model LM params are frozen.
        for name, parameter in old_language_model_params.items():
            assert parameter.requires_grad
        for name, parameter in new_language_model_params.items():
            assert not parameter.requires_grad

        old_model_encoder_params = dict(old_model._encoder.named_parameters())
        new_model_encoder_params = dict(new_model._encoder.named_parameters())
        for key in old_model_encoder_params:
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
