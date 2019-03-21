from flaky import flaky
import pytest
from allennlp.common.checks import ConfigurationError

from contexteval.common.model_test_case import ModelTestCase
from contexteval.models import PairwiseTagger  # noqa: F401


class TestEventFactualityRegression(ModelTestCase):
    def setUp(self):
        super(TestEventFactualityRegression, self).setUp()
        self.set_up_model(
            self.FIXTURES_ROOT / 'tasks' / 'event_factuality' / 'experiment.json',
            self.FIXTURES_ROOT / 'data' / 'event_factuality' / 'ithappened.json')

    def test_selective_regressor_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_selective_regressor_with_contextualizer_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(
            self.FIXTURES_ROOT / 'tasks' / 'event_factuality' /
            'experiment_model_contextualizer.json')

    def test_selective_regressor_mlp_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(
            self.FIXTURES_ROOT / 'tasks' / 'event_factuality' / 'experiment_mlp.json')

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_no_input_representations_throws_error(self):
        broken_param_file = (self.FIXTURES_ROOT / 'tasks' / 'event_factuality' /
                             'broken_experiments' / 'no_contextualizer.json')
        with pytest.raises(ConfigurationError):
            self.ensure_model_can_train_save_and_load(broken_param_file)
