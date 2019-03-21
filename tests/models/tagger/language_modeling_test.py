from flaky import flaky
import numpy
import pytest
from allennlp.common.checks import ConfigurationError

from contexteval.common.model_test_case import ModelTestCase
from contexteval.models import Tagger  # noqa: F401


class TestLanguageModel(ModelTestCase):
    def setUp(self):
        super(TestLanguageModel, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'tasks' / 'language_modeling' / 'experiment.json',
                          self.FIXTURES_ROOT / 'data' / 'language_modeling' / '1b_benchmark.txt')

    @flaky
    def test_simple_tagger_forward_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_simple_tagger_backward_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(
            self.FIXTURES_ROOT / 'tasks' / 'language_modeling' /
            'experiment_backward.json')

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    @flaky
    def test_simple_tagger_with_forward_contextualizer_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(
            self.FIXTURES_ROOT / 'tasks' / 'language_modeling' /
            'experiment_model_contextualizer.json')

    @pytest.mark.skip(reason="This test is overly flaky on CI")
    def test_simple_tagger_with_backward_contextualizer_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(
            self.FIXTURES_ROOT / 'tasks' / 'language_modeling' /
            'experiment_model_backward_contextualizer.json',
            tolerance=1e-3)

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        output_dict = self.model.decode(output_dict)
        class_probs = output_dict['class_probabilities'][0].detach().numpy()
        num_examples = class_probs.shape[0]
        numpy.testing.assert_allclose(numpy.sum(class_probs, -1),
                                      numpy.ones(num_examples), rtol=1e-5)

    def test_no_input_representations_throws_error(self):
        broken_param_file = (self.FIXTURES_ROOT / 'tasks' / 'language_modeling' /
                             'broken_experiments' / 'no_contextualizer.json')
        with pytest.raises(ConfigurationError):
            self.ensure_model_can_train_save_and_load(broken_param_file)
