from flaky import flaky
import numpy
import pytest
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model

from contexteval.common.model_test_case import ModelTestCase
from contexteval.models import PairwiseTagger  # noqa: F401


class TestSyntacticDependencyArcPrediction(ModelTestCase):
    def setUp(self):
        super(TestSyntacticDependencyArcPrediction, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'tasks' / 'syntactic_dependency_arc_prediction' / 'experiment.json',
                          self.FIXTURES_ROOT / 'data' / 'syntactic_dependency' / 'ptb.conllu')

    def test_pairwise_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_pairwise_tagger_with_contextualizer_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(
            self.FIXTURES_ROOT / 'tasks' / 'syntactic_dependency_arc_prediction' /
            'experiment_model_contextualizer.json')

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    @flaky
    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        output_dict = self.model.decode(output_dict)
        class_probs = output_dict['class_probabilities'][0].detach().numpy()
        num_examples = class_probs.shape[0]
        numpy.testing.assert_allclose(numpy.sum(class_probs, -1),
                                      numpy.ones(num_examples), rtol=1e-5)

    def test_mismatching_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # Make the decoder wrong - the input size should be 3072
        # since the combination concatenates 2 elmo embeddings and their
        # element-wise multiple (1024 each)
        params["model"]["decoder"]["input_size"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))

    def test_no_input_representations_throws_error(self):
        broken_param_file = (self.FIXTURES_ROOT / 'tasks' / 'syntactic_dependency_arc_prediction' /
                             'broken_experiments' / 'no_contextualizer.json')
        with pytest.raises(ConfigurationError):
            self.ensure_model_can_train_save_and_load(broken_param_file)
