from flaky import flaky
import numpy

from contexteval.common.model_test_case import ModelTestCase
from contexteval.models import PairwiseTagger  # noqa: F401


class TestSemanticDependencyArcPredictionWithLanguageModel(ModelTestCase):
    def setUp(self):
        super(TestSemanticDependencyArcPredictionWithLanguageModel,
              self).setUp()
        self.set_up_model(
            self.FIXTURES_ROOT / 'tasks' / 'semantic_dependency_arc_prediction' /
            'experiment_with_language_model.json',
            self.FIXTURES_ROOT / 'data' / 'semantic_dependency' / 'dm.sdp')

    def test_pairwise_tagger_with_language_model_can_train_save_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        output_dict = self.model.decode(output_dict)
        class_probs = output_dict['class_probabilities'][0].detach().numpy()
        num_examples = class_probs.shape[0]
        numpy.testing.assert_allclose(numpy.sum(class_probs, -1),
                                      numpy.ones(num_examples), rtol=1e-5)
