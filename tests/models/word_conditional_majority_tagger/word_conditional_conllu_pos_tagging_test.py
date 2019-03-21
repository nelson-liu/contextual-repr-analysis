import numpy

from contexteval.common.model_test_case import ModelTestCase
from contexteval.models import WordConditionalMajorityTagger  # noqa: F401


class TestWordConditionalConllUPosTagging(ModelTestCase):
    def setUp(self):
        super(TestWordConditionalConllUPosTagging, self).setUp()
        self.set_up_model(
            (self.FIXTURES_ROOT / 'tasks' / 'conllu_pos_tagging' /
             'word_conditional_majority_experiment.json'),
            self.FIXTURES_ROOT / 'data' / 'pos' / 'en_ewt-ud.conllu')

    def test_word_conditional_majority_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent(self.param_file)

    def test_forward_pass_runs_correctly(self):
        # This only works because the model is not in eval mode.
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        output_dict = self.model.decode(output_dict)
        class_probs = output_dict['class_probabilities'][0].detach().numpy()
        num_examples = class_probs.shape[0]
        numpy.testing.assert_allclose(numpy.sum(class_probs, -1),
                                      numpy.ones(num_examples), rtol=1e-5)
