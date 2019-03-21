from contexteval.common.model_test_case import ModelTestCase
from contexteval.models import PairwiseTagger  # noqa: F401


class TestGrammaticalErrorCorrection(ModelTestCase):
    def setUp(self):
        super(TestGrammaticalErrorCorrection, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'tasks' / 'grammatical_error_correction' / 'experiment.json',
                          self.FIXTURES_ROOT / 'data' / 'grammatical_error_correction' / 'fce.txt')

    def test_tagger_with_lm_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
