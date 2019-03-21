from contexteval.common.model_test_case import ModelTestCase
from contexteval.models import PairwiseTagger  # noqa: F401


class TestConstituencyParentPrediction(ModelTestCase):
    def setUp(self):
        super(TestConstituencyParentPrediction, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'tasks' / 'constituency_ancestor_prediction' /
                          'experiment_with_language_model.json',
                          self.FIXTURES_ROOT / 'data' / 'syntactic_constituency' / 'wsj.txt')

    def test_tagger_with_lm_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
