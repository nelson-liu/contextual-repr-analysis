from contexteval.common.model_test_case import ModelTestCase
from contexteval.models import PairwiseTagger  # noqa: F401


class TestConllXPosTaggingWithLanguageModel(ModelTestCase):
    def setUp(self):
        super(TestConllXPosTaggingWithLanguageModel, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'tasks' / 'conllx_pos_tagging' / 'experiment.json',
                          self.FIXTURES_ROOT / 'data' / 'pos' / 'wsj.conllx')

    def test_tagger_with_lm_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
