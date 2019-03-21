from contexteval.common.model_test_case import ModelTestCase
from contexteval.models import PairwiseTagger  # noqa: F401


class TestConll2003NERWithLanguageModel(ModelTestCase):
    def setUp(self):
        super(TestConll2003NERWithLanguageModel, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'tasks' / 'ner' / 'experiment_with_language_model.json',
                          self.FIXTURES_ROOT / 'data' / 'ner' / 'conll2003.txt')

    def test_tagger_with_lm_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
