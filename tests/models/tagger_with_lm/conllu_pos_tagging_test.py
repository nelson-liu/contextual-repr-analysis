from contexteval.common.model_test_case import ModelTestCase
from contexteval.models import PairwiseTagger  # noqa: F401


class TestConllUPosTagging(ModelTestCase):
    def setUp(self):
        super(TestConllUPosTagging, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'tasks' / 'conllu_pos_tagging' /
                          'experiment_with_language_model.json',
                          self.FIXTURES_ROOT / 'data' / 'pos' / 'en_ewt-ud.conllu')

    def test_simple_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
