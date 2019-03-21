from contexteval.common.model_test_case import ModelTestCase
from contexteval.models import PairwiseTagger  # noqa: F401


class TestSemanticTagging(ModelTestCase):
    def setUp(self):
        super(TestSemanticTagging, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / 'tasks' / 'semantic_tagging' /
                          'experiment_with_language_model.json',
                          self.FIXTURES_ROOT / 'data' / 'semantic_tagging' / 'semtag.txt')

    def test_tagger_with_lm_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
