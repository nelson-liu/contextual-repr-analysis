from allennlp.common import Params
from allennlp.data.dataset_readers import DatasetReader

from contexteval.common.custom_test_case import CustomTestCase


class TestDatasetReaderWithLM():
    def test_dataset_reader_with_lm(self):
        params = Params({
            "type": "datasetreader_with_lm",
            "subreader_type": "conll2003_ner",
            "token_indexers": {
                "tokens": {
                    "type": "single_id"
                },
                "elmo": {
                    "type": "elmo_characters"
                }
            }
        })

        reader_with_lm = DatasetReader.from_params(params)
        data_path = CustomTestCase.FIXTURES_ROOT / "data" / "ner" / "conll2003.txt"
        instances = reader_with_lm.read(data_path)
        assert 'lm_targets' in instances[0].fields
        assert 'tokens' in instances[0].fields
        assert hasattr(reader_with_lm, '_token_indexers')
        assert set(reader_with_lm._token_indexers.keys()) == {"tokens", "elmo"}
