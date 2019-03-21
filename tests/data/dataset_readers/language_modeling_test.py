from allennlp.common import Params
import pytest

from contexteval.common.custom_test_case import CustomTestCase
from contexteval.contextualizers import PrecomputedContextualizer
from contexteval.data.dataset_readers import LanguageModelingDatasetReader


class TestLanguageModelingDatasetReader():
    data_path = CustomTestCase.FIXTURES_ROOT / "data" / "language_modeling" / "1b_benchmark.txt"
    contextualizer_path = (CustomTestCase.FIXTURES_ROOT / "contextualizers" /
                           "precomputed_elmo" / "elmo_layers_all.hdf5")

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('use_contextualizer', (True, False))
    @pytest.mark.parametrize('backward', (True, False))
    def test_read_from_file(self, lazy, use_contextualizer, backward):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = LanguageModelingDatasetReader(backward=backward,
                                               contextualizer=contextualizer,
                                               lazy=lazy)
        instances = list(reader.read(str(self.data_path)))

        # First read instance
        instance = instances[0]
        fields = instance.fields
        tokens = ['The', 'party', 'stopped', 'at', 'the', 'remains', 'of', 'a', '15th-century', 'khan', '--',
                  'a', 'roadside', 'inn', 'built', 'by', 'the', 'Mameluks', ',', 'former', 'slave', 'soldiers',
                  'turned', 'rulers', 'of', 'Egypt', 'and', 'Palestine', '.']
        if backward:
            assert [token.metadata for token in fields["raw_tokens"].field_list] == tokens
            assert fields["labels"].labels == ["<S>"] + tokens[:-1]
        else:
            assert [token.metadata for token in fields["raw_tokens"].field_list] == tokens
            assert fields["labels"].labels == tokens[1:] + ["</S>"]

        # Second read instance
        instance = instances[1]
        fields = instance.fields
        tokens = ['He', 'claimed', 'that', 'people', 'who', 'followed', 'his', 'cultivation', 'formula',
                  'acquired', 'a', '"', 'third', 'eye', '"', 'that', 'allowed', 'them', 'to', 'peer',
                  'into', 'other', 'dimensions', 'and', 'escape', 'the', 'molecular', 'world', '.']
        if backward:
            assert [token.metadata for token in fields["raw_tokens"].field_list] == tokens
            assert fields["labels"].labels == ["<S>"] + tokens[:-1]
        else:
            assert [token.metadata for token in fields["raw_tokens"].field_list] == tokens
            assert fields["labels"].labels == tokens[1:] + ["</S>"]

        # Third read instance
        instance = instances[2]
        fields = instance.fields
        tokens = ['If', 'they', 'were', 'losing', 'so', 'much', 'money', 'on', 'me', ',', 'why', 'would', 'they',
                  'send', 'me', 'a', 'new', 'credit', 'card', ',', 'under', 'the', 'same', 'terms', ',', 'when',
                  'my', 'old', 'one', 'expired', '?']
        if backward:
            assert [token.metadata for token in fields["raw_tokens"].field_list] == tokens
            assert fields["labels"].labels == ["<S>"] + tokens[:-1]
        else:
            assert [token.metadata for token in fields["raw_tokens"].field_list] == tokens
            assert fields["labels"].labels == tokens[1:] + ["</S>"]

        # Fourth read instance
        instance = instances[3]
        fields = instance.fields
        tokens = ['His', 'most', 'impressive', 'performance', 'was', 'his', 'last', '.']
        if backward:
            assert [token.metadata for token in fields["raw_tokens"].field_list] == tokens
            assert fields["labels"].labels == ["<S>"] + tokens[:-1]
        else:
            assert [token.metadata for token in fields["raw_tokens"].field_list] == tokens
            assert fields["labels"].labels == tokens[1:] + ["</S>"]

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('use_contextualizer', (True, False))
    @pytest.mark.parametrize('backward', (True, False))
    @pytest.mark.parametrize('max_length', (15, 25, 31, 50))
    def test_max_length(self, lazy, use_contextualizer, backward, max_length):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = LanguageModelingDatasetReader(max_length=max_length,
                                               backward=backward,
                                               contextualizer=contextualizer,
                                               lazy=lazy)
        instances = list(reader.read(str(self.data_path)))
        for instance in instances:
            fields = instance.fields
            assert len([token.metadata for token in fields["raw_tokens"].field_list]) <= max_length

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('max_instances', (1, 2, 0.5, 0.75, 1.0))
    @pytest.mark.parametrize('backward', (True, False))
    def test_reproducible_with_and_without_contextualization(self, lazy, max_instances, backward):
        uncontextualized_params = Params({
            "backward": backward,
            "max_instances": max_instances,
            "lazy": lazy})
        uncontextualized_reader = LanguageModelingDatasetReader.from_params(uncontextualized_params)
        uncontextualized_instances = list(uncontextualized_reader.read(str(self.data_path)))

        contextualized_params = Params({
            "backward": backward,
            "max_instances": max_instances,
            "lazy": lazy,
            "contextualizer": {
                "type": "precomputed_contextualizer",
                "representations_path": self.contextualizer_path
            }})
        contextualized_reader = LanguageModelingDatasetReader.from_params(contextualized_params)
        contextualized_instances = list(contextualized_reader.read(str(self.data_path)))
        # Assert they are the same
        for uncontextualized_instance, contextualized_instance in zip(uncontextualized_instances,
                                                                      contextualized_instances):
            assert ([token.metadata for token in uncontextualized_instance.fields["raw_tokens"].field_list] ==
                    [token.metadata for token in contextualized_instance.fields["raw_tokens"].field_list])
            assert (uncontextualized_instance.fields["labels"].labels ==
                    contextualized_instance.fields["labels"].labels)
            contextualized_extra_keys = list(set(contextualized_instance.fields.keys()) -
                                             set(uncontextualized_instance.fields.keys()))
            assert (set(contextualized_extra_keys) == set(["token_representations"]))

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('use_contextualizer', (True, False))
    @pytest.mark.parametrize('max_instances', (1, 2, 0.5, 0.75, 1.0))
    @pytest.mark.parametrize('backward', (True, False))
    def test_truncation(self, lazy, use_contextualizer, max_instances, backward):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = LanguageModelingDatasetReader(
            backward=backward,
            contextualizer=contextualizer,
            max_instances=max_instances,
            lazy=lazy)
        instances = list(reader.read(str(self.data_path)))
        num_total_instances = 4
        max_instances_to_num_instances = {
            int(1): 1,
            int(2): 2,
            0.5: int(num_total_instances * 0.5),
            0.75: int(num_total_instances * 0.75),
            1.0: num_total_instances}
        assert len(instances) == max_instances_to_num_instances[max_instances]
