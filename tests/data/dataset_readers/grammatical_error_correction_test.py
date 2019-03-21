from allennlp.common import Params
import numpy as np
from numpy.testing import assert_allclose
import pytest

from contexteval.common.custom_test_case import CustomTestCase
from contexteval.contextualizers import PrecomputedContextualizer
from contexteval.data.dataset_readers import GrammaticalErrorCorrectionDatasetReader


class TestGrammaticalErrorCorrectionDatasetReader():
    data_path = CustomTestCase.FIXTURES_ROOT / "data" / "grammatical_error_correction" / "fce.txt"
    contextualizer_path = (CustomTestCase.FIXTURES_ROOT / "contextualizers" /
                           "precomputed_elmo" / "elmo_layers_all.hdf5")

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('use_contextualizer', (True, False))
    def test_read_from_file(self, lazy, use_contextualizer):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = GrammaticalErrorCorrectionDatasetReader(contextualizer=contextualizer,
                                                         lazy=lazy)
        instances = list(reader.read(str(self.data_path)))

        # First read instance
        instance = instances[0]
        fields = instance.fields
        assert [token.metadata for token in fields["raw_tokens"].field_list] == [
            "Dear", "Sir", "or", "Madam", ","]
        assert fields["labels"].labels == ["c", "c", "c", "c", "c"]
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[-0.26053643, -1.629769], [-1.0137627, -1.9027274],
                          [-0.04409271, -1.1803466], [-0.34054118, -1.6507447],
                          [-1.5560683, -0.6016377]]),
                rtol=1e-4)

        # Second read instance
        instance = instances[1]
        fields = instance.fields
        assert [token.metadata for token in fields["raw_tokens"].field_list] == [
            "I", "am", "writing", "in", "order", "to", "express", "my", "disappointment",
            "about", "your", "musical", "show", "\"", "Over", "the", "Rainbow", "\"", "."]
        assert fields["labels"].labels == ["c", "c", "c", "c", "c", "c", "c", "c", "c", "i",
                                           "c", "c", "c", "c", "c", "c", "c", "c", "c"]
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[-2.7139964, -0.81360805], [-1.0573617, -1.0600157],
                          [-0.161348, -0.9779604], [0.22622275, -0.8058139],
                          [-0.7877749, -1.0645485], [-1.9962536, -0.37127006],
                          [-0.6058907, -1.7283063], [-0.7854868, -2.0817282],
                          [-1.1091563, -0.36297256], [-0.11240479, -0.6179019],
                          [-0.8821919, -1.1504251], [-2.120948, -1.2713698],
                          [-0.8990475, -0.7264623], [-0.5887449, -0.9071753],
                          [-1.8447106, -1.0307727], [0.18825394, -0.60070413],
                          [0.09527874, -0.04974258], [-0.32086655, -0.94218725],
                          [-0.23476452, -0.6441687]]),
                rtol=1e-4)

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('max_instances', (1, 2, 0.5, 0.75, 1.0))
    def test_reproducible_with_and_without_contextualization(self, lazy, max_instances):
        uncontextualized_params = Params({
            "max_instances": max_instances,
            "lazy": lazy})
        uncontextualized_reader = GrammaticalErrorCorrectionDatasetReader.from_params(uncontextualized_params)
        uncontextualized_instances = list(uncontextualized_reader.read(str(self.data_path)))

        contextualized_params = Params({
            "lazy": lazy,
            "max_instances": max_instances,
            "contextualizer": {
                "type": "precomputed_contextualizer",
                "representations_path": self.contextualizer_path
            }})
        contextualized_reader = GrammaticalErrorCorrectionDatasetReader.from_params(contextualized_params)
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
    def test_truncation(self, lazy, use_contextualizer, max_instances):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = GrammaticalErrorCorrectionDatasetReader(
            max_instances=max_instances,
            contextualizer=contextualizer,
            lazy=lazy)
        instances = list(reader.read(str(self.data_path)))
        num_total_instances = 2
        max_instances_to_num_instances = {
            int(1): 1,
            int(2): 2,
            0.5: int(num_total_instances * 0.5),
            0.75: int(num_total_instances * 0.75),
            1.0: num_total_instances}
        assert len(instances) == max_instances_to_num_instances[max_instances]
