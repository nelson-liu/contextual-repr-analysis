from allennlp.common import Params
import numpy as np
from numpy.testing import assert_allclose
import pytest

from contexteval.common.custom_test_case import CustomTestCase
from contexteval.contextualizers import PrecomputedContextualizer
from contexteval.data.dataset_readers import CcgSupertaggingDatasetReader


class TestCcgSupertaggingDatasetReader():
    data_path = CustomTestCase.FIXTURES_ROOT / "data" / "ccg" / "ccgbank.txt"
    contextualizer_path = (CustomTestCase.FIXTURES_ROOT / "contextualizers" /
                           "precomputed_elmo" / "elmo_layers_all.hdf5")

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('use_contextualizer', (True, False))
    def test_read_from_file(self, lazy, use_contextualizer):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = CcgSupertaggingDatasetReader(contextualizer=contextualizer,
                                              lazy=lazy)
        instances = list(reader.read(str(self.data_path)))
        # First read instance
        instance = instances[0]
        fields = instance.fields
        assert [token.metadata for token in fields["raw_tokens"].field_list] == [
            'Pierre', 'Vinken', ',', '61', 'years', 'old', ',', 'will', 'join', 'the', 'board', 'as',
            'a', 'nonexecutive', 'director', 'Nov.', '29', '.']
        assert fields["labels"].labels == [
            "N/N", "N", ",", "N/N", "N", r"(S[adj]\NP)\NP", ",", r"(S[dcl]\NP)/(S[b]\NP)",
            r"((S[b]\NP)/PP)/NP", r"NP[nb]/N", "N", r"PP/NP", r"NP[nb]/N", "N/N", "N",
            r"((S\NP)\(S\NP))/N[num]", "N[num]", "."]
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[-2.2218986, -2.451917], [-0.7551352, -0.7919447],
                          [-0.9525466, -0.985806], [-1.4567664, 0.1637534],
                          [0.21862003, -0.00878072], [-0.7341557, -0.57776076],
                          [-1.6816409, -1.2562131], [-0.9079286, 0.15607932],
                          [-0.44011104, -0.3434037], [0.56341827, -0.97181696],
                          [-0.7166206, -0.33435553], [-0.14051008, -1.260754],
                          [0.42426592, -0.35762805], [-1.0153385, -0.7949409],
                          [-0.7065723, 0.05164766], [-0.11002721, -0.11039695],
                          [0.41112524, 0.27260625], [-1.0369725, -0.6278316]]),
                rtol=1e-4)

        # Second read instance
        instance = instances[1]
        fields = instance.fields
        assert [token.metadata for token in fields["raw_tokens"].field_list] == [
            'Mr.', 'Vinken', 'is', 'chairman', 'of', 'Elsevier', 'N.V.', ',', 'the', 'Dutch',
            'publishing', 'group', '.']
        assert fields["labels"].labels == [
            'N/N', 'N', r'(S[dcl]\NP)/NP', 'N', r'(NP\NP)/NP', 'N/N', 'N', ',',
            'NP[nb]/N', 'N/N', 'N/N', 'N', '.']
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[0.7069745, -0.5422047], [-1.8885247, -1.4432149],
                          [-1.7570897, -1.1201282], [-1.2288755, -0.8003752],
                          [-0.08672556, -0.99020493], [-0.6416313, -1.147429],
                          [-0.7924329, 0.14809224], [-1.0645872, -1.0505759],
                          [0.69725895, -0.8735154], [0.27878952, -0.339666],
                          [0.20708983, -0.7103262], [-1.1115363, -0.16295972],
                          [-1.3495405, -0.8656957]]),
                rtol=1e-4)

    @pytest.mark.parametrize('max_instances', (1, 2, 0.5, 0.75, 1.0))
    @pytest.mark.parametrize('lazy', (True, False))
    def test_reproducible_with_and_without_contextualization(self, max_instances, lazy):
        uncontextualized_params = Params({
            "max_instances": max_instances,
            "lazy": lazy})
        uncontextualized_reader = CcgSupertaggingDatasetReader.from_params(uncontextualized_params)
        uncontextualized_instances = list(uncontextualized_reader.read(str(self.data_path)))

        contextualized_params = Params({
            "lazy": lazy,
            "max_instances": max_instances,
            "contextualizer": {
                "type": "precomputed_contextualizer",
                "representations_path": self.contextualizer_path
            }})
        contextualized_reader = CcgSupertaggingDatasetReader.from_params(contextualized_params)
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
        reader = CcgSupertaggingDatasetReader(
            contextualizer=contextualizer,
            max_instances=max_instances,
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
