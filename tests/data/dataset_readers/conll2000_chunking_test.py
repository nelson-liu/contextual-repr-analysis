from allennlp.common import Params
import numpy as np
from numpy.testing import assert_allclose
import pytest

from contexteval.common.custom_test_case import CustomTestCase
from contexteval.contextualizers import PrecomputedContextualizer
from contexteval.data.dataset_readers import Conll2000ChunkingDatasetReader


class TestConll2000ChunkingDatasetReader():
    data_path = CustomTestCase.FIXTURES_ROOT / "data" / "chunking" / "conll.txt"
    contextualizer_path = (CustomTestCase.FIXTURES_ROOT / "contextualizers" /
                           "precomputed_elmo" / "elmo_layers_all.hdf5")

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('use_contextualizer', (True, False))
    def test_read_from_file(self, lazy, use_contextualizer):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = Conll2000ChunkingDatasetReader(contextualizer=contextualizer,
                                                lazy=lazy)
        instances = list(reader.read(str(self.data_path)))

        # First read instance
        instance = instances[0]
        fields = instance.fields
        assert [token.metadata for token in fields["raw_tokens"].field_list] == [
            'Confidence', 'in', 'the', 'pound', 'is', 'widely', 'expected', 'to', 'take', 'another',
            'sharp', 'dive', 'if', 'trade', 'figures', 'for', 'September', ',', 'due', 'for', 'release',
            'tomorrow', ',', 'fail', 'to', 'show', 'a', 'substantial', 'improvement', 'from', 'July', 'and',
            'August', "'s", 'near-record', 'deficits', '.']
        assert fields["labels"].labels == [
            'B-NP', 'B-PP', 'B-NP', 'I-NP', 'B-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'B-NP', 'I-NP',
            'I-NP', 'B-SBAR', 'B-NP', 'I-NP', 'B-PP', 'B-NP', 'O', 'B-ADJP', 'B-PP', 'B-NP', 'B-NP', 'O',
            'B-VP', 'I-VP', 'I-VP', 'B-NP', 'I-NP', 'I-NP', 'B-PP', 'B-NP', 'I-NP', 'I-NP', 'B-NP', 'I-NP',
            'I-NP', 'O']
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[-1.258513, -1.0370141], [-0.05817706, -0.34756088],
                          [-0.06353955, -0.4938563], [-1.8024218, -1.0316596],
                          [-2.257492, -0.5222637], [-2.4755964, -0.24860916],
                          [-1.4937682, -1.3631285], [-1.5874765, 0.58332765],
                          [-0.6599875, -0.34025198], [-2.0129712, -1.7125161],
                          [-2.0061035, -2.0411587], [-2.111752, -0.17662084],
                          [-1.036485, -0.95351875], [-1.1027372, -0.8811481],
                          [-3.2971778, -0.80117923], [0.14612085, 0.2907345],
                          [-1.0681806, -0.11506036], [-0.89108264, -0.75120807],
                          [1.4598572, -1.5135024], [-0.19162387, -0.5925277],
                          [0.3152356, -0.67221195], [0.0429894, -0.3435017],
                          [-2.107685, 0.02174884], [-0.6821988, -1.6696682],
                          [-1.8384202, -0.22075021], [-1.033319, -1.1420834],
                          [-0.6265656, -0.8096429], [-1.0296414, -0.834536],
                          [-0.9962367, -0.09962708], [0.16024095, 0.43128008],
                          [-0.28929204, -1.4249148], [0.00278845, 0.6611263],
                          [0.50334555, -0.35937083], [1.147023, -0.6687972],
                          [0.77036375, -0.23009405], [-1.0500407, -0.02021815],
                          [-1.3865266, -0.85197794]]),
                rtol=1e-4)

        # Second read instance
        instance = instances[1]
        fields = instance.fields
        assert [token.metadata for token in fields["raw_tokens"].field_list] == [
            'Chancellor', 'of', 'the', 'Exchequer', 'Nigel', 'Lawson', "'s", 'restated', 'commitment',
            'to', 'a', 'firm', 'monetary', 'policy', 'has', 'helped', 'to', 'prevent', 'a', 'freefall',
            'in', 'sterling', 'over', 'the', 'past', 'week', '.']
        assert fields["labels"].labels == [
            'O', 'B-PP', 'B-NP', 'I-NP', 'B-NP', 'I-NP', 'B-NP', 'I-NP', 'I-NP', 'B-PP', 'B-NP', 'I-NP',
            'I-NP', 'I-NP', 'B-VP', 'I-VP', 'I-VP', 'I-VP', 'B-NP', 'I-NP', 'B-PP', 'B-NP', 'B-PP', 'B-NP',
            'I-NP', 'I-NP', 'O']
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[-2.4198189, -1.606727], [-1.2829566, -0.8627869],
                          [0.44851404, -0.8752346], [-1.2563871, -0.9349538],
                          [-2.3628764, 0.61601055], [-2.5294414, -0.8608694],
                          [-1.0940088, 0.36207741], [-1.3594072, -0.44920856],
                          [-2.1531758, -0.72061414], [-0.8710089, -0.01074989],
                          [1.1241767, 0.27293408], [-0.20302701, -0.3308825],
                          [-1.577058, -0.9223033], [-3.2015433, -1.4600563],
                          [-1.8444527, -0.3150784], [-1.4566939, -0.18184504],
                          [-2.097283, 0.02337693], [-1.4785317, 0.2928276],
                          [-0.47859374, -0.46162963], [-1.4853759, 0.30421454],
                          [0.25645372, -0.12769623], [-1.311865, -1.1461734],
                          [-0.75683033, -0.37533844], [-0.13498223, 1.1350582],
                          [0.3819366, 0.2941534], [-1.2304902, -0.67328024],
                          [-1.2757114, -0.43673947]]),
                rtol=1e-4)

    @pytest.mark.parametrize('max_instances', (1, 2, 0.5, 0.75, 1.0))
    @pytest.mark.parametrize('lazy', (True, False))
    def test_reproducible_with_and_without_contextualization(self, max_instances, lazy):
        uncontextualized_params = Params({
            "max_instances": max_instances,
            "lazy": lazy})
        uncontextualized_reader = Conll2000ChunkingDatasetReader.from_params(uncontextualized_params)
        uncontextualized_instances = list(uncontextualized_reader.read(str(self.data_path)))

        contextualized_params = Params({
            "lazy": lazy,
            "max_instances": max_instances,
            "contextualizer": {
                "type": "precomputed_contextualizer",
                "representations_path": self.contextualizer_path
            }})
        contextualized_reader = Conll2000ChunkingDatasetReader.from_params(contextualized_params)
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
        reader = Conll2000ChunkingDatasetReader(
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
