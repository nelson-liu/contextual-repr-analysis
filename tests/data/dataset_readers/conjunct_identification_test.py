from allennlp.common import Params
import numpy as np
from numpy.testing import assert_allclose
import pytest

from contexteval.common.custom_test_case import CustomTestCase
from contexteval.contextualizers import PrecomputedContextualizer
from contexteval.data.dataset_readers import ConjunctIdentificationDatasetReader


class TestConjunctIdentificationDatasetReader():
    data_path = CustomTestCase.FIXTURES_ROOT / "data" / "coordination_boundary" / "conjunct_id.tsv"
    contextualizer_path = (CustomTestCase.FIXTURES_ROOT / "contextualizers" /
                           "precomputed_elmo" / "elmo_layers_all.hdf5")

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('use_contextualizer', (True, False))
    def test_read_from_file(self, lazy, use_contextualizer):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = ConjunctIdentificationDatasetReader(contextualizer=contextualizer,
                                                     lazy=lazy)
        instances = list(reader.read(str(self.data_path)))
        # One instance is skipped because of nested coordination
        assert len(instances) == 2

        # First read instance
        instance = instances[0]
        fields = instance.fields
        assert [token.metadata for token in fields["raw_tokens"].field_list] == [
            'They', 'shredded', 'it', 'simply', 'because', 'it', 'contained', 'financial',
            'information', 'about', 'their', 'creditors', 'and', 'depositors', '.', "''"]
        assert fields["labels"].labels == [
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'B', 'O', 'O']
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[-2.047799, -0.3107947], [0.40388513, 0.15957603],
                          [0.4006851, -0.1980469], [0.409753, -0.48708656],
                          [0.65417755, -0.03706935], [-0.53143466, -1.057557],
                          [0.7815078, -0.21813926], [-1.3369036, -0.77031285],
                          [0.11985331, -0.39474356], [0.68627775, -0.72502434],
                          [0.569624, -2.3243494], [-0.69559455, -1.248917],
                          [0.2524291, -0.47938287], [0.2019696, -0.66839015],
                          [-0.5914014, -0.8587656],
                          [-0.521717, 0.04716678]]),
                rtol=1e-4)

        # Second read instance
        instance = instances[1]
        fields = instance.fields
        assert [token.metadata for token in fields["raw_tokens"].field_list] == [
            'Suppression', 'of', 'the', 'book', ',', 'Judge', 'Oakes', 'observed',
            ',', 'would', 'operate', 'as', 'a', 'prior', 'restraint', 'and', 'thus',
            'involve', 'the', 'First', 'Amendment', '.']
        assert fields["labels"].labels == [
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I',
            'I', 'I', 'O', 'O', 'B', 'I', 'I', 'I', 'O']
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[-0.00731754, 0.00914195], [0.6760086, -0.11198741],
                          [0.8685149, 0.15874714], [-0.9228251, -1.3684492],
                          [-0.17535079, 0.36266953], [-0.85589266, -1.4212742],
                          [-1.8647766, -0.9377552], [-0.34395775, 0.18579313],
                          [-1.6104316, 0.5044512], [-1.6913524, 0.5832756],
                          [0.6513059, 1.1528094], [-0.24509574, 0.49362227],
                          [-0.47929475, 0.6173321], [-0.431388, 0.15780556],
                          [-1.4048593, 0.44075668], [-0.32530123, 0.23048985],
                          [-0.23973304, 1.2190828], [0.4657239, 0.20590879],
                          [0.16104633, 0.04873788], [0.8202704, -0.7126241],
                          [-0.59338295, 1.2020597], [-0.5741635, -0.05905316]]),
                rtol=1e-4)

    @pytest.mark.parametrize('max_instances', (1, 2, 0.5, 0.75, 1.0))
    @pytest.mark.parametrize('lazy', (True, False))
    def test_reproducible_with_and_without_contextualization(self, max_instances, lazy):
        uncontextualized_params = Params({
            "max_instances": max_instances,
            "lazy": lazy})
        uncontextualized_reader = ConjunctIdentificationDatasetReader.from_params(uncontextualized_params)
        uncontextualized_instances = list(uncontextualized_reader.read(str(self.data_path)))

        contextualized_params = Params({
            "max_instances": max_instances,
            "lazy": lazy,
            "contextualizer": {
                "type": "precomputed_contextualizer",
                "representations_path": self.contextualizer_path
            }})
        contextualized_reader = ConjunctIdentificationDatasetReader.from_params(contextualized_params)
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
        reader = ConjunctIdentificationDatasetReader(
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
