from allennlp.common import Params
import numpy as np
from numpy.testing import assert_allclose
import pytest

from contexteval.common.custom_test_case import CustomTestCase
from contexteval.contextualizers import PrecomputedContextualizer
from contexteval.data.dataset_readers import SemanticTaggingDatasetReader


class TestSemanticTaggingDatasetReader():
    data_path = CustomTestCase.FIXTURES_ROOT / "data" / "semantic_tagging" / "semtag.txt"
    contextualizer_path = (CustomTestCase.FIXTURES_ROOT / "contextualizers" /
                           "precomputed_elmo" / "elmo_layers_all.hdf5")

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('use_contextualizer', (True, False))
    def test_read_from_file(self, lazy, use_contextualizer):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = SemanticTaggingDatasetReader(contextualizer=contextualizer,
                                              lazy=lazy)
        instances = list(reader.read(str(self.data_path)))

        # First read instance
        instance = instances[0]
        fields = instance.fields
        assert [token.metadata for token in fields["raw_tokens"].field_list] == [
            "The", "report", "comes", "as", "Afghan", "officials", "announced", "that", "the",
            "bodies", "of", "20", "Taleban", "militants", "had", "been", "recovered", "from", "Bermel",
            "district", "in", "Paktika", "province", ",", "where", "NATO", "and", "Afghan", "forces",
            "recently", "conducted", "a", "mission", "."]
        assert fields["labels"].labels == [
            "DEF", "CON", "ENS", "SUB", "GPE", "CON", "PST", "SUB", "DEF", "CON", "REL", "DOM", "ORG",
            "CON", "EPT", "ETV", "EXV", "REL", "LOC", "CON", "REL", "LOC", "CON", "NIL", "PRO", "ORG",
            "AND", "GPE", "CON", "REL", "PST", "DIS", "CON", "NIL"]
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[0.2937507, -0.03459462], [-2.0376801, -1.6185987],
                          [0.80633676, 0.25174493], [-0.31453115, -0.16706648],
                          [0.2778436, 0.8754083], [-3.912532, 0.15716752],
                          [0.03259511, 1.074891], [0.60919964, 0.28122807],
                          [0.2766431, -0.57389474], [-1.5917854, 0.14402057],
                          [0.46617347, 0.5476148], [-0.3859496, 0.55521],
                          [-0.19902334, 0.51852816], [-0.49617743, 0.50021535],
                          [0.89773405, 0.33418086], [-1.0823509, 0.8463002],
                          [0.9214894, 0.17294498], [-0.98676234, 0.46858853],
                          [-1.1950549, 1.0456221], [-0.06810452, 1.8754647],
                          [-0.31319135, 0.5955827], [0.8572887, 0.9902405],
                          [0.18385345, 0.88080823], [-0.2386447, 0.273946],
                          [1.0159383, 0.2908004], [-0.84152496, -1.8987631],
                          [0.6318563, -1.3307623], [0.77291626, -0.9464708],
                          [-2.5105689, 0.05288363], [-1.8620715, 0.05540787],
                          [0.8963124, 0.88138795], [1.0833803, 0.29445225],
                          [-0.33804226, -0.5501779], [-0.80601907, -0.6653841]]),
                rtol=1e-4)

        # Second read instance
        instance = instances[1]
        fields = instance.fields
        assert [token.metadata for token in fields["raw_tokens"].field_list] == [
            "Turkish", "Prime", "Minister", "Tayyip", "Erdogan", ",", "in", "London", "for", "talks",
            "with", "British", "Prime", "Minister", "Tony", "Blair", ",", "said", "Wednesday",
            "Ankara", "would", "sign", "the", "EU", "protocol", "soon", "."]
        assert fields["labels"].labels == [
            "GPE", "UNK", "UNK", "PER", "PER", "NIL", "REL", "LOC", "REL", "CON", "REL", "GPE",
            "UNK", "UNK", "PER", "PER", "NIL", "PST", "TIM", "TIM", "FUT", "EXS", "DEF", "ORG", "CON",
            "REL", "NIL"]
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[0.28560728, 0.34812376], [-1.7316533, -0.5265728],
                          [-2.6642923, -0.9582914], [-1.6637948, -2.5388384],
                          [-2.9503021, -0.74373335], [-3.1062536, -0.47450644],
                          [-2.2821736, -0.08023855], [-1.9760342, -0.4066736],
                          [-1.9215266, -0.81184065], [-2.2942708, -0.13005577],
                          [-1.1666149, -0.82010025], [1.2843199, -0.04729652],
                          [-0.35602665, -1.9205997], [0.1594456, -2.390737],
                          [-1.0997499, -0.11030376], [-1.7266417, 0.01889065],
                          [-2.9103873, -1.6603167], [-1.3453144, 0.0276348],
                          [-1.5531495, 0.24530894], [-4.1084657, -0.24038172],
                          [-3.6353674, -1.2928469], [-1.527199, 1.9692067],
                          [-0.86209273, 1.5000844], [-1.3264929, 0.35947016],
                          [-2.4620879, 1.5387912], [-1.9274603, 0.67314804],
                          [-1.1620884, -0.63547856]]),
                rtol=1e-4)

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('max_instances', (1, 2, 0.5, 0.75, 1.0))
    def test_reproducible_with_and_without_contextualization(self, lazy, max_instances):
        uncontextualized_params = Params({
            "max_instances": max_instances,
            "lazy": lazy})
        uncontextualized_reader = SemanticTaggingDatasetReader.from_params(uncontextualized_params)
        uncontextualized_instances = list(uncontextualized_reader.read(str(self.data_path)))

        contextualized_params = Params({
            "lazy": lazy,
            "max_instances": max_instances,
            "contextualizer": {
                "type": "precomputed_contextualizer",
                "representations_path": self.contextualizer_path
            }})
        contextualized_reader = SemanticTaggingDatasetReader.from_params(contextualized_params)
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
        reader = SemanticTaggingDatasetReader(
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
