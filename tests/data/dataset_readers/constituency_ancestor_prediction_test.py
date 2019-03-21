from allennlp.common import Params
import numpy as np
from numpy.testing import assert_allclose
import pytest

from contexteval.common.custom_test_case import CustomTestCase
from contexteval.contextualizers import PrecomputedContextualizer
from contexteval.data.dataset_readers import ConstituencyAncestorPredictionDatasetReader


class TestConstituencyAncestorPredictionDatasetReader():
    data_path = CustomTestCase.FIXTURES_ROOT / "data" / "syntactic_constituency" / "wsj.txt"
    contextualizer_path = (CustomTestCase.FIXTURES_ROOT / "contextualizers" /
                           "precomputed_elmo" / "elmo_layers_all.hdf5")

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('use_contextualizer', (True, False))
    @pytest.mark.parametrize('ancestor', ('parent', 'grandparent', 'greatgrandparent'))
    def test_read_from_file(self, lazy, use_contextualizer, ancestor):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = ConstituencyAncestorPredictionDatasetReader(ancestor=ancestor,
                                                             contextualizer=contextualizer,
                                                             lazy=lazy)
        instances = list(reader.read(str(self.data_path)))

        # First read instance
        instance = instances[0]
        fields = instance.fields
        assert [token.metadata for token in fields["raw_tokens"].field_list] == [
            'In', 'an', 'Oct.', '19', 'review', 'of', '``', 'The', 'Misanthrope', "''", 'at', 'Chicago',
            "'s", 'Goodman', 'Theatre', '(', '``', 'Revitalized', 'Classics', 'Take', 'the', 'Stage',
            'in', 'Windy', 'City', ',', "''", 'Leisure', '&', 'Arts', ')', ',', 'the', 'role', 'of',
            'Celimene', ',', 'played', 'by', 'Kim', 'Cattrall', ',', 'was', 'mistakenly', 'attributed',
            'to', 'Christina', 'Haag', '.']
        assert len([token.metadata for token in fields["raw_tokens"].field_list]) == len(fields["labels"].labels)
        if ancestor == "parent":
            assert fields["labels"].labels == [
                'PP', 'NP', 'NP', 'NP', 'NP', 'PP', 'NP', 'NP', 'NP',
                'NP', 'PP', 'NP', 'NP', 'NP', 'NP', 'PRN', 'PRN', 'NP',
                'NP', 'VP', 'NP', 'NP', 'PP', 'NP', 'NP', 'PRN', 'PRN',
                'NP', 'NP', 'NP', 'PRN', 'S', 'NP', 'NP', 'PP', 'NP', 'NP', 'VP',
                'PP', 'NP', 'NP', 'NP', 'VP', 'ADVP', 'VP', 'PP', 'NP', 'NP', 'S']
        elif ancestor == "grandparent":
            assert fields["labels"].labels == [
                'S', 'NP', 'NP', 'NP', 'NP', 'NP', 'PP', 'NP', 'NP', 'PP', 'NP',
                'NP', 'NP', 'PP', 'PP', 'NP', 'NP', 'S', 'S', 'S', 'VP', 'VP',
                'VP', 'PP', 'PP', 'NP', 'NP', 'PRN', 'PRN', 'PRN', 'NP', 'None',
                'NP', 'NP', 'NP', 'PP', 'S', 'NP', 'VP', 'PP', 'PP', 'S', 'S',
                'VP', 'VP', 'VP', 'PP', 'PP', 'None']
        else:
            # ancestor is greatgrandparent
            assert fields["labels"].labels == [
                'None', 'PP', 'PP', 'PP', 'PP', 'PP', 'NP', 'PP', 'PP', 'NP', 'PP', 'PP',
                "PP", 'NP', 'NP', 'PP', 'PP', 'PRN', 'PRN', 'PRN', 'S', 'S',
                'S', 'VP', 'VP', 'PP', "PP", 'NP', 'NP', 'NP', 'PP', 'None', 'NP', 'NP', 'NP',
                'NP', 'None', 'S', 'NP', 'VP', 'VP', 'None', 'None', 'VP', 'S',
                'VP', 'VP', 'VP', 'None']
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[0.7541596, 0.36606207], [-0.3912218, 0.2728929],
                          [0.4532569, 0.59446496], [-0.034773, 0.6178972],
                          [0.05996126, -0.21075758], [-0.00675234, -0.19188942],
                          [-0.25371405, -0.98044276], [0.55180097, -1.3375797],
                          [-0.76439965, -0.8849516], [-0.1852389, -0.76670283],
                          [-0.6538293, -2.109323], [0.11706313, -0.14159685],
                          [-0.26565668, 0.08206904], [-1.0511935, -0.28469092],
                          [0.22915375, 0.2485466], [1.4214072, 0.02810444],
                          [0.7648947, -1.3637407], [-0.01231889, -0.02892348],
                          [-0.1330762, 0.0219465], [0.8961761, -1.2976432],
                          [0.83349395, -1.8242016], [0.15122458, -0.9597366],
                          [0.7570322, -0.73728824], [-0.04838032, -0.8663991],
                          [0.32632858, -0.5200325], [0.7823914, -1.020006],
                          [0.5874542, -1.020459], [-0.4918128, -0.85094],
                          [-0.24947, -0.20599724], [-1.4349735, 0.19630724],
                          [-0.49690107, -0.58586204], [0.06130999, -0.14850587],
                          [0.66610545, -0.06235093], [-0.29052478, 0.40215907],
                          [0.24728307, 0.23677489], [-0.05339833, 0.22958362],
                          [-0.44152835, -0.58153844], [0.4723678, -0.06656095],
                          [0.32210657, -0.03144099], [0.6663985, 0.39230958],
                          [0.57831913, 0.19480982], [-0.96823174, 0.00828598],
                          [-0.7640736, 0.00441009], [-0.5589211, 0.17509514],
                          [0.01523143, -0.7975017], [0.3268571, -0.1870772],
                          [1.4704096, 0.8472788], [0.23348817, -0.48313117],
                          [-0.57006484, -0.77375746]]),
                rtol=1e-3)

        # Second read instance
        instance = instances[1]
        fields = instance.fields
        assert [token.metadata for token in fields["raw_tokens"].field_list] == [
            'Ms.', 'Haag', 'plays', 'Elianti', '.']
        if ancestor == "parent":
            assert fields["labels"].labels == ['NP', 'NP', 'VP', 'NP', 'S']
        elif ancestor == "grandparent":
            assert fields["labels"].labels == ['S', 'S', 'S', 'VP', 'None']
        else:
            # ancestor is greatgrandparent
            assert fields["labels"].labels == ['None', 'None', 'None', 'S', 'None']
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[0.6757653, -0.80925614], [-1.9424553, -1.0854281],
                          [-0.09960067, 0.17525218], [0.09222834, -0.8534998],
                          [-0.66507375, -0.5633631]]),
                rtol=1e-3)

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('max_instances', (1, 2, 0.5, 0.75, 1.0))
    def test_reproducible_with_and_without_contextualization(self, lazy, max_instances):
        uncontextualized_params = Params({
            "max_instances": max_instances,
            "lazy": lazy})
        uncontextualized_reader = ConstituencyAncestorPredictionDatasetReader.from_params(uncontextualized_params)
        uncontextualized_instances = list(uncontextualized_reader.read(str(self.data_path)))

        contextualized_params = Params({
            "lazy": lazy,
            "max_instances": max_instances,
            "contextualizer": {
                "type": "precomputed_contextualizer",
                "representations_path": self.contextualizer_path
            }})
        contextualized_reader = ConstituencyAncestorPredictionDatasetReader.from_params(contextualized_params)
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
    @pytest.mark.parametrize('ancestor', ('parent', 'grandparent', 'greatgrandparent'))
    def test_truncation(self, lazy, use_contextualizer, max_instances, ancestor):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = ConstituencyAncestorPredictionDatasetReader(
            contextualizer=contextualizer,
            ancestor=ancestor,
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
