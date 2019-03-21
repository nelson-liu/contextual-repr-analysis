from allennlp.common import Params
import numpy as np
from numpy.testing import assert_allclose
import pytest

from contexteval.common.custom_test_case import CustomTestCase
from contexteval.contextualizers import PrecomputedContextualizer
from contexteval.data.dataset_readers import SyntacticDependencyArcClassificationDatasetReader


class TestSyntacticDependencyArcClassificationDatasetReader():
    data_path = CustomTestCase.FIXTURES_ROOT / "data" / "syntactic_dependency" / "ptb.conllu"
    contextualizer_path = (CustomTestCase.FIXTURES_ROOT / "contextualizers" /
                           "precomputed_elmo" / "elmo_layers_all.hdf5")

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('use_contextualizer', (True, False))
    @pytest.mark.parametrize('include_raw_tokens', (True, False))
    def test_read_from_file(self, lazy, use_contextualizer, include_raw_tokens):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = SyntacticDependencyArcClassificationDatasetReader(
            contextualizer=contextualizer,
            include_raw_tokens=include_raw_tokens,
            lazy=lazy)
        instances = list(reader.read(str(self.data_path)))
        assert len(instances) == 2

        first_sentence = [
            'In', 'an', 'Oct.', '19', 'review', 'of', '``', 'The', 'Misanthrope', "''", 'at', 'Chicago',
            "'s", 'Goodman', 'Theatre', '(', '``', 'Revitalized', 'Classics', 'Take', 'the', 'Stage',
            'in', 'Windy', 'City', ',', "''", 'Leisure', '&', 'Arts', ')', ',', 'the', 'role', 'of',
            'Celimene', ',', 'played', 'by', 'Kim', 'Cattrall', ',', 'was', 'mistakenly', 'attributed',
            'to', 'Christina', 'Haag', '.']
        # First read instance
        instance = instances[0]
        fields = instance.fields
        if include_raw_tokens:
            assert [token.metadata for token in fields["raw_tokens"].field_list] == first_sentence
        assert_allclose(fields["arc_indices"].array, np.array([
            (0, 4), (1, 4), (2, 4), (3, 4), (4, 44), (5, 8), (6, 8),
            (7, 8), (8, 4), (9, 8), (10, 14), (11, 14), (12, 11), (13, 14),
            (14, 8), (15, 19), (16, 19), (17, 18), (18, 19), (19, 4), (20, 21),
            (21, 19), (22, 24), (23, 24), (24, 19), (25, 19), (26, 19), (27, 19),
            (28, 27), (29, 27), (30, 19), (31, 44), (32, 33), (33, 44), (34, 35),
            (35, 33), (36, 33), (37, 33), (38, 40), (39, 40), (40, 37), (41, 33),
            (42, 44), (43, 44), (45, 47), (46, 47), (47, 44), (48, 44)]))
        assert fields["labels"].labels == [
            'case', 'det', 'compound', 'nummod', 'nmod', 'case', 'punct', 'det', 'nmod',
            'punct', 'case', 'nmod:poss', 'case', 'compound', 'nmod', 'punct', 'punct',
            'amod', 'nsubj', 'dep', 'det', 'dobj', 'case', 'compound', 'nmod', 'punct',
            'punct', 'dep', 'cc', 'conj', 'punct', 'punct', 'det', 'nsubjpass', 'case',
            'nmod', 'punct', 'acl', 'case', 'compound', 'nmod', 'punct', 'auxpass',
            'advmod', 'case', 'compound', 'nmod', 'punct']
        if use_contextualizer:
            assert fields["token_representations"].array.shape[0] == len(first_sentence)
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

        # second read instance
        instance = instances[1]
        fields = instance.fields
        second_sentence = ['Ms.', 'Haag', 'plays', 'Elianti', '.']
        if include_raw_tokens:
            assert [token.metadata for token in fields["raw_tokens"].field_list] == second_sentence
        assert_allclose(fields["arc_indices"].array,
                        np.array([[0, 1], [1, 2], [3, 2], [4, 2]]))
        assert fields["labels"].labels == ['compound', 'nsubj', 'dobj', 'punct']
        if use_contextualizer:
            assert fields["token_representations"].array.shape[0] == len(second_sentence)
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[0.6757653, -0.80925614], [-1.9424553, -1.0854281],
                          [-0.09960067, 0.17525218], [0.09222834, -0.8534998],
                          [-0.66507375, -0.5633631]]),
                rtol=1e-3)

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('include_raw_tokens', (True, False))
    def test_read_from_file_undirected(self, lazy, include_raw_tokens):
        """
        Test that reading from a file in undirected mode produces instances
        that match up with those in directed mode.
        """
        directed_reader = SyntacticDependencyArcClassificationDatasetReader(
            directed=True,
            include_raw_tokens=include_raw_tokens,
            lazy=lazy)
        directed_instances = list(directed_reader.read(str(self.data_path)))

        reader = SyntacticDependencyArcClassificationDatasetReader(
            directed=False,
            include_raw_tokens=include_raw_tokens,
            lazy=lazy)
        undirected_instances = list(reader.read(str(self.data_path)))

        for directed_instance, undirected_instance in zip(directed_instances,
                                                          undirected_instances):
            # Assert that their tokens are the same
            if include_raw_tokens:
                assert ([token.metadata for token in
                         undirected_instance.fields["raw_tokens"].field_list] ==
                        [token.metadata for token in
                         directed_instance.fields["raw_tokens"].field_list])
            # Iterate over the pair token indices
            for idx in range(directed_instance.fields["arc_indices"].array.shape[0]):
                # This pair in the undirected dataset should match
                # the directed instance exactly.
                assert_allclose(
                    undirected_instance.fields["arc_indices"].array[2 * idx],
                    directed_instance.fields["arc_indices"].array[idx])
                assert (undirected_instance.fields["labels"].labels[2 * idx] ==
                        directed_instance.fields["labels"].labels[idx])
                # This pair in the undirected dataset should match the
                # directed instance except for the fact that it is reversed.
                undirected_arc_index = undirected_instance.fields["arc_indices"].array[
                    (2 * idx) + 1]
                assert_allclose(
                    np.flip(undirected_arc_index, axis=-1),
                    directed_instance.fields["arc_indices"].array[idx])
                assert (undirected_instance.fields["labels"].labels[(2 * idx) + 1] ==
                        directed_instance.fields["labels"].labels[idx])

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('directed', (True, False))
    @pytest.mark.parametrize('include_raw_tokens', (True, False))
    @pytest.mark.parametrize('max_instances', (1, 2, 0.5, 0.75, 1.0))
    def test_reproducible_with_and_without_contextualization(
            self, lazy, directed, include_raw_tokens, max_instances):
        """
        Test that the generated examples are the same with and without contextualization.
        """
        uncontextualized_params = Params({
            "lazy": lazy,
            "directed": directed,
            "max_instances": max_instances,
            "include_raw_tokens": include_raw_tokens})
        uncontextualized_reader = SyntacticDependencyArcClassificationDatasetReader.from_params(
            uncontextualized_params)
        uncontextualized_instances = list(uncontextualized_reader.read(str(self.data_path)))

        contextualized_params = Params({
            "lazy": lazy,
            "directed": directed,
            "max_instances": max_instances,
            "contextualizer": {
                "type": "precomputed_contextualizer",
                "representations_path": self.contextualizer_path
            },
            "include_raw_tokens": include_raw_tokens})
        contextualized_reader = SyntacticDependencyArcClassificationDatasetReader.from_params(
            contextualized_params)
        contextualized_instances = list(contextualized_reader.read(str(self.data_path)))
        # Assert they are the same
        for uncontextualized_instance, contextualized_instance in zip(uncontextualized_instances,
                                                                      contextualized_instances):
            if include_raw_tokens:
                assert ([token.metadata for token in uncontextualized_instance.fields["raw_tokens"].field_list] ==
                        [token.metadata for token in contextualized_instance.fields["raw_tokens"].field_list])
            assert_allclose(uncontextualized_instance.fields["arc_indices"].array,
                            contextualized_instance.fields["arc_indices"].array)
            assert (uncontextualized_instance.fields["labels"].labels ==
                    contextualized_instance.fields["labels"].labels)
            contextualized_extra_keys = list(set(contextualized_instance.fields.keys()) -
                                             set(uncontextualized_instance.fields.keys()))
            assert (set(contextualized_extra_keys) == set(["token_representations"]))

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('use_contextualizer', (True, False))
    @pytest.mark.parametrize('directed', (True, False))
    @pytest.mark.parametrize('include_raw_tokens', (True, False))
    @pytest.mark.parametrize('max_instances', (1, 2, 0.5, 0.75, 1.0))
    def test_truncation(self, lazy, use_contextualizer, directed,
                        include_raw_tokens, max_instances):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = SyntacticDependencyArcClassificationDatasetReader(
            contextualizer=contextualizer,
            directed=directed,
            include_raw_tokens=include_raw_tokens,
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
