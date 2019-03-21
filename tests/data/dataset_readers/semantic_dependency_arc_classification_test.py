from allennlp.common import Params
import numpy as np
from numpy.testing import assert_allclose
import pytest

from contexteval.common.custom_test_case import CustomTestCase
from contexteval.contextualizers import PrecomputedContextualizer
from contexteval.data.dataset_readers import SemanticDependencyArcClassificationDatasetReader


class TestSemanticDependencyArcClassificationDatasetReader():
    data_path = CustomTestCase.FIXTURES_ROOT / "data" / "semantic_dependency" / "dm.sdp"
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
        reader = SemanticDependencyArcClassificationDatasetReader(
            contextualizer=contextualizer,
            include_raw_tokens=include_raw_tokens,
            lazy=lazy)
        instances = list(reader.read(str(self.data_path)))

        first_sentence = ['Pierre', 'Vinken', ',', '61', 'years', 'old',
                          ',', 'will', 'join', 'the', 'board', 'as',
                          'a', 'nonexecutive', 'director', 'Nov.', '29', '.']
        # First read instance
        instance = instances[0]
        fields = instance.fields
        if include_raw_tokens:
            assert [token.metadata for token in fields["raw_tokens"].field_list] == first_sentence
        assert_allclose(fields["arc_indices"].array, np.array([
            [1, 0], [1, 5], [1, 8], [4, 3],
            [5, 4], [8, 11], [8, 16], [10, 8],
            [10, 9], [14, 11], [14, 12], [14, 13],
            [16, 15]]))
        assert fields["labels"].labels == [
            'compound', 'ARG1', 'ARG1', 'ARG1', 'measure', 'ARG1', 'loc',
            'ARG2', 'BV', 'ARG2', 'BV', 'ARG1', 'of']
        if use_contextualizer:
            assert fields["token_representations"].array.shape[0] == len(first_sentence)
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[-2.2218986, -2.451917],
                          [-0.7551352, -0.7919447],
                          [-0.9525466, -0.985806],
                          [-1.4567664, 0.1637534],
                          [0.21862003, -0.00878072],
                          [-0.7341557, -0.57776076],
                          [-1.6816409, -1.2562131],
                          [-0.9079286, 0.15607932],
                          [-0.44011104, -0.3434037],
                          [0.56341827, -0.97181696],
                          [-0.7166206, -0.33435553],
                          [-0.14051008, -1.260754],
                          [0.42426592, -0.35762805],
                          [-1.0153385, -0.7949409],
                          [-0.7065723, 0.05164766],
                          [-0.11002721, -0.11039695],
                          [0.41112524, 0.27260625],
                          [-1.0369725, -0.6278316]]),
                rtol=1e-4)

        # Test the second sentence
        second_sentence = ['Mr.', 'Vinken', 'is', 'chairman', 'of',
                           'Elsevier', 'N.V.', ',', 'the', 'Dutch',
                           'publishing', 'group', '.']
        instance = instances[1]
        fields = instance.fields
        if include_raw_tokens:
            assert [token.metadata for token in fields["raw_tokens"].field_list] == second_sentence
        assert_allclose(fields["arc_indices"].array, np.array([
            [1, 0], [1, 2], [3, 2], [3, 4], [5, 4], [5, 6], [5, 11], [11, 8],
            [11, 9], [11, 10]]))
        assert fields["labels"].labels == [
            'compound', 'ARG1', 'ARG2', 'ARG1', 'ARG2', 'compound',
            'appos', 'BV', 'ARG1', 'compound']
        if use_contextualizer:
            assert fields["token_representations"].array.shape[0] == len(second_sentence)
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[0.7069745, -0.5422047],
                          [-1.8885247, -1.4432149],
                          [-1.7570897, -1.1201282],
                          [-1.2288755, -0.8003752],
                          [-0.08672556, -0.99020493],
                          [-0.6416313, -1.147429],
                          [-0.7924329, 0.14809224],
                          [-1.0645872, -1.0505759],
                          [0.69725895, -0.8735154],
                          [0.27878952, -0.339666],
                          [0.20708983, -0.7103262],
                          [-1.1115363, -0.16295972],
                          [-1.3495405, -0.8656957]]),
                rtol=1e-4)

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('include_raw_tokens', (True, False))
    def test_read_from_file_undirected(self, lazy, include_raw_tokens):
        """
        Test that reading from a file in undirected mode produces instances
        that match up with those in directed mode.
        """
        directed_reader = SemanticDependencyArcClassificationDatasetReader(
            directed=True,
            include_raw_tokens=include_raw_tokens,
            lazy=lazy)
        directed_instances = list(directed_reader.read(str(self.data_path)))

        reader = SemanticDependencyArcClassificationDatasetReader(
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
        Test that the generated positive and negative examples are
        the same with and without contextualization.
        """
        uncontextualized_params = Params({
            "max_instances": max_instances,
            "lazy": lazy,
            "directed": directed,
            "include_raw_tokens": include_raw_tokens})
        uncontextualized_reader = SemanticDependencyArcClassificationDatasetReader.from_params(
            uncontextualized_params)
        uncontextualized_instances = list(uncontextualized_reader.read(str(self.data_path)))

        contextualized_params = Params({
            "max_instances": max_instances,
            "lazy": lazy,
            "directed": directed,
            "contextualizer": {
                "type": "precomputed_contextualizer",
                "representations_path": self.contextualizer_path
            },
            "include_raw_tokens": include_raw_tokens})
        contextualized_reader = SemanticDependencyArcClassificationDatasetReader.from_params(contextualized_params)
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
        reader = SemanticDependencyArcClassificationDatasetReader(
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
