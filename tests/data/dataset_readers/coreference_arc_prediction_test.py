from allennlp.common import Params
import numpy as np
from numpy.testing import assert_allclose
import pytest

from contexteval.common.custom_test_case import CustomTestCase
from contexteval.contextualizers import PrecomputedContextualizer
from contexteval.data.dataset_readers import CoreferenceArcPredictionDatasetReader


class TestCoreferenceArcPredictionDatasetReader():
    data_path = CustomTestCase.FIXTURES_ROOT / "data" / "coreference_resolution" / "coref.gold_conll"
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
        reader = CoreferenceArcPredictionDatasetReader(contextualizer=contextualizer,
                                                       include_raw_tokens=include_raw_tokens,
                                                       lazy=lazy)
        instances = list(reader.read(str(self.data_path)))

        document = ['What', 'kind', 'of', 'memory', '?', 'We', 'respectfully', 'invite', 'you', 'to',
                    'watch', 'a', 'special', 'edition', 'of', 'Across', 'China', '.', 'WW', 'II',
                    'Landmarks', 'on', 'the', 'Great', 'Earth', 'of', 'China', ':', 'Eternal', 'Memories',
                    'of', 'Taihang', 'Mountain', 'Standing', 'tall', 'on', 'Taihang', 'Mountain', 'is', 'the',
                    'Monument', 'to', 'the', 'Hundred', 'Regiments', 'Offensive', '.', 'It', 'is', 'composed',
                    'of', 'a', 'primary', 'stele', ',', 'secondary', 'steles', ',', 'a', 'huge', 'round',
                    'sculpture', 'and', 'beacon', 'tower', ',', 'and', 'the', 'Great', 'Wall', ',', 'among',
                    'other', 'things', '.', 'A', 'primary', 'stele', ',', 'three', 'secondary', 'steles',
                    ',', 'and', 'two', 'inscribed', 'steles', '.', 'The', 'Hundred', 'Regiments',
                    'Offensive', 'was', 'the', 'campaign', 'of', 'the', 'largest', 'scale', 'launched',
                    'by', 'the', 'Eighth', 'Route', 'Army', 'during', 'the', 'War', 'of', 'Resistance',
                    'against', 'Japan', '.', 'This', 'campaign', 'broke', 'through', 'the', 'Japanese',
                    'army', "'s", 'blockade', 'to', 'reach', 'base', 'areas', 'behind', 'enemy', 'lines',
                    ',', 'stirring', 'up', 'anti-Japanese', 'spirit', 'throughout', 'the', 'nation', 'and',
                    'influencing', 'the', 'situation', 'of', 'the', 'anti-fascist', 'war', 'of', 'the',
                    'people', 'worldwide', '.', 'This', 'is', 'Zhuanbi', 'Village', ',', 'Wuxiang', 'County',
                    'of', 'Shanxi', 'Province', ',', 'where', 'the', 'Eighth', 'Route', 'Army', 'was',
                    'headquartered', 'back', 'then', '.', 'On', 'a', 'wall', 'outside', 'the', 'headquarters',
                    'we', 'found', 'a', 'map', '.', 'This', 'map', 'was', 'the', 'Eighth', 'Route', 'Army',
                    "'s", 'depiction', 'of', 'the', 'Mediterranean', 'Sea', 'situation', 'at', 'that', 'time',
                    '.', 'This', 'map', 'reflected', 'the', 'European', 'battlefield', 'situation', '.', 'In',
                    '1940', ',', 'the', 'German', 'army', 'invaded', 'and', 'occupied', 'Czechoslovakia', ',',
                    'Poland', ',', 'the', 'Netherlands', ',', 'Belgium', ',', 'and', 'France', '.', 'It',
                    'was', 'during', 'this', 'year', 'that', 'the', 'Japanese', 'army', 'developed', 'a',
                    'strategy', 'to', 'rapidly', 'force', 'the', 'Chinese', 'people', 'into', 'submission',
                    'by', 'the', 'end', 'of', '1940', '.', 'In', 'May', ',', 'the', 'Japanese', 'army',
                    'launched', '--', 'From', 'one', 'side', ',', 'it', 'seized', 'an', 'important', 'city',
                    'in', 'China', 'called', 'Yichang', '.', 'Um', ',', ',', 'uh', ',', 'through', 'Yichang',
                    ',', 'it', 'could', 'directly', 'reach', 'Chongqing', '.', 'Ah', ',', 'that', 'threatened',
                    'Chongqing', '.', 'Then', 'they', 'would', ',', 'ah', ',', 'bomb', 'these', 'large',
                    'rear', 'areas', 'such', 'as', 'Chongqing', '.', 'So', ',', 'along', 'with', 'the',
                    'coordinated', ',', 'er', ',', 'economic', 'blockade', ',', 'military', 'offensives', ',',
                    'and', 'strategic', 'bombings', ',', 'er', ',', 'a', 'simultaneous', 'attack', 'was',
                    'launched', 'in', 'Hong', 'Kong', 'to', 'lure', 'the', 'KMT', 'government', 'into',
                    'surrender', '.', 'The', 'progress', 'of', 'this', 'coordinated', 'offensive', 'was',
                    'already', 'very', 'entrenched', 'by', 'then', '.']

        if include_raw_tokens:
            for instance in instances:
                assert [token.metadata for token in instance.fields["raw_tokens"].field_list] == document

        # First read instance
        instance = instances[0]
        fields = instance.fields
        if include_raw_tokens:
            assert [token.metadata for token in fields["raw_tokens"].field_list] == document
        assert_allclose(fields["arc_indices"].array, np.array([
            (298, 285), (298, 288), (298, 267), (298, 288), (293, 288), (293, 273)]))
        assert fields["labels"].labels == ['1', '0', '1', '0', '1', '0']
        if use_contextualizer:
            assert fields["token_representations"].array.shape[0] == len(document)
            assert_allclose(
                fields["token_representations"].array[:4, :2],
                np.array([[-0.40419546, 0.18443017], [-0.4557378, -0.50057644],
                          [0.10493508, -0.7943226],
                          [-0.8075396, 0.87755275]]),
                rtol=1e-4)

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('include_raw_tokens', (True, False))
    @pytest.mark.parametrize('max_instances', (1, 2, 1.0))
    def test_reproducible_with_and_without_contextualization(
            self, lazy, include_raw_tokens, max_instances):
        """
        Test that the generated positive and negative examples are
        the same with and without contextualization.
        """
        uncontextualized_params = Params({
            "max_instances": max_instances,
            "lazy": lazy,
            "include_raw_tokens": include_raw_tokens,
            "seed": 0})
        uncontextualized_reader = CoreferenceArcPredictionDatasetReader.from_params(uncontextualized_params)
        uncontextualized_instances = list(uncontextualized_reader.read(str(self.data_path)))

        contextualized_params = Params({
            "max_instances": max_instances,
            "lazy": lazy,
            "include_raw_tokens": include_raw_tokens,
            "contextualizer": {
                "type": "precomputed_contextualizer",
                "representations_path": self.contextualizer_path
            },
            "seed": 0})
        contextualized_reader = CoreferenceArcPredictionDatasetReader.from_params(contextualized_params)
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
    @pytest.mark.parametrize('max_instances', (1, 2, 1.0))
    def test_truncation(self, lazy, use_contextualizer, max_instances):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = CoreferenceArcPredictionDatasetReader(
            contextualizer=contextualizer,
            max_instances=max_instances,
            lazy=lazy)
        instances = list(reader.read(str(self.data_path)))
        num_total_instances = 1
        max_instances_to_num_instances = {
            int(1): 1,
            int(2): 1,
            1.0: num_total_instances}
        assert len(instances) == max_instances_to_num_instances[max_instances]
