from allennlp.common import Params
import numpy as np
from numpy.testing import assert_allclose
import pytest

from contexteval.common.custom_test_case import CustomTestCase
from contexteval.contextualizers import PrecomputedContextualizer
from contexteval.data.dataset_readers import AdpositionSupersenseTaggingDatasetReader


class TestAdpositionSupersenseTaggingDatasetReader():
    data_path = CustomTestCase.FIXTURES_ROOT / "data" / "adposition_supersenses" / "streusle.json"
    contextualizer_path = (CustomTestCase.FIXTURES_ROOT / "contextualizers" /
                           "precomputed_elmo" / "elmo_layers_all.hdf5")

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('use_contextualizer', (True, False))
    @pytest.mark.parametrize('include_raw_tokens', (True, False))
    @pytest.mark.parametrize('mode', ("role", "function"))
    def test_read_from_file(self, lazy, use_contextualizer, include_raw_tokens, mode):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = AdpositionSupersenseTaggingDatasetReader(
            mode=mode,
            include_raw_tokens=include_raw_tokens,
            contextualizer=contextualizer,
            lazy=lazy)
        instances = list(reader.read(str(self.data_path)))
        assert len(instances) == 3

        # First read instance
        instance = instances[0]
        fields = instance.fields
        if include_raw_tokens:
            assert [token.metadata for token in fields["raw_tokens"].field_list] == [
                'Have', 'a', 'real', 'mechanic', 'check', 'before', 'you', 'buy', '!!!!']
        assert_allclose(fields["label_indices"].array, np.array([5]))
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[1.0734302, 0.01504201], [0.22717158, -0.07871069],
                          [-0.04494515, -1.5083733], [-0.99489564, -0.6427601],
                          [-0.6134137, 0.21780868], [-0.72287357, 0.00998633],
                          [-2.4904299, -0.49546975], [-1.2544577, -0.3230043],
                          [-0.33858764, -0.4852887]]),
                rtol=1e-4)
        assert fields["labels"].labels == ['p.Time']

        # Second read instance
        instance = instances[1]
        fields = instance.fields
        if include_raw_tokens:
            assert [token.metadata for token in fields["raw_tokens"].field_list] == [
                "Very", "good", "with", "my", "5", "year", "old", "daughter", "."]

        assert_allclose(fields["label_indices"].array, np.array([3]))
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[1.6298808, 1.1453593], [0.5952766, -0.9594625],
                          [0.97710425, -0.19498791], [0.01909786, -0.9163474],
                          [-0.06802255, -1.0863125], [-1.223998, 0.2686447],
                          [-0.3791673, -0.71468884], [-1.1185161, -1.2551097],
                          [-1.3264754, -0.55683744]]),
                rtol=1e-4)
        if mode == "role":
            assert fields["labels"].labels == ["p.SocialRel"]
        else:
            assert fields["labels"].labels == ["p.Gestalt"]

        # Third read instance
        instance = instances[2]
        fields = instance.fields
        if include_raw_tokens:
            assert [token.metadata for token in fields["raw_tokens"].field_list] == [
                'After', 'firing', 'this', 'company', 'my', 'next', 'pool',
                'service', 'found', 'the', 'filters', 'had', 'not', 'been',
                'cleaned', 'as', 'they', 'should', 'have', 'been', '.']

        assert_allclose(fields["label_indices"].array, np.array([0, 4, 15]))
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[0.11395034, -0.20784551], [0.15244134, -0.55084044],
                          [0.7266098, -0.6036074], [0.6412934, -0.3448509],
                          [0.57441425, -0.73139024], [0.5528518, -0.19321561],
                          [0.5668789, -0.20008], [0.5737846, -1.2053688],
                          [-0.3721336, -0.8618743], [0.59511614, -0.18732266],
                          [0.72423995, 0.4306308], [0.96764237, 0.21643513],
                          [-0.40797114, 0.67060745], [-0.04556704, 0.5140952],
                          [0.422831, 0.32669073], [0.6339446, -0.44046107],
                          [-0.19289528, -0.18465114], [0.09728494, -1.0248029],
                          [0.791354, -0.2504376], [0.7951995, -0.7192571],
                          [-0.345582, -0.8098198]]),
                rtol=1e-4)
        if mode == "role":
            assert fields["labels"].labels == ["p.Time", "p.OrgRole", "p.ComparisonRef"]
        else:
            assert fields["labels"].labels == ["p.Time", "p.Gestalt", "p.ComparisonRef"]

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('mode', ("role", "function"))
    @pytest.mark.parametrize('include_raw_tokens', (True, False))
    @pytest.mark.parametrize('max_instances', (1, 2, 0.5, 0.75, 1.0))
    def test_reproducible_with_and_without_contextualization(self, lazy, mode, include_raw_tokens,
                                                             max_instances):
        uncontextualized_params = Params({
            "lazy": lazy,
            "mode": mode,
            "max_instances": max_instances,
            "include_raw_tokens": include_raw_tokens})
        uncontextualized_reader = AdpositionSupersenseTaggingDatasetReader.from_params(
            uncontextualized_params)
        uncontextualized_instances = list(uncontextualized_reader.read(str(self.data_path)))

        contextualized_params = Params({
            "lazy": lazy,
            "mode": mode,
            "max_instances": max_instances,
            "include_raw_tokens": include_raw_tokens,
            "contextualizer": {
                "type": "precomputed_contextualizer",
                "representations_path": self.contextualizer_path
            }})
        contextualized_reader = AdpositionSupersenseTaggingDatasetReader.from_params(
            contextualized_params)
        contextualized_instances = list(contextualized_reader.read(str(self.data_path)))
        # Assert they are the same
        for uncontextualized_instance, contextualized_instance in zip(uncontextualized_instances,
                                                                      contextualized_instances):
            if include_raw_tokens:
                assert ([token.metadata for token in uncontextualized_instance.fields["raw_tokens"].field_list] ==
                        [token.metadata for token in contextualized_instance.fields["raw_tokens"].field_list])
            assert_allclose(uncontextualized_instance.fields["label_indices"].array,
                            contextualized_instance.fields["label_indices"].array)
            assert (uncontextualized_instance.fields["labels"].labels ==
                    contextualized_instance.fields["labels"].labels)
            contextualized_extra_keys = list(set(contextualized_instance.fields.keys()) -
                                             set(uncontextualized_instance.fields.keys()))
            assert (set(contextualized_extra_keys) == set(["token_representations"]))

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('use_contextualizer', (True, False))
    @pytest.mark.parametrize('include_raw_tokens', (True, False))
    @pytest.mark.parametrize('mode', ("role", "function"))
    @pytest.mark.parametrize('max_instances', (1, 2, 0.5, 0.75, 1.0))
    def test_truncation(self, lazy, use_contextualizer, include_raw_tokens,
                        mode, max_instances):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = AdpositionSupersenseTaggingDatasetReader(
            mode=mode,
            include_raw_tokens=include_raw_tokens,
            contextualizer=contextualizer,
            max_instances=max_instances,
            lazy=lazy)
        instances = list(reader.read(str(self.data_path)))
        num_total_instances = 3
        max_instances_to_num_instances = {
            int(1): 1,
            int(2): 2,
            0.5: int(num_total_instances * 0.5),
            0.75: int(num_total_instances * 0.75),
            1.0: num_total_instances}
        assert len(instances) == max_instances_to_num_instances[max_instances]
