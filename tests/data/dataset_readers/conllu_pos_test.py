from allennlp.common import Params
import numpy as np
from numpy.testing import assert_allclose
import pytest

from contexteval.common.custom_test_case import CustomTestCase
from contexteval.contextualizers import PrecomputedContextualizer
from contexteval.data.dataset_readers import ConllUPOSDatasetReader


class TestConllUPOSDatasetReader():
    data_path = CustomTestCase.FIXTURES_ROOT / "data" / "pos" / "en_ewt-ud.conllu"
    contextualizer_path = (CustomTestCase.FIXTURES_ROOT / "contextualizers" /
                           "precomputed_elmo" / "elmo_layers_all.hdf5")

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('use_contextualizer', (True, False))
    def test_read_from_file(self, lazy, use_contextualizer):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = ConllUPOSDatasetReader(contextualizer=contextualizer,
                                        lazy=lazy)
        instances = list(reader.read(str(self.data_path)))

        # First read instance
        instance = instances[0]
        fields = instance.fields
        assert [token.metadata for token in fields["raw_tokens"].field_list] == [
            'Al', '-', 'Zaman', ':', 'American', 'forces', 'killed', 'Shaikh', 'Abdullah',
            'al', '-', 'Ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the',
            'town', 'of', 'Qaim', ',', 'near', 'the', 'Syrian', 'border', '.']
        assert fields["labels"].labels == [
            'PROPN', 'PUNCT', 'PROPN', 'PUNCT', 'ADJ', 'NOUN', 'VERB', 'PROPN', 'PROPN',
            'PROPN', 'PUNCT', 'PROPN', 'PUNCT', 'DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'ADP',
            'DET', 'NOUN', 'ADP', 'PROPN', 'PUNCT', 'ADP', 'DET', 'ADJ', 'NOUN', 'PUNCT']
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[0.43633628, -0.5755847], [-0.2244201, -0.3955103],
                          [-1.8495967, -1.6728945], [-1.0596983, -0.10573974],
                          [-0.15140322, -0.7195155], [-2.3639536, -0.42766416],
                          [-0.3464077, -0.6743664], [-0.5407328, -0.9869094],
                          [-1.2095747, 0.8123201], [0.46097872, 0.8609313],
                          [-0.46175557, 0.42401582], [-0.42247432, -0.91118157],
                          [-0.41762316, -0.5272959], [0.69995964, -0.16589859],
                          [-1.4730558, -0.23568547], [-0.30440047, -0.8264297],
                          [-0.40472034, -0.15715468], [-1.3681564, -0.08945632],
                          [-0.6464306, 0.52979404], [-0.35902542, 0.8537967],
                          [-2.1601028, 1.0484889], [-0.42148307, 0.11593458],
                          [-0.81707406, 0.47127616], [-0.8185376, -0.20927876],
                          [-1.4944136, 0.2279036], [-1.244726, 0.27427846],
                          [-1.366718, 0.9977276], [-1.0117195, 0.27465925],
                          [-0.6697843, -0.24481633]]),
                rtol=1e-4)

        # Second read instance
        instance = instances[1]
        fields = instance.fields
        assert [token.metadata for token in fields["raw_tokens"].field_list] == [
            '[', 'This', 'killing', 'of', 'a', 'respected', 'cleric', 'will', 'be', 'causing',
            'us', 'trouble', 'for', 'years', 'to', 'come', '.', ']']
        assert fields["labels"].labels == [
            'PUNCT', 'DET', 'NOUN', 'ADP', 'DET', 'ADJ', 'NOUN', 'AUX', 'AUX', 'VERB',
            'PRON', 'NOUN', 'ADP', 'NOUN', 'PART', 'VERB', 'PUNCT', 'PUNCT']
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[-0.21313506, -0.9986056], [-0.9670943, -1.293689],
                          [-0.9337523, -0.2829439], [-0.14427447, -1.3481213],
                          [1.0426146, -1.2611127], [-0.03402041, -0.90879065],
                          [-2.1094723, -0.65833807], [-2.52652, 0.05855975],
                          [-1.5565295, -0.62821376], [-1.016165, -0.6203798],
                          [-0.5337064, -1.0520142], [-1.2524656, -1.2280166],
                          [0.05167481, -0.63919723], [-1.9454485, -1.7038071],
                          [0.24676055, 1.0511997], [-1.4455109, -2.3033257],
                          [-2.0335193, -1.3011322], [-0.9321909, -0.09861001]]),
                rtol=1e-4)

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('max_instances', (1, 2, 0.5, 0.75, 1.0))
    def test_reproducible_with_and_without_contextualization(self, lazy, max_instances):
        uncontextualized_params = Params({
            "max_instances": max_instances,
            "lazy": lazy})
        uncontextualized_reader = ConllUPOSDatasetReader.from_params(uncontextualized_params)
        uncontextualized_instances = list(uncontextualized_reader.read(str(self.data_path)))

        contextualized_params = Params({
            "max_instances": max_instances,
            "lazy": lazy,
            "contextualizer": {
                "type": "precomputed_contextualizer",
                "representations_path": self.contextualizer_path
            }})
        contextualized_reader = ConllUPOSDatasetReader.from_params(contextualized_params)
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
        reader = ConllUPOSDatasetReader(
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
