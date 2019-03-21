from allennlp.common import Params
import numpy as np
from numpy.testing import assert_allclose
import pytest

from contexteval.common.custom_test_case import CustomTestCase
from contexteval.contextualizers import PrecomputedContextualizer
from contexteval.data.dataset_readers import Conll2003NERDatasetReader


class TestConll2003NERDatasetReader():
    data_path = CustomTestCase.FIXTURES_ROOT / "data" / "ner" / "conll2003.txt"
    contextualizer_path = (CustomTestCase.FIXTURES_ROOT / "contextualizers" /
                           "precomputed_elmo" / "elmo_layers_all.hdf5")

    @pytest.mark.parametrize('lazy', (True, False))
    @pytest.mark.parametrize('use_contextualizer', (True, False))
    @pytest.mark.parametrize('label_encoding', ("IOB1", "BIOUL"))
    def test_read_from_file(self, lazy, use_contextualizer, label_encoding):
        # Set up contextualizer, if using.
        contextualizer = None
        if use_contextualizer:
            contextualizer = PrecomputedContextualizer(self.contextualizer_path)
        reader = Conll2003NERDatasetReader(label_encoding=label_encoding,
                                           contextualizer=contextualizer,
                                           lazy=lazy)
        instances = list(reader.read(str(self.data_path)))
        print(len(instances))
        # First read instance
        instance = instances[0]
        fields = instance.fields
        assert [token.metadata for token in fields["raw_tokens"].field_list] == [
            "EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."]
        if label_encoding == "IOB1":
            assert fields["labels"].labels == [
                "I-ORG", "O", "I-MISC", "O", "O", "O", "I-MISC", "O", "O"]
        else:
            assert fields["labels"].labels == [
                "U-ORG", "O", "U-MISC", "O", "O", "O", "U-MISC", "O", "O"]
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[0.9611156, -0.63280857], [-0.5790216, -0.13829914],
                          [-0.35322708, -0.22807068], [-1.6707208, -1.1125797],
                          [-2.0587592, -1.5086308], [-1.3151755, -1.6046834],
                          [0.5072891, -1.5075727], [0.11287686, -1.2473724],
                          [-0.5029946, -1.4319026]]),
                rtol=1e-4)

        # Second read instance
        instance = instances[1]
        fields = instance.fields
        assert [token.metadata for token in fields["raw_tokens"].field_list] == [
            "Peter", "Blackburn"]
        if label_encoding == "IOB1":
            assert fields["labels"].labels == [
                "I-PER", "I-PER"]
        else:
            assert fields["labels"].labels == [
                "B-PER", "L-PER"]
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[-0.5346743, -1.1600235], [-0.7508778, -1.031188]]),
                rtol=1e-4)

        # Third read instance
        instance = instances[2]
        fields = instance.fields
        assert [token.metadata for token in fields["raw_tokens"].field_list] == [
            "BRUSSELS", "1996-08-22"]
        if label_encoding == "IOB1":
            assert fields["labels"].labels == [
                "I-LOC", "O"]
        else:
            assert fields["labels"].labels == [
                "U-LOC", "O"]
        if use_contextualizer:
            assert_allclose(
                fields["token_representations"].array[:, :2],
                np.array([[0.0324477, -0.06925768], [-0.47278678, 0.530316]]),
                rtol=1e-4)

    @pytest.mark.parametrize('max_instances', (1, 2, 0.5, 0.75, 1.0))
    @pytest.mark.parametrize('lazy', (True, False))
    def test_reproducible_with_and_without_contextualization(self, max_instances, lazy):
        uncontextualized_params = Params({
            "max_instances": max_instances,
            "lazy": lazy})
        uncontextualized_reader = Conll2003NERDatasetReader.from_params(uncontextualized_params)
        uncontextualized_instances = list(uncontextualized_reader.read(str(self.data_path)))

        contextualized_params = Params({
            "lazy": lazy,
            "max_instances": max_instances,
            "contextualizer": {
                "type": "precomputed_contextualizer",
                "representations_path": self.contextualizer_path
            }})
        contextualized_reader = Conll2003NERDatasetReader.from_params(contextualized_params)
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
        reader = Conll2003NERDatasetReader(
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
