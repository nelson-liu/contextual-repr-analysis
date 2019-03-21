from allennlp.common import Params
import numpy as np
from numpy.testing import assert_allclose

from contexteval.contextualizers import Contextualizer, PrecomputedContextualizer
from contexteval.common.custom_test_case import CustomTestCase


class TestPrecomputedContextualizer(CustomTestCase):
    model_paths = CustomTestCase.FIXTURES_ROOT / "contextualizers" / "precomputed_elmo"
    sentence_1 = ("The cryptocurrency space is now figuring out to have "
                  "the highest search on Google globally .".split(" "))
    sentence_2 = "Bitcoin alone has a sixty percent share of global search .".split(" ")
    sentence_3 = "ELMo has had trouble handling sentences / with forward slashes .".split(" ")

    def test_precomputed_contextualizer_all_elmo_layers(self):
        all_elmo_layers_path = self.model_paths / "elmo_layers_all.hdf5"
        rep_dim = 1024
        num_sentences = 3

        # Test the first layer (index 0)
        all_elmo_layers_0 = PrecomputedContextualizer(all_elmo_layers_path,
                                                      layer_num=0)
        representations = all_elmo_layers_0([
            self.sentence_1, self.sentence_2, self.sentence_3])
        assert len(representations) == num_sentences

        first_sentence_representation = representations[0]
        seq_len = 16
        assert first_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            first_sentence_representation[:, :1].cpu().numpy()[:4],
            np.array([[-0.3288476], [-0.28436223], [0.9835328], [0.1915474]]),
            rtol=1e-5)
        second_sentence_representation = representations[1]
        seq_len = 11
        assert second_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            second_sentence_representation[:, :1].cpu().numpy()[:4],
            np.array([[-0.23547989], [-1.7968968], [-0.09795779], [0.10400581]]),
            rtol=1e-5)
        third_sentence_representation = representations[2]
        seq_len = 11
        assert third_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            third_sentence_representation[:, :1].cpu().numpy()[:4],
            np.array([[0.7506348], [-0.09795779], [0.08865512], [0.6102083]]),
            rtol=1e-5)

        # Test the second layer (index 1)
        all_elmo_layers_1 = PrecomputedContextualizer(all_elmo_layers_path,
                                                      layer_num=1)
        representations = all_elmo_layers_1([
            self.sentence_1, self.sentence_2, self.sentence_3])
        assert len(representations) == num_sentences

        first_sentence_representation = representations[0]
        seq_len = 16
        assert first_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            first_sentence_representation[:, :1].cpu().numpy()[:4],
            np.array([[0.02916196], [-0.618347], [0.04200662], [-0.28494996]]),
            rtol=1e-5)
        second_sentence_representation = representations[1]
        seq_len = 11
        assert second_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            second_sentence_representation[:, :1].cpu().numpy()[:4],
            np.array([[0.04939255], [-0.08436887], [-0.10033038], [0.23103642]]),
            rtol=1e-5)
        third_sentence_representation = representations[2]
        seq_len = 11
        assert third_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            third_sentence_representation[:, :1].cpu().numpy()[:4],
            np.array([[0.19448458], [-0.014540106], [0.23244698], [-1.1397098]]),
            rtol=1e-5)

        # Test the third / last layer (index 2)
        all_elmo_layers_2 = PrecomputedContextualizer(all_elmo_layers_path)
        representations = all_elmo_layers_2([
            self.sentence_1, self.sentence_2, self.sentence_3])
        assert len(representations) == num_sentences

        first_sentence_representation = representations[0]
        seq_len = 16
        assert first_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            first_sentence_representation[:, :1].cpu().numpy()[:4],
            np.array([[0.28029996], [-1.1247718], [-0.45496008], [-0.25592107]]),
            rtol=1e-5)
        second_sentence_representation = representations[1]
        seq_len = 11
        assert second_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            second_sentence_representation[:, :1].cpu().numpy()[:4],
            np.array([[-0.12891075], [-0.67801315], [0.021882683], [0.03998524]]),
            rtol=1e-5)
        third_sentence_representation = representations[2]
        seq_len = 11
        assert third_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            third_sentence_representation[:, :1].cpu().numpy()[:4],
            np.array([[0.17843074], [0.49779615], [0.36996722], [-1.154212]]),
            rtol=1e-5)

    def test_precomputed_contextualizer_all_elmo_layers_first_half(self):
        all_elmo_layers_path = self.model_paths / "elmo_layers_all.hdf5"
        num_sentences = 3

        # Test the first layer (index 0)
        for layer_num in [0, 1, 2]:
            all_elmo = PrecomputedContextualizer(
                all_elmo_layers_path, layer_num=0)
            first_half_elmo = PrecomputedContextualizer(
                all_elmo_layers_path, layer_num=0, first_half_only=True)
            first_half_representations = first_half_elmo([
                self.sentence_1, self.sentence_2, self.sentence_3])
            representations = all_elmo([
                self.sentence_1, self.sentence_2, self.sentence_3])
            assert len(first_half_representations) == num_sentences
            assert len(representations) == num_sentences
            for first_half_repr, full_repr in zip(first_half_representations, representations):
                assert_allclose(first_half_repr.cpu().numpy(),
                                full_repr[:, :512].cpu().numpy(),
                                rtol=1e-5)

    def test_precomputed_contextualizer_all_elmo_layers_second_half(self):
        all_elmo_layers_path = self.model_paths / "elmo_layers_all.hdf5"
        num_sentences = 3

        # Test the first layer (index 0)
        for layer_num in [0, 1, 2]:
            all_elmo = PrecomputedContextualizer(
                all_elmo_layers_path, layer_num=0)
            second_half_elmo = PrecomputedContextualizer(
                all_elmo_layers_path, layer_num=0, second_half_only=True)
            second_half_representations = second_half_elmo([
                self.sentence_1, self.sentence_2, self.sentence_3])
            representations = all_elmo([
                self.sentence_1, self.sentence_2, self.sentence_3])
            assert len(second_half_representations) == num_sentences
            assert len(representations) == num_sentences
            for second_half_repr, full_repr in zip(second_half_representations, representations):
                assert_allclose(second_half_repr.cpu().numpy(),
                                full_repr[:, 512:].cpu().numpy(),
                                rtol=1e-5)

    def test_precomputed_contextualizer_top_elmo_layers(self):
        top_elmo_layers_path = self.model_paths / "elmo_layers_top.hdf5"
        params = Params({
            "type": "precomputed_contextualizer",
            "representations_path": top_elmo_layers_path})
        top_elmo_layers = Contextualizer.from_params(params)
        rep_dim = 1024

        representations = top_elmo_layers([
            self.sentence_1, self.sentence_2, self.sentence_3])
        assert len(representations) == 3

        first_sentence_representation = representations[0]
        seq_len = 16
        assert first_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            first_sentence_representation[:, :1].cpu().numpy()[:4],
            np.array([[0.28029996], [-1.1247723], [-0.45496008], [-0.25592047]]),
            rtol=1e-5)
        second_sentence_representation = representations[1]
        seq_len = 11
        assert second_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            second_sentence_representation[:, :1].cpu().numpy()[:4],
            np.array([[-0.12891075], [-0.67801315], [0.021882683], [0.03998524]]),
            rtol=1e-5)
        third_sentence_representation = representations[2]
        seq_len = 11
        assert third_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            third_sentence_representation[:, :1].cpu().numpy()[:4],
            np.array([[0.17843074], [0.49779615], [0.36996722], [-1.154212]]),
            rtol=1e-5)

    def test_precomputed_contextualizer_scalar_mix(self):
        all_elmo_layers_path = self.model_paths / "elmo_layers_all.hdf5"
        all_elmo_layers_params = Params({
            "type": "precomputed_contextualizer",
            "representations_path": all_elmo_layers_path,
            "scalar_weights": [0.0, 0.0, 1.0],
            "gamma": 0.5
        })
        all_elmo_layers = Contextualizer.from_params(all_elmo_layers_params)

        top_elmo_layers_path = self.model_paths / "elmo_layers_top.hdf5"
        top_elmo_layers_params = Params({
            "type": "precomputed_contextualizer",
            "representations_path": top_elmo_layers_path})
        top_elmo_layers = Contextualizer.from_params(top_elmo_layers_params)

        rep_dim = 1024

        top_layers_representations = top_elmo_layers([
            self.sentence_1, self.sentence_2, self.sentence_3])
        assert len(top_layers_representations) == 3
        all_layers_representations = all_elmo_layers([
            self.sentence_1, self.sentence_2, self.sentence_3])
        assert len(all_layers_representations) == 3

        first_sentence_representation = all_layers_representations[0]
        seq_len = 16
        assert first_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            first_sentence_representation.cpu().numpy(),
            (top_layers_representations[0] * 0.5).cpu().numpy(),
            rtol=1e-5)

        second_sentence_representation = all_layers_representations[1]
        seq_len = 11
        assert second_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            second_sentence_representation.cpu().numpy(),
            (top_layers_representations[1] * 0.5).cpu().numpy(),
            rtol=1e-5)

        third_sentence_representation = all_layers_representations[2]
        seq_len = 11
        assert third_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            third_sentence_representation.cpu().numpy(),
            (top_layers_representations[2] * 0.5).cpu().numpy(),
            rtol=1e-5)
