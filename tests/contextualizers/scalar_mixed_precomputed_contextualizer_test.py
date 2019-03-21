import numpy as np
from numpy.testing import assert_allclose

from contexteval.contextualizers import ScalarMixedPrecomputedContextualizer
from contexteval.common.custom_test_case import CustomTestCase


class TestScalarMixedPrecomputedContextualizer(CustomTestCase):
    model_paths = CustomTestCase.FIXTURES_ROOT / "contextualizers" / "precomputed_elmo"
    sentence_1 = ("The cryptocurrency space is now figuring out to have "
                  "the highest search on Google globally .".split(" "))
    sentence_2 = "Bitcoin alone has a sixty percent share of global search .".split(" ")
    sentence_3 = "ELMo has had trouble handling sentences / with forward slashes .".split(" ")

    def test_scalar_mixed_precomputed_contextualizer(self):
        all_elmo_layers_path = self.model_paths / "elmo_layers_all.hdf5"
        rep_dim = 1024
        num_sentences = 3
        num_layers = 3

        contextualizer = ScalarMixedPrecomputedContextualizer(
            representations_path=all_elmo_layers_path,
            num_layers=num_layers)
        representations = contextualizer([
            self.sentence_1, self.sentence_2, self.sentence_3])
        assert len(representations) == num_sentences

        first_sentence_representation = representations[0]
        seq_len = 16
        assert first_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            first_sentence_representation.cpu().detach().numpy()[:, :1],
            np.array([[-0.00646189], [-0.675827],
                      [0.1901931], [-0.11644121],
                      [-0.11938891], [-0.18601154],
                      [-0.28897947], [-0.46416435],
                      [0.28800112], [0.03580474],
                      [-0.02012073], [-0.47831267],
                      [-0.09510745], [-0.14722879],
                      [-0.20355265], [-0.9177323]]),
            rtol=1e-4)
        second_sentence_representation = representations[1]
        seq_len = 11
        assert second_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            second_sentence_representation.cpu().detach().numpy()[:, :1],
            np.array([[-0.10499936], [-0.853093], [-0.05880183],
                      [0.12500916], [-0.48518908], [-0.75637007],
                      [-0.94097614], [-0.15677695], [0.5876276],
                      [-0.20230056], [-1.0954572]]),
            rtol=1e-4)
        third_sentence_representation = representations[2]
        seq_len = 11
        assert third_sentence_representation.size() == (seq_len, rep_dim)
        assert_allclose(
            third_sentence_representation.cpu().detach().numpy()[:, :1],
            np.array([[0.3745167], [0.12843275], [0.23035645],
                      [-0.5612379], [-0.20278083], [0.2561654],
                      [0.6484469], [0.4885599], [0.505579],
                      [-0.3527689], [-0.7086841]]),
            rtol=1e-4)
