from allennlp.common import Params
import numpy as np
from numpy.testing import assert_allclose

from contexteval.contextualizers import Contextualizer
from contexteval.common.custom_test_case import CustomTestCase
from contexteval.common.util import pad_contextualizer_output


class TestGloveContextualizer(CustomTestCase):
    glove_path = str(CustomTestCase.FIXTURES_ROOT / "contextualizers" / "glove" /
                     "glove_5d_fixture.txt")
    representation_dim = 5
    num_sentences = 3
    sentence_1 = ("The cryptocurrency space is now figuring out to have "
                  "the highest search on Google globally .".split(" "))
    sentence_2 = "Bitcoin alone has a sixty percent share of global search .".split(" ")
    sentence_3 = "ELMo has had trouble handling sentences / with forward slashes .".split(" ")

    def test_glove_contextualizer_default(self):
        params = Params({
            "type": "glove_contextualizer",
            "glove_path": self.glove_path,
            "embedding_dim": self.representation_dim
        })
        glove_contextualizer = Contextualizer.from_params(params)
        representations = glove_contextualizer([
            self.sentence_1, self.sentence_2, self.sentence_3])
        assert len(representations) == self.num_sentences

        first_sentence_representation = representations[0]
        seq_len = 16
        assert first_sentence_representation.size() == (seq_len, self.representation_dim)
        assert_allclose(
            first_sentence_representation[:, :1].detach().cpu().numpy(),
            np.array([[0.464], [0.246], [0.458], [0.649], [0.273], [0.465],
                      [0.012], [0.19], [0.219], [0.199], [0.944], [0.432],
                      [0.28], [glove_contextualizer.weight[0, :1]], [0.083], [0.681]]),
            rtol=1e-5)

        second_sentence_representation = representations[1]
        seq_len = 11
        assert second_sentence_representation.size() == (seq_len, self.representation_dim)
        assert_allclose(
            second_sentence_representation[:, :1].detach().cpu().numpy(),
            np.array([[glove_contextualizer.weight[0, :1]], [0.761], [0.249], [0.571],
                      [0.952], [0.41], [0.791], [0.063], [0.555], [0.432], [0.681]]),
            rtol=1e-5)

        third_sentence_representation = representations[2]
        seq_len = 11
        assert third_sentence_representation.size() == (seq_len, self.representation_dim)
        assert_allclose(
            third_sentence_representation[:, :1].detach().cpu().numpy(),
            np.array([[glove_contextualizer.weight[0, :1]], [0.249], [0.56],
                      [0.591], [0.739], [0.222], [0.439], [0.308], [0.793],
                      [0.118], [0.681]]),
            rtol=1e-5)

    def test_glove_contextualizer_trainable(self):
        params = Params({
            "type": "glove_contextualizer",
            "glove_path": self.glove_path,
            "embedding_dim": self.representation_dim,
            "trainable": True
        })
        glove_contextualizer = Contextualizer.from_params(params)
        unpadded_representations = glove_contextualizer([
            self.sentence_1, self.sentence_2, self.sentence_3])
        token_representations, mask = pad_contextualizer_output(
            unpadded_representations)
        loss = token_representations.sum()
        loss.backward()
        glove_grads = [param.grad for name, param in
                       glove_contextualizer.named_parameters()]
        assert all([grad is not None for grad in glove_grads])

    def test_glove_contextualizer_frozen(self):
        params = Params({
            "type": "glove_contextualizer",
            "glove_path": self.glove_path,
            "embedding_dim": self.representation_dim,
            "trainable": False
        })
        glove_contextualizer = Contextualizer.from_params(params)
        unpadded_representations = glove_contextualizer([
            self.sentence_1, self.sentence_2, self.sentence_3])
        token_representations, mask = pad_contextualizer_output(
            unpadded_representations)
        loss = token_representations.sum()
        # Nothing in the contextualizer is requires_grad=True, so this
        # should be requires_grad=False and grad_fn should be None
        assert loss.grad_fn is None
        assert loss.requires_grad is False
