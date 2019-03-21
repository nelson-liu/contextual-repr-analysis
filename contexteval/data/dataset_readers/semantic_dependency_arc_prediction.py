from typing import List, Optional, Set, Tuple
import logging
from overrides import overrides
import itertools

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from contexteval.data.dataset_readers import DependencyArcPredictionDatasetReader

logger = logging.getLogger(__name__)

FIELDS = ["id", "form", "lemma", "pos", "head", "deprel", "top", "pred", "frame"]


def parse_sentence(sentence: str):
    annotated_sentence = []
    arc_indices = []
    arc_labels = []
    preds = []

    lines = [line for line in sentence.split("\n")
             if line and not line.strip().startswith("#")]

    for line_idx, line in enumerate(lines):
        annotated_token = dict(zip(FIELDS, line.split("\t")))
        if annotated_token['pred'] == "+":
            preds.append(line_idx)
        annotated_sentence.append(annotated_token)

    for line_idx, line in enumerate(lines):
        for pred_idx, arg in enumerate(line.split("\t")[len(FIELDS):]):
            if arg != "_":
                arc_indices.append((line_idx, preds[pred_idx]))
                arc_labels.append(arg)

    return annotated_sentence, arc_indices, arc_labels


def lazy_parse(text: str):
    for sentence in text.split("\n\n"):
        if sentence:
            yield parse_sentence(sentence)


@DatasetReader.register("semantic_dependency_arc_prediction")
class SemanticDependencyArcPredictionDatasetReader(DependencyArcPredictionDatasetReader):
    """
    Reads a file in the SemEval 2015 Task 18 (Broad-coverage Semantic Dependency Parsing)
    format and produce instances suitable for use by an auxiliary classifier that aims
    to predict, given a pair of contextualized word representations, whether a
    directed arc goes from the second to the first (child, parent).

    Parameters
    ----------
    negative_sampling_method: str
        How to generate negative examples for use in the dependency arc
        prediction task. "balanced" indicates that for each positive (child, parent)
        example, we will produce a new example with (child, random_word), where
        random_word is a random token in the sequence that is not a parent of the child.
        "all" indicates that we will take all available negative examples
        (i.e., all pairs of words a, b where a is not the child of b).
    contextualizer: Contextualizer, optional (default=``None``)
        If provided, it is used to produce contextualized representations of the text.
    include_raw_tokens: bool, optional (default=``False``)
        Whether to include the raw tokens in the generated instances. This is
        False by default since it's slow, but it is necessary if you want to use
        your ``Contextualizer`` as part of a model (e.g., for finetuning).
    max_instances: int or float, optional (default=``None``)
        The number of instances to use during training. If int, this value is taken
        to be the absolute amount of instances to use. If float, this value indicates
        that we should use that proportion of the total training data. If ``None``,
        all instances are used.
    seed: int, optional (default=``0``)
        The random seed to use.
    lazy : ``bool``, optional (default=``False``)
        If this is true, ``instances()`` will return an object whose ``__iter__`` method
        reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.
    """
    @overrides
    def _read_dataset(self,
                      file_path: str,
                      count_only: bool = False,
                      keep_idx: Optional[Set[int]] = None):
        """
        Yield instances from the file_path.

        Parameters
        ----------
        file_path: str, required
            The path to the data file.
        count_only: bool, optional (default=``False``)
            If True, no instances are returned and instead a dummy object is
            returned. This is useful for quickly counting the number of instances
            in the data file, since creating instances is relatively expensive.
        keep_idx: Set[int], optional (default=``None``)
            If not None, only yield instances whose index is in this set.
        """
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        # Reseed for reproducibility
        self._reseed(seed=self._seed)
        if count_only:
            logger.info("Counting instances (arc prediction) from dataset at: %s", file_path)
        else:
            logger.info("Reading semantic dependency parsing data (arc prediction) from dataset at: %s", file_path)

        index = 0
        with open(file_path) as sdp_file:
            # We are only predicting whether there exists an arc from parent to child
            for annotation, directed_arc_indices, arc_labels in lazy_parse(sdp_file.read()):
                # If there are no arc indices, then this sentence does not produce any
                # Instances and we should thus skip it.
                if not directed_arc_indices:
                    continue

                if keep_idx is not None and index not in keep_idx:
                    index += 1
                    continue
                if count_only:
                    yield 1
                    continue

                # Get the tokens in the sentence and contextualize them, storing the results.
                tokens = [x["form"] for x in annotation]

                # Contextualize the tokens if a Contextualizer was provided.
                # TODO (nfliu): How can we make this batched?
                # Would make contextualizers that use the GPU much faster.
                if self._contextualizer:
                    token_representations = self._contextualizer([tokens])[0]
                else:
                    token_representations = None

                sentence_arc_indices: List[Tuple[int, int]] = []
                sentence_labels: List[str] = []
                # Generate negative examples for "all" negative_sampling_method
                if self._negative_sampling_method == "all":
                    all_arcs = set(itertools.permutations(list(range(len(tokens))), 2))
                    negative_arcs = all_arcs - set(directed_arc_indices)
                    for negative_arc_index in negative_arcs:
                        sentence_arc_indices.append(negative_arc_index)  # type: ignore
                        sentence_labels.append("0")

                # Iterate over each of the (directed) arc_indices
                for positive_arc_index in directed_arc_indices:
                    sentence_arc_indices.append(positive_arc_index)
                    sentence_labels.append("1")

                    # If negative_sampling_method is balanced, we sample a negative example
                    # for the child_index, if we can. If no negative examples can be
                    # created for a sentence (the method below returns None in this case).
                    if self._negative_sampling_method == "balanced":
                        negative_arc_index = self._sample_negative_indices(
                            child_index=positive_arc_index[0],
                            all_arc_indices=directed_arc_indices,
                            seq_len=len(tokens))
                        if negative_arc_index:
                            sentence_arc_indices.append(negative_arc_index)  # type: ignore
                            sentence_labels.append("0")
                yield self.text_to_instance(
                    tokens=tokens,
                    arc_indices=sentence_arc_indices,
                    token_representations=token_representations,
                    labels=sentence_labels)
                index += 1
