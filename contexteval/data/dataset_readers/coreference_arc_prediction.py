import logging
import collections
from typing import DefaultDict, Dict, List, Optional, Set, Tuple, Union
import random

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, Field, ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_utils import Ontonotes
import numpy as np
from torch import FloatTensor

from contexteval.contextualizers import Contextualizer
from contexteval.data.dataset_readers import TruncatableDatasetReader
from contexteval.data.fields import SequenceArrayField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def canonicalize_clusters(clusters: DefaultDict[int, List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """
    The CONLL 2012 data includes 2 annotatated spans which are identical,
    but have different ids. This checks all clusters for spans which are
    identical, and if it finds any, merges the clusters containing the
    identical spans.
    """
    merged_clusters: List[Set[Tuple[int, int]]] = []
    for cluster in clusters.values():
        cluster_with_overlapping_mention = None
        for mention in cluster:
            # Look at clusters we have already processed to
            # see if they contain a mention in the current
            # cluster for comparison.
            for cluster2 in merged_clusters:
                if mention in cluster2:
                    # first cluster in merged clusters
                    # which contains this mention.
                    cluster_with_overlapping_mention = cluster2
                    break
            # Already encountered overlap - no need to keep looking.
            if cluster_with_overlapping_mention is not None:
                break
        if cluster_with_overlapping_mention is not None:
            # Merge cluster we are currently processing into
            # the cluster in the processed list.
            cluster_with_overlapping_mention.update(cluster)
        else:
            merged_clusters.append(set(cluster))
    return [list(c) for c in merged_clusters]


def filter_clusters(clusters: List[List[Tuple[int, int]]],
                    max_span_size: int) -> List[List[Tuple[int, int]]]:
    if max_span_size < 1:
        raise ValueError("max_span_size must greater than or equal to 1.")
    filtered_clusters = []
    for cluster in clusters:
        filtered_cluster = [span for span in cluster if span[1] - span[0] < max_span_size]
        if filtered_cluster:
            filtered_clusters.append(filtered_cluster)
    return filtered_clusters


@DatasetReader.register("coreference_arc_prediction")
class CoreferenceArcPredictionDatasetReader(TruncatableDatasetReader):
    """
    Reads a single CoNLL-formatted file, which is a pre-processed version of the Ontonotes 5.0 data.
    Produce instances suitable for use by an auxiliary classifier that aims to predict, given a pair
    of contextualized word representations, whether a coreference arc goes from one to the
    other (child, parent). We define the child as the entity that occurs _later_ in the passage,
    and the parent as the entity that occurs _earlier_ in the passage.

    Parameters
    ----------
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
    def __init__(self,
                 contextualizer: Optional[Contextualizer] = None,
                 include_raw_tokens: bool = False,
                 max_instances: Optional[Union[int, float]] = None,
                 seed: int = 0,
                 lazy: bool = False) -> None:
        super().__init__(
            max_instances=max_instances,
            lazy=lazy)
        self._contextualizer = contextualizer
        self._include_raw_tokens = include_raw_tokens
        self._rng = random.Random(seed)
        self._seed = seed

    def _reseed(self, seed: int = 0):
        """
        Reseed the Random instance underlying the dataset reader.

        Parameters
        ----------
        seed: int, optional (default=``0``)
            The random seed to use.
        """
        self._rng = random.Random(seed)

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

        index = 0
        ontonotes_reader = Ontonotes()
        for sentences in ontonotes_reader.dataset_document_iterator(file_path):
            clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)

            total_tokens = 0
            for sentence in sentences:
                for typed_span in sentence.coref_spans:
                    # Coref annotations are on a _per sentence_
                    # basis, so we need to adjust them to be relative
                    # to the length of the document.
                    span_id, (start, end) = typed_span
                    clusters[span_id].append((start + total_tokens,
                                              end + total_tokens))
                total_tokens += len(sentence.words)

            canonical_clusters = canonicalize_clusters(clusters)

            text_sentences: List[List[str]] = [s.words for s in sentences]
            flattened_text_sentences: List[str] = [self._normalize_word(word)
                                                   for text_sentence in text_sentences
                                                   for word in text_sentence]
            sentence_arc_indices: List[Tuple[int, int]] = []
            sentence_labels: List[str] = []

            # Filter the clusters to only have single-token entities
            # TODO(nfliu): How do we handle spans here?
            filtered_clusters = filter_clusters(canonical_clusters,
                                                max_span_size=1)

            # Check if there are at least two clusters, each of which has at least 2 different items.
            # If not, then skip creating examples from this passage.
            counter = 0
            all_cluster_words = []
            all_cluster_unique_words = []
            for cluster in filtered_clusters:
                # Get the words that show up in the cluster
                cluster_words = list(tuple(flattened_text_sentences[index] for
                                           index in range(item[0], item[1] + 1)) for item in cluster)
                all_cluster_words.append(cluster_words)

                cluster_unique_words = set(cluster_words)
                all_cluster_unique_words.append(cluster_unique_words)
                if len(set(cluster_words)) >= 2:
                    counter += 1
            if counter < 2:
                continue

            if keep_idx is not None and index not in keep_idx:
                index += 1
                continue
            if count_only:
                yield 1
                continue

            # Contextualize the tokens if a Contextualizer was provided.
            # TODO (nfliu): How can we make this batched?
            # Would make contextualizers that use the GPU much faster.
            if self._contextualizer:
                token_representations = self._contextualizer([flattened_text_sentences])[0]
            else:
                token_representations = None

            # For each cluster with 2+ different items, make positive examples between each of the different items
            # that are different strings and make negative examples between each of the different items and a
            # random token from another cluster.
            assert ((len(filtered_clusters) == len(all_cluster_words)) &
                    (len(all_cluster_words) == len(all_cluster_unique_words)))

            for cluster_index, (cluster_spans, cluster_words, cluster_unique_words) in enumerate(zip(
                    filtered_clusters, all_cluster_words, all_cluster_unique_words)):
                # Don't make examples from this if there is only 1 unique item.
                if len(cluster_unique_words) < 2:
                    continue
                # Get all combinations of cluster spans (a, b), where a occurs
                # in the text before b.
                all_coreferring_spans = []
                for parent_cluster_span in cluster_spans:
                    for child_cluster_span in cluster_spans:
                        # Skip child_cluster_span if it occurs before the parent_span.
                        # TODO (nfliu): this is single-word specific
                        if child_cluster_span[0] < parent_cluster_span[0]:
                            continue

                        # Skip this (child_cluster_span, parent_cluster_span) pair if the words are identical
                        if (flattened_text_sentences[child_cluster_span[0]:child_cluster_span[1] + 1] ==
                                flattened_text_sentences[parent_cluster_span[0]:parent_cluster_span[1] + 1]):
                            continue
                        # Add to the set of coreference candidates
                        all_coreferring_spans.append((child_cluster_span, parent_cluster_span))

                # Take the coreference_candidates and generate positive and negative examples
                for (child_span, parent_span) in all_coreferring_spans:
                    # TODO (nfliu): This is single-word specific, will have to change
                    # if we generalize to spans
                    sentence_arc_indices.append((child_span[0], parent_span[0]))
                    sentence_labels.append("1")

                    # Generate a negative example for the child.
                    other_clusters = [cluster for i, cluster in
                                      enumerate(filtered_clusters) if i != cluster_index]
                    negative_coreferent = self._sample_negative_coreferent(
                        other_clusters, child_span[0])
                    if negative_coreferent:
                        sentence_arc_indices.append(
                            (child_span[0], negative_coreferent[0]))
                        sentence_labels.append("0")
            yield self.text_to_instance(
                tokens=flattened_text_sentences,
                arc_indices=sentence_arc_indices,
                token_representations=token_representations,
                labels=sentence_labels)
            index += 1

    def _sample_negative_coreferent(self,  # type: ignore
                                    all_cluster_spans,
                                    child_index):
        negative_coreferent_options = [span for cluster_spans in all_cluster_spans
                                       for span in cluster_spans if span[0] < child_index]
        if not negative_coreferent_options:
            return None
        return self._rng.choice(negative_coreferent_options)

    def text_to_instance(self,  # type: ignore
                         tokens: List[str],
                         arc_indices: List[Tuple[int, int]],
                         token_representations: FloatTensor = None,
                         labels: List[str] = None):
        """
        Parameters
        ----------
        tokens : ``List[str]``, required.
            The tokens in the sentence to be encoded.
        arc_indices: ``List[Tuple[int, int]]``, required.
            A List of tuples, where each item denotes an arc. An arc is a
            tuple of (child index, parent index). Indices are 0 indexed.
        token_representations: ``FloatTensor``, optional (default=``None``)
            Precomputed token representations to use in the instance. If ``None``,
            we use a ``Contextualizer`` provided to the dataset reader to calculate
            the token representations. Shape is (seq_len, representation_dim).
        labels: ``List[str]``, optional (default=``None``)
            The labels of the arcs. ``None`` indicates that labels are not
            provided.

        Returns
        -------
        An ``Instance`` containing the following fields:
            raw_tokens : ListField[MetadataField]
                The raw str tokens in the sequence. Each MetadataField stores the raw string
                of a single token.
            arc_indices : ``SequenceArrayField``
                Array of shape (num_arc_indices, 2) corresponding to the arc indices.
                The first column holds the child indices, and the 2nd column holds
                their respective parent indices.
            token_representations: ``ArrayField``
                Contains the representation of the tokens.
            labels: ``SequenceLabelField``
                The labels corresponding each arc represented in token_indices.
        """
        fields: Dict[str, Field] = {}

        # Add tokens to the field
        if self._include_raw_tokens:
            fields["raw_tokens"] = ListField([MetadataField(token) for token in tokens])
        # Add arc indices to the field
        arc_indices_field = SequenceArrayField(
            np.array(arc_indices, dtype="int64"))
        fields["arc_indices"] = arc_indices_field

        if token_representations is None and self._contextualizer:
            # Contextualize the tokens
            token_representations = self._contextualizer([tokens])[0]

        # Add representations of the tokens at the arc indices to the field
        # If we don't have representations, use an empty numpy array.
        if token_representations is not None:
            fields["token_representations"] = ArrayField(
                token_representations.numpy())
        if labels:
            fields["labels"] = SequenceLabelField(labels, arc_indices_field)
        return Instance(fields)

    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word
