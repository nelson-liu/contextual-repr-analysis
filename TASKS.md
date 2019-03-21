This document describes in detail the various tasks we evaluate on and the setup
of the experiments.

## CCG Supertagging

Given a sequence of contextualized word representations, label each token with a
CCG supertag. We train and evaluate on the CCGBank with the complete tagset.

## POS Tagging

Given a sequence of contextualized word representations, label each token with a
part of speech tag. We train and evaluate on the WSJ section of the PTB and the
universal POS tags of the Universal Dependencies English Web Treebank.

## Chunking

Given a sequence of contextualized word representations, label each token with a
syntactic chunk tag (BIO tagset). This is the same data as used in the CoNLL
2000 shared task, but we follow the setup
of [Søgaard and Goldberg (2016)](http://anthology.aclweb.org/P16-2038) and use
section 19 of the Penn Treebank as development data.

## Ancestor (Parent, Grandparent, Great-Grandparent) Constituency Prediction

Given a sequence of contextualized word representations, label each token with
the syntactic constituent of its parent, grandparent, or great-grandparent. The
data is taken from the Penn Treebank. We remove the empty top layer in the
treebank trees.

In cases where the token doesn't have a grandparent or great-grandparent, we
predict a special `"None"` label.

## {Syntactic / Semantic} Dependency Arc Prediction

Given the representation of two tokens (`child` and `parent`), predict whether
there exists a syntactic dependency arc going from (`parent` -> `child`).

For syntactic dependencies, we use the Penn Treebank automatically converted
into Universal Dependencies, as well as the Universal Dependencies English Web
Treebank. For semantic dependencies, we use the DM data of SemEval 2015 Task 18
(Broad-coverage Semantic Dependency Parsing).

To generate the negative instances used during training, for each (`child`,
`parent`) example, we create a new example (`child`, `fake parent`), where `fake
parent` is a random other token in the sentence that is not a head of `child`.

During evaluation, we evaluate on both the positive examples and the generated
negative examples.

## {Syntactic / Semantic} Dependency Arc Classification

Given the representation of two tokens (`child` and `parent`), where there is a
directed arc (`parent` -> `child`), predict the label of the arc.

For syntactic dependencies, we use the Penn Treebank automatically converted
into Universal Dependencies, as well as the Universal Dependencies English Web
Treebank. For semantic dependencies, we use the DM data of SemEval 2015 Task 18
(Broad-coverage Semantic Dependency Parsing).

## Named Entity Recognition

Given a sequence of contextualized word representations, label each token with a
NER tag (IOB-1). This is the same data as used in the CoNLL 2003 shared task.

## Coreference Arc Prediction

Given the representation of two tokens `child` and `parent`, where `child` and
`parent` are orthographically differing tokens and `parent` occurs before
`child`, predict whether `child` and `parent` corefer. We use the CoNLL 2012 OntoNotes
dataset for this task.

To generate the negative examples used during training, for each (`child`,
`parent`) example, we create a new example (`child`, `fake parent`) where `fake
parent` occurs before `child`, orthographically differs, and `fake parent` is
not in the same coreference cluster as `child`.

## Semantic Tagging

Given a sequence of contextualized word representations, label each token with a
semantic tag representing the lexical semantics. This uses the semantic tagging
dataset from
["Semantic Tagging with Deep Residual Networks" (Bjerva et al., 2016)](https://arxiv.org/abs/1609.07053),
which is built from the Gronigen Meaning Bank.

## Grammatical Error Correction

Given a sequence of contextualized word representations, label each token by
whether is a grammatical or error or not in the sequence (IO tagging). We use
the data from a version of the publicly released FCE dataset
[(Yannakoudakis et al., 2011)](https://www.aclweb.org/anthology/P11-1019),
modified for experiments on error detection
by [Rei & Yannakoudakis (2016)](https://arxiv.org/abs/1607.06153).

## Coordination Boundary Detection

Given a sentence with a conjunction, tag the conjuncts in the sentence (IOB-1)
tagging. We use the
[Penn Treebank extended with coordination annotation](https://github.com/Jess1ca/CoordinationExtPTB),
as described in [Ficler et al., 2016](https://arxiv.org/abs/1606.02529), and train and
evaluate only on sentences with at least one conjunct.

## Adposition Supersense Prediction (Role and Function)

Given an adposition (preposition or possessive) in context, label the "role" and
"form" of the adposition. We use
the [STREUSLE 4.0 dataset](https://github.com/nert-gu/streusle/tree/v4.0), and
only train and evaluate on single-token adpositions (role or function starts
with `p.`).
