This document describes the various datasets used by the evaluation toolkit and
where they were sourced from, if any preprocessing was needed, etc.

I'm keeping track of this information to provide in an eventual public release --- much of this
data is proprietary, so we aren't allowed to distribute it. The users are thus responsible for
sourcing it themselves, and we want to make the process of doing so / using the data once they have it
as painless as possible.

## Chunking

We use the [CoNLL 2000 chunking shared task data](https://www.clips.uantwerpen.be/conll2000/chunking/).

**Why is this task interesting?**

This task lets us evaluate whether ELMo learns intuitive notions of syntactic spans.

## POS Tagging

We use the [Penn Treebank](https://catalog.ldc.upenn.edu/ldc99t42). We use the Penn Treebank in the CoNLL-X format.
We also experiment with predicting the universal POS tags of the [Universal Dependencies English Web Treebank](http://universaldependencies.org/treebanks/en_ewt/index.html).

**Why is this task interesting?**

English POS tagging is an evaluation of the morphological knowledge encoded in these models, since English
morphology is highly predictive of POS.

## Named Entity Recognition

We use the [CoNLL-2003 shared task data](https://www.clips.uantwerpen.be/conll2003/ner/).

There's a publicly available copy of dubious legality at
[this github repo](https://github.com/rishiabhishek/Important-Datasets/tree/master/conll2003).

**Why is this task interesting?**

NER is a longstanding task in NLP, and we are interesting in seeing whether contextualized representations can capture notions of named entities (since you'd expect such a property to be useful in downstream tasks, like coref or sentiment).

## CCG Supertagging

We use [CCGBank](https://catalog.ldc.upenn.edu/ldc2005t13) as the CCG
supertagging dataset. I don't think there's any way around using a LDC-licensed
dataset for the CCG supertagging task --- there are no open-access datasets.

**Why is this task interesting?**

CCG Supertagging is "almost parsing" (Srinivas and Joshi, 1998), since the detailed labels encode much hierarchical
syntactic information. This task tests how much syntactic knowledge is encoded in the contextualized representations.

## Dependency Parsing

We use the [Penn Treebank](https://catalog.ldc.upenn.edu/ldc99t42), automatically converted to Universal
Dependencies in [CoNLL-U](http://universaldependencies.org/format.html) format by 
[the Stanford Parser](https://nlp.stanford.edu/software/stanford-dependencies.html#Universal). As of June 13th 2018,
the Stanford Parser generates files in Universal Dependencies v1 format --- we thus use that format as well.

We follow the standard WSJ PTB splits in the dependency parsing literature: sections 2-21 for training, 
section 22 as development set and 23 as test set. The parsed sections can be found at: `<PTB FOLDER>/RAW/parsed/prd/wsj/`.

To generate the we use files, you'll want to download the Penn Treebank from the LDC, and then create 3 subfolders for 
`train`, `dev`, and `test`:

```
.
+-- train
|   +-- 02/
|   +-- 03/
|   +-- 04
|   etc...sections 02 to 21.
+-- dev
|   +-- 22/
+-- test
|   +-- 23/
```

Then, download and unzip the [Stanford Parser](https://nlp.stanford.edu/software/lex-parser.shtml#Download). Change
directories to wherever you unzipped, and run the following 
(as of 6/13/2018, see [here for latest instructions](https://nlp.stanford.edu/software/stanford-dependencies.html#Universal)) 
to generate the conllu files from the raw Penn Treebanks:

```
java -cp "./*" -mx1g edu.stanford.nlp.trees.ud.UniversalDependenciesConverter -treeFile <PATH TO SPLIT FOLDER> > ptb_<SPLIT>.conllu
```

This should generate files `ptb_train.conllu`, `ptb_dev.conllu`, and `ptb_test.conllu` for use, representing the 
splits of the Universal Dependencies v1 Penn Treebank in CoNLL-U format  .

### Syntactic Dependency Tasks

There are two tasks we explore with this data:

1. Syntactic Dependency Arc Prediction

**Why is this task interesting?**

This enables us to evaluate whether ELMo representations can determine whether two tokens are syntactically related. We'd hope that the representations would capture this knowledge, since understanding of how different tokens are syntactically connected would help in overall comprehension.

2. Syntactic Dependency Arc Classification

**Why is this task interesting?**

This enables us to evaluate whether ELMo encodes specific syntactic relationships between tokens. This is useful because, for example, it'd be useful to distinguish whether two words are related by SUBJ or DOBJ.

## Semantic Dependency Parsing

We use the the data in [Open SDP 1.2](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1956).
This is identical to the data used in the SemEval 2014 and 2015 shared tasks on broad coverage semantic
dependency parsing, it only includes English data in the DM target representation (the other 
representations are restrictively licensed).

### Semantic Dependency Tasks

There are two tasks we explore with this data:

1. Semantic Dependency Arc Prediction

**Why is this task interesting?**

This enables us to evaluate whether ELMo representations can determine whether two tokens are semantically related. We'd hope that the representations would capture this knowledge, since understanding of how different tokens are semantically connected would help in overall comprehension.

2. Semantic Dependency Arc Classification

**Why is this task interesting?**

This enables us to evaluate whether ELMo encodes specific semantic relationships between tokens.
