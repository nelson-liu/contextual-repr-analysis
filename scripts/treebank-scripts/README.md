# treebank-scripts
Suite of scripts for preprocessing the Penn Treebank, primarily to extract lexical subcategorization frames and dependencies. 

Author: Jason Eisner <jason@cs.jhu.edu>

These scripts are filters that process the Penn Treebank.  They can be pipelined together in various combinations.  Their main purpose is to extract lexical subcategorization frames or lexical dependencies from the Penn Treebank.  In particular, they can mark the head child of a constituent.  They also convert empty categories into slashed nonterminals.

## Overview

The scripts read and write files in a common format.  To convert from the original Penn Treebank to this format, use the oneline script.  Some features of the format:

- one sentence or rule per line
- there is a `prettyprint` script available
- a line optionally begins with a location string (filename:linenumber:)
   indicating where it came from in the original Penn Treebank
- comment lines start with ``# and are inserted automatically during processing
- the file begins with a block of automatic comments that explain how the
   file was prepared; each new filter adds a comment to the top of this block
- certain special characters have reserved meanings; instances of these
    characters in the original Treebank are transformed; see the oneline
    script for documentation of this

## Usage

Each script is documented through initial comments.  For an explanation of how some of these scripts are pipelined in a typical case, see section 6.2 ("Data Preparation") of [Jason Eisner's Ph.D. thesis](http://cs.jhu.edu/~jason/papers/#eisner-2001-thesis).

If you plan to run the scripts from the command line, then you may
want to add the script directory to your `PATH` environment variable.
The `stamp.inc` script needs to be in a directory that is listed in
the `PERL5LIB` or `PERLLIB` environment variable.

Alternatively, you can run the scripts by invoking Perl directly, e.g.,

	perl -I/path/to/treebank-scripts /path/to/treebank-scripts/oneline

## Documentation Files

* `README.md`: this file

* `HOW-TO.txt`: a sample pipeline you may want to check out

* `output`: a directory with some sample output (created by Martin Cmejrek)

* `SLASH-AND-PLUS.txt`: discussion related to the `slashnulls` script, which transforms empty categories into a GPSG-style notation

* `MULTI-ROLES.txt`: some notes from Jason to himself on the interaction of bilexical probabilities with gaps

## Scripts

* `addcomment`: add a human-written comment at the top of one or more data files

* `articulate`: make the Treebank structure less flat.  also automatically corrects some simple, common annotator errors.

* `artic.inc`: the rules used by `articulate`

* `binarize`: ensure that no node has more than 2 children

* `canonicalize`: simplify the nonterminal tags

* `canon.inc`: the rules used by `canonicalize`

* `canonindices`: renumber the coindices on traces

* `commentsentids`: like `striplocations`, but moves the location into a comment; can be undone by `mergesentenceidssents` (by Martin Cmejrek)

* `discardbugs`: discard sentences that appear to contain annotation errors

* `discardconj`: discard sentences that contain conjunctions

* `discardsingletons`: discard singletons from a list of dependency frames

* `do_all_steps`: something uncommented, by Martin Cmejrek

* `fixsay`: fixes an odd annotation convention in the Treebank that interferes with `slashnulls`

* `flat2dep`: converts the output of `flatten` into a different dependency parse format that works with some of Jason's other code (including a dependency parse viewer/editor in Emacs)

* `flatten`: turns headed parses (output of `headify`) into dependency-like parses

* `flatten.adj`: appears to be a obsolete version of `flatten`, but with one extra feature (`-a` option to mark adjuncts specially)

* `fringe`: turns a tree back into a word sequence

* `headall`: ensure that an incompletely headed corpus is fully headed (by discarding sentences or making a last-resort guess of the head)

* `headify`: mark the head subconsituent of each constituent

* `killnulls`: removes phonologically empty constituents

* `killpunc`: removes punctuation

* `listrules`: lists all the phrase-structure rules used in a parsed corpus

* `markargs`: replicates Collins (1997)'s rules for distinguishing arguments from adjuncts; marks the arguments

* `mergesentenceidssents`: undoes the effect of `commentsentids` (by Martin Cmejrek)

* `moreknobs`: can be used to adjust the output of `slashnulls`

* `morph*`: used by `taggedmorphfilter`

* `nobadnonterm`: removes test sentences or rules that mention a nonterminal not appearing in training data

* `normcase`: heuristically normalizes the case (uppercase, lowercase ...) of words, perhaps limited to sentence-initial words

* `oneline`: converts from Treebank format to the format assumed by these scripts; reversed by `prettyprint`

* `predict.inc`: the head prediction rules (used by `headify`)

* `prefixcounts`: count occurrences of each rule in the corpus (similar to `uniq -c` in Unix, but works with our format)

* `prettyprint`: prettyprints a corpus that is in the format we use

* `rootify`: wraps every tree in `(ROOT ...)`

* `rules2frames`: turns a list of headed rules (produced by `listrules`) into a list of dependency frames

* `selectsect`: selects out only the sentences from a particular section of the Treebank

* `slashnulls`: converts parses from using traces to using slashed categories as in GPSG; see `SLASH-AND-PLUS.txt` for discussion of why slashes weren't quite enough

* `stamp.inc`: used to create the automatic comments at the top of output files

* `stripall`: concatenates files and passes them through striplocations and stripcomments

* `stripcomments`: removes comments (`# ...`) 

* `striplocations`: removes the location string (`filename:linenumber:`) from the start of each line

* `summarize`: gives simple statistics about the output of `rules2frames`

* `swapwords`: used to prepare data for a forced-disambiguation task

* `taggedmorphfilter`: morphologizes words?

## Data Files

* `newmarked.mrk`: some rule head annotations produced by a human, either confirming or overriding an automatic annotation. Pass this file to `headify`, which will use it as an exception list.

* `newmarked_coord.mrk`: looks like someone (Martin?) dumped out all the rules involving conjunctions, and marked the conjunction as the head child, as a cheap way to avoid having to use `discardconj`.

* `newmarked.bug`: lists some rules that appear to indicate Treebank annotation errors; these were flagged during head annotation.  Pass this file to `discardbugs`.

## History

These materials are primarily by Jason Eisner <jason@cs.jhu.edu>, with
some later improvements by Martin Cmejrek <cmejrek@ufal.ms.mff.cuni.cz>.

* A number of people have requested the materials over the years.  For many years, Jason distributed these files on request as `wsj_add_heads.tar`.  In 2016, he put them on github at another researcher's suggestion, and converted the `TO-DO` file to issues on the github issue tracker.

* In 2002, Martin made minor updates, primarily to make sure the scripts all ran in Perl 5.  (Some of the scripts had been originally writtten in Perl 4 since that was the default on the system Jason used at the time.)

* The scripts were written by Jason in 1998 or so.  They were used in Jason's Ph.D. thesis and several subsequent projects by others.

* The head rules (`predict.inc`), and head exception lists (`newmarked.mrk`, `newmarked.bug`) were developed earlier, in 1995.  At the time, they were used to prepare data for Jason's 1996 papers on dependency parsing.  They were developed using an Emacs-based head-annotation environment also written by Jason; that environment is not currently included here, but could be made available on request.

    The articulation rules (`artic.inc`) impose some structure on subtrees that the Penn Treebank leaves flat.  Jason recalls that these were developed after the head rules.  He didn't make much effort to modify the head rules to work with the articulated structure, but they should continue to work in most respects.
