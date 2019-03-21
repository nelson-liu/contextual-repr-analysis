#!/usr/local/bin/perl 

# Include file.
# Author: Jason Eisner, University of Pennsylvania

# Parse a treebank rule from the bottom up, yielding at least one
# constituent.  We say "parse" because some treebank rules are too
# flat (either by design or by annotator error): we may add some more
# structure here, as follows.  For each constituent we find associated
# with this rule, including the top-level constituent, we call
# $reducerulefn, which reduces a specified subsequence of the RHS to a
# single nonterminal and takes some appropriate action associated with the
# discovery of that constituent, like printing it out.
# 
# $LHS and @RHS are passed by reference so that they can be changed.
# The contents of *RHSaux are up to the caller.  It will be passed on
# to $reducerulefn, which can change it; the idea is that @RHSaux 
# is a parallel array to @RHS, and $reducerulefn will make parallel changes.
#
# $LHS is generally a raw tag as read from the corpus -- it might not
#    even be canonicalized.  This means that we have to be careful when
#    checking and modifying it -- e.g., when modifying, we should leave
#    qualifiers and indices alone.  
# @RHS consists of more processed tags -- the LHS values that reducerulefn
#    produced for the subconstituents.
# Any of these tags may include ~ marks.

# they might not even be canonicalized.  Of course, if subseqs of 
# @RHS are reduced to a single tag, that tag is specified within
# this routine.
# 
# Returns total number of constituents found (i.e., reduced).

sub parserule {
    local($reducerulefn, *LHS, *RHS, *RHSaux) = @_;
    local($constits, $j);

    # replace maximal sequences of proper nouns NNP<suffix> with 
    # the proper noun phrase tag NPR<suffix>
    # The <suffix> is copied from the final tag in the sequence; it could be 
    # anything, but is usually either empty or S, to distinguish between
    # singular and plural proper nouns.
    for($i = $#RHS; $i >= 0; $i--) {
	if ($RHS[$i] =~ /^NNP/) {
            local($label) = $RHS[$i];  $label =~ s/^NNP/NPR/;  
	    $j = $i;          
            $i-- while $i-1 >=0 && $RHS[$i-1] =~ /^NNP/;
	    $constits += &callreducerulefn(@_, $i, $j, $label);
	}
    }

    # replace maximal sequences of CD with QP.  
    for($i = $#RHS; $i >= 0; $i--) {
	if ($RHS[$i] =~ /^CD/) {
	    $j = $i;          
            $i-- while $i-1 >= 0 && $RHS[$i-1] =~ /^CD/; 
	    $constits += &callreducerulefn(@_, $i, $j, "QP");
	}
    }

    # replace $ QP with QPMONEY.  The Treebank rules (frequently violated)
    # say that "$ 2.2 million" should be a single flat QP, but that prevents
    # us from getting the desired dependency structure, which is 2.2 -> million -> $,
    # i.e., million is the head of a number that is dependent on $.
    #
    # The reason to make the result a QPMONEY rather than another QP is just that 
    # the two might conceivably have different headedness rules (rare, but note
    # 6 EST has a different head than $ 60,000 BMW).

    for ($i = $#RHS; $i >= 1; $i--) {
	if ($RHS[$i-1] =~ /^(\$|\#)/ && $RHS[$i] =~ /^QP/) {
	    $constits += &$reducerulefn(*RHS, *RHSaux, $i-1, $i, "QPMONEY");
	}
    }


    # if the rule is now QP -> QPMONEY, as will happen if the annotators
    # remembered to bracket "$ 2.2 million" as a QP themselves, then
    # make it QPMONEY -> QPMONEY, i.e., project up the MONEY feature.   
    # (The resulting redundant unary rule will be ignored in &callreducerulefn, 
    # so we'll just get (QPMONEY ...) rather than (QPMONEY (QPMONEY ...)) or
    # (QP (QPMONEY ...)).
    #
    # More generally, we do this for any QP that appears to be headed by QPMONEY,
    # so we'll have (QPMONEY (QPMONEY $1 million) (IN to) (QPMONEY $2 million)),
    # (QPMONEY (IN at) (JJS least) (QPMONEY $110 million)), etc.

    if ($LHS =~ /^QP/) {
      require("predict.inc");             
      local($qphead, $sure) = &predicthead($LHS, @RHS);   # rather early call to predicthead!
      $LHS =~ s/^QP/QPMONEY/ if ($qphead > 0 && $RHS[$qphead-1] =~ /^QPMONEY/)   # -1 corrects for fact that predicthead returns index into ($LHS, @RHS)
    }


    # replace maximal sequences of NN immediately following NPR with NP
    # (This implementation may try to replace a sequence of length 0, but 
    # callreducerulefn will catch that.)
    for($i = $#RHS; $i >= 0; $i--) {
	if ($RHS[$i] =~ /^NPR/) {
	    for ($j=$i+1; $j <= $#RHS && $RHS[$j] =~ /^NN/; $j++) {}
    	    $constits += &callreducerulefn(@_, $i+1, $j-1, "NP");
	}
    }

    # Do something about conjunctions.  Not implemented yet.
    #
    # We will do something simple with conjunctions, so that the
    # output could be improved in principle by another script.  A
    # phrase like
    #   (NP light cars and trucks with big horns) 
    # will be articulated as 
    #   (NP (NP light cars) and (NP trucks with big horns)).  
    # Later on, "and" will be assigned as the head.
    # It is possible for a later script to change the choice of head, or to
    # decide to reattach "light" or "with big horns" so that it scopes
    # over the entire conjoined phrase.
    #
    # Multiple conjunctions are handled using left branching:
    #    (ADJP big , and bigger , and even bigger ,)
    # will become
    #    (ADJP (ADJP (ADJP big , ) and (ADJP bigger ,)) and (ADJP even bigger ,))
    # 
    # Lists with commas are handled by giving the conjunction more than 2 children:
    #    (ADJP big , bigger , and even bigger ,)
    # will become
    #    (ADJP (ADJP big , ) (ADJP bigger ,) and (ADJP even bigger ,))
    # Note that it is also possible for a conjunction to have only one child, e.g.,
    # if the conjuction is at the start or end of the phrase.

    # !!!!!!!!!!!!!!!!!!!! NEW CODE GOES HERE !!!!!!!!!!!!!!!!!!!!!!!
    # !!!! WE ALSO HAVE TO CHANGE PREDICT.INC TO MARK CONJ OR CONJP

    # within a noun phrase, any determiner or prep after the first nominal
    # starts a new NP or PP=(P NP).  (This will get the dispreferred reading for 
    # "two to three billion", though.)  This corrects a common family
    # of annotator bracketing errors.

    if ($LHS =~ /^(NP|CD|QP|NAC|NX|WHNP)/) {
	for ($j = 0; $j <= $#RHS && $RHS[$j] !~ /^NN|^NP$|^NP|^NX|^NAC|-NOM$|^PRP$|^CD|^QP/; $j++) {}  # find first nominal (maybe after end)
	for ($i = $#RHS; $i > $j; $i--) {
	    if ($RHS[$i] =~ /^(IN|TO)/) {
		$constits += &callreducerulefn(@_, $i+1, $#RHS, "NP");
		$constits += &callreducerulefn(@_, $i, $#RHS, "PP");
	    } elsif ($RHS[$i] =~ /^DT/) {
		$constits += &callreducerulefn(@_, $i, $#RHS, "NP");
	    }
	}
    }

    # likewise, within a PP, any determiner or prep after the first 
    # preposition or RB starts a new NP or PP.  This handles especially the case
    # because/IN of/IN taxes/NNS, but it might also get bracketing errors.

    if ($LHS =~ /^PP/) {    # this code is cribbed from above
	for ($j = 0; $j <= $#RHS && $RHS[$j] !~ /^IN/; $j++) {}  # find first preposition (maybe after end)
	for ($i = $#RHS; $i > $j; $i--) {
	    if ($RHS[$i] =~ /^(IN|TO|RB)/) {
		$constits += &callreducerulefn(@_, $i+1, $#RHS, "NP");
		$constits += &callreducerulefn(@_, $i, $#RHS, "PP");
	    } elsif ($RHS[$i] =~ /^DT/) {
		$constits += &callreducerulefn(@_, $i, $#RHS, "NP");
	    }
	}
    }

    # correct common tagging error: POS at the start of a phrase (especially a VP) is a mistagged 's  (possibly "us" but usually "is")
    if ($RHS[0] =~ /^POS/) {
       warn "$0: Fixing likely \"\'s\"-tagging error: $LHS -> @RHS starts with POS, changing to VBZ\n";
       $RHS[0] = "VBZ";
    }


    # anything before POS (the possessive morpheme) must be an NP, and
    # taken together with the POS, it's an NP$.  
    # We go left to right, despite the inconvenience, because we might
    # get something like "[[[Sally]'s] mother]'s]".
    for ($i = 1; $i <= $#RHS; $i++) {     # ignore case of POS at $i == 0  (which is usually an error -- a contraction 's = is or us, mistagged as a possessive)
        if ($RHS[$i] =~ /^POS/) {
             $constits += &callreducerulefn(@_, 0, $i-1, "NP"); # this always does something since we skipped $i==0.  
             $constits += &callreducerulefn(@_, 0, 1, "NP\$");   # replace the NP POS now at the start of RHS with NP$ 
             $i = 1;                                            # resume following the POS
        } 
    }

    # sometimes, something that should be an NP$ is misannotated an
    # NP.  If that's so, we should now have the rule NP -> NP$ (i.e.,
    # the RHS originally ended in POS).  In that case, project the
    # "$" feature up to get NP$ -> NP$ (which will be ignored by
    # callreducerulefn since it's vacuous).
    #
    # We are careful here to deal properly with the case where 
    # LHS 

    $LHS =~ s/^NP/NP\$/ if @RHS==1 && $RHS[0] =~ /^NP\$/;


    local ($c, @rest) = &callreducerulefn(@_, 0, $#RHS, $LHS);    # reduce the entire RHS to the LHS (which may have been modified)


# --- replace preceding line with this code if multiple passes might be required.  	    
#    f (!$constits) {       # if we haven't called $reducerulefn yet,
#        local ($c, @rest) = &callreducerulefn(@_, 0, $#RHS, $LHS);    # reduce the entire RHS to the LHS (unless they're already equal) and stop.
#    } else {
#        # try another pass.   
#        $constits += &parserule(*LHS, *RHS, *RHSaux) if $constits;
#    }
# --- replace preceding line with this code if multiple passes might be required.


    $constits += $c;
    ($constits, @rest);
}


# calls reducerulefn to replace the stretch of RHS between $start and
# $end with $replacement, unless this will have no effect on RHS
# (e.g., it just replaces CD with CD, or tries to replace an empty sequence).
# Returns the number of replacements made (0 or 1).

sub callreducerulefn {
    local($reducerulefn, *LHS, *RHS, *RHSaux, $start, $end, $replacement) = @_;
    if ($start > $end) {
        0;
    } elsif ($start == $end && $replacement eq $RHS[$start]) {  # would just replace one symbol with itself
	0;
    } else { 
	&$reducerulefn(*RHS, *RHSaux, $start, $end, $replacement);
	1;
    }
}

1;
