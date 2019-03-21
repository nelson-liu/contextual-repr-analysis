#!/usr/local/bin/perl 

# Include file.
# Author: Jason Eisner, University of Pennsylvania

# Given a rule in vanilla array form (the 0th element contains the LHS
# tag, and the rest contain the canonicalized, ~-free RHS tags, all in
# uppercase with no marks or spaces), predict the head from general
# principles.  This prediction may be overridden later by specific
# rules we know about.
# 
# !!! Might want to extend this by allowing this to generalize
# directly from examples we know about.
#
# Returns "" if no prediction,
# otherwise a pair (num, sure) where num >= 1 is the number of
# the kid, and sure is a flag that says we're so sure of this decision
# that the user doesn't even need to review it.


sub predicthead {
    local($answer);

    # # make sure the tags really are canonical.
    # require("canon.inc");
    # foreach $tag (@_) { 
    # 	die "$0: predicthead got noncanonical tag as input, stopped" unless $tag eq &canonicalizetag($tag);
    # 	die "$0: predicthead got argmarked tag as input, stopped" if $tag =~ /~/;
    # }

    if (2==@_) {                                              # only one RHS element
	(1,1);
    } elsif (1==grep(/[A-Z\$\#]/ && !/^-/, @_[1..$#_])) {     # only one overt, non-punctuation RHS element (in particular we don't count -NONE-, -LRB-, -RRB-; we do count $ and # signs)
	for ($answer=1; !($_[$answer] =~ /[A-Z]/ && $_[$answer] !~ /^-/); $answer++) {}    # find that element
	($answer, 1);

    } elsif ($answer = &adjunct(@_)) {     # LHS appears exactly once on RHS, and is not NP
       ($answer, 1);

    } elsif (($_[0] !~ /^S|VP/) && grep(/,|:|-LRB-/, @_) && ($answer = &standardhead(1,@_))) {   
                                                # we can find a likely head in RHS (e.g., NN where LHS=NP);
	                                        # first try only looking before first comma (really we should add more commas one at a time!!!)
	                                        # we don't do this step if LHS is S, since sentences often have "CLAUSE ," adjoined to front or contain parentheticals like "he says."  VP sometimes also contains parenthetical.
	                                        # we also don't do this step unless there is a comma on the RHS somewhere.
       ($answer, 0);
    } elsif ($answer = &standardhead(0,@_)) {   # we can find a likely head in RHS, now looking at whole RHS
       ($answer, 0);
    }   
}


# does LHS appear exactly once on RHS?  If so, where?
# Input and output are as in predicthead.

sub adjunct {
    local(@rule) = @_;   # make local copy so we can modify its elements
    local($answer, $i);

    $rule[0] =~ s/(.)-.+/$1/g;                    # locally get rid of remaining subscripts
    for ($i=1; $i <= $#rule; $i++) {
        $rule[$i] =~ s/(.)-.+/$1/g;
	if ($rule[$i] eq $rule[0]) {
	    $answer = "", last if $answer;    # duplicates are bad
	    $answer = $i;                     # otherwise, remember which kid it was
	}
    }
    $answer if $rule[0] !~ /^(NP|PP|S|VP|QP)/;   
    # The hedge here for the categories NP, PP, S, VP, QP is necessary for
    # the following reasons: 
    # NPs can be used internal to other NPs without being adjuncts, as in
    # "The Rhode Island storeowner".  Likewise PPs, e.g., "compared with
    # last year".  Likewise S's -- especially in "x, said y"
    # constructions.  Likewise verbs can take VP complements to form VPs.
    # Finally, QPs may contain other QPs, in that the Treebank's (QP (CD
    # 8) (CD 11\/16) (NN %)) is rebracketed by us as (QP (QP (CD 8) (CD
    # 11\/16)) (NN %)).
}
	    

# Is there exactly one "natural" head on the RHS, given the LHS?  If so, where?
#
# Input and output are as in predicthead, except that there is now an additional
# input argument (the first), which acts as a flag saying to consider only the
# part of the RHS before the first comma or colon.

sub standardhead {
    local($beforecomma) = shift(@_);
    local($LHS) = " " . shift(@_) . " ";
    local(@rr) = @_;          # copy this because otherwise any modifications to the scalars in @_ 
                              #   (e.g., s/^INTJ/ITJN/ below) are seen back at the caller

    if ($beforecomma) {
       # ignore any part of RHS from first comma or colon on, not counting 
       # very first tag (so start counting from 1).  Do this by changing length
       # of array.

       for ($i=1; $i<=$#rr; $i++) {              # ignore any comma at index 0
   	  $#rr=$i-1, last if $rr[$i] =~ "^(,|:|-LRB-)\$";
       }
    }

    # locally change the interjection tag, INTJ, to ITJN so that
    # it doesn't interfere with prepositional tags starting with IN, like
    # IN|RB.
    #
    # Also, anything on the RHS that has -NOM or -SBJ should count as an NP 
    # for present purposes.
    foreach $tag (@rr) {          
       $tag =~ s/^INTJ/ITJN/;
       $tag =~ s/^[^-]*/NP/ if $tag =~ s/-(NOM|SBJ)//g;   # delete any NOM or SBJ, and change initial category to NP if so
    }

    # In examples like "section 89" (and "may 31"), we locally jimmy the
    # QP following the nominal (turning it to POSTQP) so that the NPR can win.  However, phrases
    # like this are always determinerless; we're careful not to do
    # this if there's a determiner, since these are usually phrases
    # like "the big three," "the s&p 500" -- (though occasionally:
    # "the TI Model 12")
    $rr[$#rr] =~ s/^/POST/  if ($#rr >= 1 && $rr[0] ne "DT" && $rr[$#rr] eq "QP" && $rr[$#rr-1] =~ /^(NN|NPR)$/);
    $rr[$#rr-1] =~ s/^/POST/ if ($#rr >= 2 && $rr[0] ne "DT" && $rr[$#rr-1] eq "QP" && $rr[$#rr-2] =~ /^(NN|NPR)$/ && $rr[$#rr] =~ /^(POS|\W+)$/);

    # final NPR or NPR POS immediately after NN... or NP (or separated only by punctuation) 
    # is virtually always an appositive; 
    # locally jimmy the NPR (turning it to APPOSNPR) so that it doesn't get mistaken for the head.
    # In fact, even non-final NPR is an appositive after NN, so long as it's the final nominal.  We jimmy all such
    # cases, since if it's NOT the final nominal it may not be an appositive but it certainly won't be the head
    # (and the head rules will know that regardless of whether we do the jimmying).
    for ($i=$#rr; $i >= 1; $i--) {
       $rr[$i] =~ s/^(NP|NX|TTL$)/APPOS$1/ if $rr[$i-1] =~ /^(NN|NP$)/ 
                                                 || ($rr[$i-2] =~ /^(NN|NP$)/ && $rr[$i-1] !~ /A-Z/);
    }
   

    # change -lrb-, -rrb- to parens so we know they're punctuation.
    foreach (@rr) { s/-[LR]RB-/(/; }

    local($_) = " " . join("  ",@rr) . " ";  # make RHS into string, using double space 
                                             # in between so that patterns below with space around 
                                             #   can match successive tags
    # see if RHS has exactly one match to the appropriate pattern for the LHS
    # can give several RHS patterns (in order of precedence) for same LHS
    # use a space as tag boundary in the search patterns, if desired.
    # use "^ " or " $" to match first or last tag
    # to match first or last occurrence in tag sequence, don't require s to return 1;
    #    omit g flag for first and include it for last; careful about return value (to require multiple, usually force 1<s/.../ 
    #     but if g flag absent, may need a conjunction with a subsequent match, say)

    if (   
           $LHS =~ / ADJP/  &&(1==s/ NN $/$&/g                # note that final singular noun will end an ADJP
        ||		       1==s/ \$ | \# /$&/g 
        ||		       1==s/ JJ| QP| VBN | VBG | ADJP/$&/g # normal headwords for ADJP.  
        ||		       1==s/ JJ| VBN | VBG /$&/g     
        ||		       s/ QP/$&/g                    # try last QP or QPMONEY if there's more than one (I think this deals with bracketing errors?)
      # ||		       1==s/ RB/$&/g                some treebank errors could be saved this way ...
                              )

        || $LHS =~ / ADVP/  &&(1==s/^ IN /$&/g                # initial preposition wins; but note "down here"
        ||		       1==s/ RB| RP | VBG | JJ| ADVP/$&/g   # note "slightly lower", "as low" -- should favor JJ
        ||		       1< s/ RB/$&/g                  # last RB if there are multiple ones ("as slow", "very definitely"; but note "up/down slightly", "somewhere else")
                              )
	                                            # for "early in October", "prior to the elections," etc., we'll make first word the head by hand.  
	                                            # ditto "late yesterday."

        || $LHS =~ / PP| WHPP| NP-LOC/     && 1==s/ IN | TO /$&/g     # PP headed by preposition  (don't do this for NP-TMP because of the common error "next/IN march 31")
        || $LHS =~ / PP/     &&(1==s/ VBN | VBG /$&/g            # other possible heads for PP
        ||			1==s/ VBN | VBG | PP/$&/g        
        ||			1==s/ VBN | VBG | PP| RB /$&/g   
                               )

        || $LHS =~ / QPMONEY/ && 1==s/ \$ | \# /$&/        # explicit dollar figures headed by dollar sign                        
                                                               
        || $LHS =~ / NP\$/ && s/ POS/$&/g                    # possessive NPs are headed by the possessive morpheme, assuming there is one (there should be!!)
        || $LHS =~ / NP/ && /^ QP  (RB |NPR $)/ && /^ QP/    # initial "6 p.m. eastern time" or "6:00 edt" is headed by 6.  Note that we only allow QP, not QPMONEY, and only allow singular NPR, not NPRS (though this is a little dangerous even so - how about something like "[NP [QP only about one] [NPR Cadillac]]" -- also get "one Charles Doolittle Walcott")
                                                               
        || $LHS =~ / NP| QP| NAC| NX | WHNP/ && s/ NN| NP | NPR| NX| NAC |-NOM |TTL | PRP | CD | PDT | QP/$&/g
                                                               # last nominal gets to head a noun phrase or QP
        || $LHS =~ / NP| QP| NAC| NX/ && s/ JJ| ADJP | PRP$| FW | DT | NP\$ | RB/$&/g && !$beforecomma     # else try last JJ  (the biggest, the first, etc.) or foreign word or DT/PDT ("that", "all", "half").  (We do prefer nouns to adjs, as in "ATTORNEY general").  But don't take this fallback position if we're only considering pre-comma stuff.

        || $LHS =~ / S/ && $LHS !~ / SBAR/ &&(1==s/ VP /$&/g    # !!! this whole sequence thing is a little off.  should continue to next conjunct only if no match at all, and throw up our hands if there are 2 matches.  
        ||				      1==s/-PRD /$&/g    
        ||				      1==s/ VB| MD/$&/g    
        ||				      1==s/ S/$&/g    
					     )

        || $LHS =~ / SBAR/   &&(1==s/ IN| PP| DT| W/$&/g         
        ||			/ IN | RB| S|WH/      # first of these wins (note double prep cases like "as though" and "so that")
                               )

        || $LHS =~ / VP/   &&(1==s/ VB[-A-Z]*  N/$&/g  && 1==s/ VB/$&/g  
                                                     # except when VB is followed by a noun,  
                                                     #         (note rematch to ensure only a single VB and to leave $& set properly)
        ||		      1==s/ VP/$&/g          # prefer VP to VB because VB is often just auxiliary and shouldn't be head.  (the cases where VB is followed by a noun cover many of the cases where the VP is not the head, but just a complement to the VB)
        ||		      1==s/ V/$&/g           # take any V-like thing we can get.  Occasionally, 
        ||		      1==s/ JJ/$&/g         
        ||		      1==s/ S/$&/g         
                             )

        || $LHS =~ / W/   && 1==s/ W/$&/g         

	|| s/ FW /$&/g       # as last resort, if there are foreign words, try last one
       ) {
        # print "[[$`][$&]]";
	local($match) = $` . $&;  # RHS up through matched portion
	$match =~ s/ *$//;        # kill final spaces of match, if any
	$match =~ s/ +//g;        # return number of tags in $match (= number of pre-tag space blocks)
    } 
}        



1;
