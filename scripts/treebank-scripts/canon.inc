#!/usr/local/bin/perl 

# Include file.
# Author: Jason Eisner, University of Pennsylvania


# Throw away most subscripts or postmodifiers from a tag, because
# they don't affect the choice of head.  (We call this on tags once as we
# read them in, not later on when we pick heads etc.)
#
# There are certain type-changing postmodifiers on S, NP that we do keep,
#   because the correct mark may depend on whether the postmodifier is present
#   or not.  We don't want to equivalence-class rules that have different heads:
#   that's bad for predicthead, and also bad for the human annotators.
#
# We also keep any ~ argument marks, which follow the initial segment of a tag.

sub canonicalizetag {
    local($_) = @_;

    # Regexps below assume that relevant portions of the tag
    # are uppercase.  Sanity check that we weren't handed some
    # lowercased version of the tag: lowercase should only appear in 
    # a context like :pound: (certain special symbols are converted to
    # this form by oneline).

    die "$0: canonicalizetag: argument $_ must be uppercase; stopped" if /[a-z]/ && (local($s)=$_, $s =~ s/:[a-z]*://g, $s =~ /[a-z]/);

    # occasionally numeric indices are marked with = rather than - (this is for gapping).
    # change that so that we handle all indices and tags identically.
    s/=/-/g;  

    # change NP-TTL to NPR.
    s/NP-TTL/NPR/;
    s/NP~-TTL/NPR~/;

    # Disguise the subscripts we want to keep, by temporarily changing - to (, which we know
    # can't appear.
    # Also disguise an initial - (for -NONE-).
    s/-(TMP|LOC|ADV|PRD)/($1/g;              
    s/(S(BAR)?Q?)-(NOM|SBJ|PRP|TPC)/$1($3/g;
    s/^-/(/;

    # now throw away any (non-empty) subscripts that start with - rather than (.
    s/-[^-(]+//g;         

    # change ( back to -.
    s/\(/-/g;

    $_;
}

1;
