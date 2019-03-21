#!/usr/local/bin/perl

# modify $0 for nicer error messages, 
# add the program's runtime directory to @INC,
# and return the program's runtime directory.

sub fixprog {
  if ($0 =~ s|.*/||) {
    unshift(@INC, $&);      # prepend to @INC the directory in which this program is running (if not ".")
    $&;
  } else {
    "";
  }
}


# print a timestamp line.
# also call fixprog, which does some nice things.
# returns the timestamp line in case it needs to be printed somewhere else.

sub stamp {
  local($progdir) = &fixprog;
  local($answer) = "";
  local(@args) = @ARGV;

  foreach (@args) {
    $_ = "'$_'" if /[;&()|<>\#?* \t\n"'`]/;   # quote args that contain special chars (not foolproof!)
  }

  local($currdate) = &localtimestring(time);
  $answer .= "# created $currdate by \"" . join(" ",$0,@args) . "\"";   # comment line

  local($workingdir) = `pwd`;
  chop($workingdir);
  $answer .= "\t[working dir: $workingdir]";

  $answer .= "\t[program dir: $progdir]" if $progdir;

  local($ver);
  if (defined $RCS_revision) {    # main file should set $RCS_revision = '$Revision: 1.4 $ ' before calling us; such a line will be munged by RCS to reflect correct version number.  (Initial and final whitespace in this string are optional; a final space is helpful to prevent Emacs cperl-mode from thinking it saw the token $' and getting confused.)
     local($rev) = $RCS_revision;
     local($dollar) = '\$';       # protect pattern below from RCS substitution
     $rev =~ m/^\s*${dollar}Revision: (.*) ${dollar}\s*$/ || die "$0: \$RCS_revision has bad format";
     $ver = "RCS $1";
  }
  if (local($dev,$ino,$mode,$nlink,$uid,$gid,$rdev,$size, 
            $atime,$mtime,$ctime,$blksize,$blocks)=stat($progdir.$0)) {  # if success
    local($progdate) = &localtimestring($mtime);
    if (defined $ver) { $ver .= " of " . $progdate; }
    else { $ver = $progdate; }
  }
  $answer .= "\t[program version: $ver]" if defined $ver;

  $answer .= "\n";
  print $answer;
  return $answer;
}


# Cite the timestamps and other leading comments for all files whose names are
# passed in as arguments.  (Any args that are not readable files are ignored, so 
# it's okay to pass @ARGV in.)  Note that stdin's leading comments can't be cited.

sub citestamps {
  local(@fnames) = @_;
  foreach $fname (@fnames) {
    local($|) = 1;		# autoflush
    if (-f $fname && -r $fname && open(ARGFILE,$fname)) { # if this arg is an ordinary file and we can open it
      # print its leading comments (initial comment lines without locations)
      while (defined($_ = <ARGFILE>) && /^\#/) {
	print "# $0: $fname: $_";
      }
      close(ARGFILE);
    }
  }
}



sub localtimestring {
  if ($] >= 5) {		# perl version 5 or later
    scalar localtime($_[0]);
  } else {
    `perl5 -e "print scalar(localtime($_[0]))"`;
  }
}

1;
