/* Author: Jason Eisner, Univ. of Pennsylvania
           Only lightly adapted from Anoop Sarkar's morphit
           (/mnt/linc/xtag/work/anoop/tagp/src/db.src/morphit.c) 
           Uses PC-KIMMO's hashed morphological database, from U. Penn. */

/* To compile: gcc -o morphfilter -I/mnt/linc/xtag/morphdir/morph/hash -L/mnt/linc/xtag/morphdir/morph/lib morphfilter.c -lhash -lmorph -lm */

#include <stdio.h>
#ifdef BSD
#include <strings.h>
#else
#include <string.h>
#endif
#include <sys/file.h>
#include <fcntl.h>
#include <db.h>
#include "morph.h"

main(argc, argv)
     int argc;
     char **argv;
{
  char	decoded_string[DATABUFFER];
  DBT entry, key;
  DB	*dbp;
  int i;
  char word[256];

  /* Check arguments */
  if (argc != 2)
    {
      fprintf(stderr,"Usage: %s <database file>

A filter that inputs one word per line, and outputs a corresponding line
that contains a #-delimited list of morphological analyses.
This list may be empty if the word is unknown.  Case-sensitive.
Some .db database files are in /mnt/linc/xtag/morphdir/morph/data.\n", argv[0]);
      exit(1);
    }

  /* Open DB file */
  if (!(dbp = hash_open( argv[1], O_RDONLY, 0440, NULL ))) 
    {
      fprintf( stderr, "%s: cannot access %s\n", argv[0], argv[2] );
      exit(1);
    }

  /* Main loop (read input line, look up in hash table, print output) */
 
  while ((i = (word[0]='\0', scanf("%255[^\n]", word))) != EOF) {     /* the word[0]='\0' is in case scanf fails, which will happen on a blank line */
    while (getchar() != '\n');   /* clear out rest of line, usually just \n */
    key.data = word;
    key.size = strlen(word)+1;
    decoded_string[0] = '\0';
    /* Retrieve key from database */
    if ((dbp->get)(dbp, &key, &entry, 0)) 
      {
	printf("\n");   /* not found */
      }
    else
      {
	/* decode the encoded entries */
	decode(word, entry.data, entry.size, decoded_string);
	printf ( "%s\n",  decoded_string);
      }
    fflush(stdout);   /* because we want other programs to be able to run us as an oracle, giving us a question and getting an immediate answer */
  }

  (dbp->close)(dbp);
  exit(0);
}
