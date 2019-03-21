/*
 * Copyright 1992, 1993 The University of Pennsylvania
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose and without fee is hereby granted, provided
 * that the above copyright notice appear in all copies and that both that
 * copyright notice and this permission notice appear in supporting 
 * documentation, and that the name of U. of Pennsylvania not be used in 
 * advertising or publicity pertaining to distribution of the software without 
 * specific, written prior permission.  U. of Pennsylvania makes no 
 * representations about the suitability of this software for any purpose.  
 * It is provided "as is" without express or implied warranty.
 *
 * THE UNIVERSITY OF PENNSYLVANIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS 
 * SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, 
 * IN NO EVENT SHALL THE UNIVERSITY OF PENNSYLVANIA BE LIABLE FOR ANY SPECIAL, 
 * INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM 
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE.
 *
 */

#define VERSION_NUMBER "v1.4"
/* Information specific to the encoding and decoding of the inflections
   for the morphological databases.  */
#define CODE_ZERO 255    
/* when common length of the key and content is 0, */
/* this causes a slight problem with the encoded */
/* string, since strings are null-terminated.  We use */
/* 255 instead */
#define MAX_INFLECTIONS 5   
/* Maximum number of inflections, not counting part of */
/* speech, i.e. N, V, etc.  This is used for the sort_infl */
/* routine.  A very simple sort is used.  If the number of */
/* becomes quite larger (i.e. approx. 10 or more), a different */
/* sort algorithm should be used */
#define MAX_STD_INFL 5
/* Maximum number of standard inflections that can be generated */
/* for a given root word within 1 type of part-of-speech */
#define A_START 1
#define A_END 3
#define ADV_START 6
#define ADV_END 7
#define PART 8
#define COMP 10
#define CONJ 11
#define INTERJ 13
#define PUNCT 14
#define PREP 15
#define D_START 16
#define D_END 26
#define N_START 31
#define N_END 38
#define PROPN_START 41
#define PROPN_END 44
#define V_START 46
#define V_END 92
#define NVC_START 95
#define NVC_END 117
#define VVC_START 119
#define VVC_END 120
#define PRON_START 121
#define PRON_END 162
#define END_INFLECTIONS 162

static char *inflections[] = 
{
"\0", /* This element should never be given an inflectional meaning.  See */
      /* delete_the_key() for more information. */
 
"A", "A COMP", "A SUPER", "\0", "\0",

"Adv","Adv wh", "Part", "\0", "Comp",

"Conj","\0", "I", "Punct", "Prep",

"Det", "Det wh", "Det GEN wh", "Det GEN ref1sg", "Det GEN ref2sg",
"Det GEN ref2nd", "Det GEN ref1pl", "Det GEN ref3pl", "Det GEN ref3sg reffem",
"Det GEN ref3sg refmasc", "Det GEN ref3sg", "\0", "\0", "\0", "\0",

"N 3pl", "N 3pl GEN", "N 3pl masc", "N 3pl GEN masc", "N 3sg", 
"N 3sg GEN", "N 3sg masc", "N 3sg GEN masc","\0", "\0", 

"PropN 3sg", "PropN 3sg GEN", "PropN 3pl", "PropN 3pl GEN", "\0", 

"V 1sg PAST STR", "V 1sg PRES", "V 2sg PAST STR", "V 2sg PRES", "V 3pl PRES",
"V 3sg PAST STR", "V 3sg PRES", "V INF", "V INF NEG", "V NEG PRES",
"V PAST", "V PAST PPART", "V PAST STR", "V PAST WK", "V PPART",
"V PASSIVE PPART STR", "V PPART STR", "V PPART WK", "V PPART STR to", "V PRES", "V INDAUX", 
"V PROG", "V PROG to", "V PAST STR pl", "V PRES pl","V NEG PRES", 
"V NEG PRES pl", "V 2sg NEG PRES", "V 3sg NEG PRES", "V NEG PAST STR","V NEG PAST STR pl",
"V 1sg NEG PAST STR", "V 2sg NEG PAST STR", "V 3sg NEG PAST STR", "V NEG PPART STR", "V CONTR INF NEG", 
"V CONTR NEG PRES", "V CONTR NEG PRES pl", "V 2sg CONTR NEG PRES", "V 3sg CONTR NEG PRES", "V CONTR NEG PAST STR", 
"V CONTR NEG PPART STR","V 1sg CONTR NEG PAST STR", "V 2sg CONTR NEG PAST STR", "V 3sg CONTR NEG PAST STR", "V CONTR NEG PAST STR pl",
"V TO", "\0", "\0", 

"NVC 3sg PAST", "NVC 3sg PAST wh", "NVC 1sg PRES", "NVC 1sg PAST", "NVC 1pl PRES", "NVC 1pl PAST", "NVC 2nd PRES", "NVC 2nd PAST", "NVC 3sg PRES", "NVC 3pl PRES",  "NVC 3pl PAST", "NVC 3sg PRES fem", "NVC 3sg PRES masc", "NVC 3sg PRES neut", "NVC 3sg PRES wh", "NVC 3sg PAST fem", "NVC 3sg PAST masc", "NVC 3pl PRES wh", "NVC 1sg INF", "NVC 1pl INF", "NVC 2nd INF", "NVC 3pl INF","NVC 3sg PAST STR neut", "\0",

"VVC PRES INF", "VVC PAST INF", 

"Pron", "Pron 1sg nom", "Pron 3pl", "Pron 3rd NEG nomacc", "Pron 3sg nomacc",
"Pron 3sg NEG nomacc", "Pron 3sg GEN nomacc", "Pron GEN ref1sg", "Pron GEN ref2nd", "Pron GEN ref1pl",
"Pron GEN ref2sg","Pron GEN ref3pl", "Pron GEN ref3sg reffem", "Pron GEN ref3sg refmasc","Pron GEN ref3sg", 
"Pron 3sg GEN NEG nomacc", "Pron 1pl acc", "Pron 1sg acc","Pron 2sg acc", "Pron 3pl acc", 
"Pron 3sg acc fem", "Pron 3sg acc masc", "Pron 3sg neut nomacc","Pron 1pl nom", "Pron 2sg nom", 
"Pron 3pl nom", "Pron 3sg fem nom", "Pron 3sg masc nom", "Pron 2nd nomacc", "Pron 2pl nomacc", 
"Pron 1pl refl", "Pron 1sg refl", "Pron 2pl refl", "Pron 2sg refl", "Pron 3pl refl", 
"Pron 3sg refl", "Pron 3sg fem refl", "Pron 3sg masc refl", "Pron 3sg neut refl", "Pron 3sg wh", 
"Pron 3sg acc wh", "Pron 3sg nom wh"
};


/*  Various globals needed for morphological and DB programs */

#define MAXCHARS   256
#define KEYBUFFER  64
#define DATABUFFER 256   /* Size of string buffers */
#define MAX_KEYS    8     /* Maximum number of inflected forms (ie, distinct */
                         /* keys) which a given Kimmo lexicon entry word can */
                         /* generate = 4 for nouns * 2 for hyphenated words  */
#define SEP_STRING "#"  /* Character separating entries in a stored database */
#define SEP_CHAR '#'
#define MAX_ENTRIES 6   /* Maximum # of entries allowed */

#define BUCKET_SIZE 1024 /* Size of buckets within hash table */
#define NUM_ENTRIES 300000 /* Estimate of number of entries in database */
#define FILL_FACTOR 128     /* >= BUCKET_SIZE/(Avg. key + avg. entry +4) */
#define CACHE_SIZE  1048576    /* 1 MEG */
/*
 * Definitions for byte order,
 * according to byte significance from low address to high.
 */
#define	LITTLE_ENDIAN	1234	/* least-significant byte first (vax) */
#define	BIG_ENDIAN	4321	/* most-significant byte first (IBM, net) */
#define	PDP_ENDIAN	3412	/* LSB first in word, MSW first in long (pdp) */
typedef struct {
  int word_count;
  int entry_count;
  int key_size;
  int content_size;
} stats_t;

