import csv
from backend.db.db import connect, execute_values_wrap
from backend.lib.setup import db_setup
from backend.settings import ORTHOGRAPHIC_FILE


def main():
    target_file = ORTHOGRAPHIC_FILE
    process_data(target_file)


# https://www.neuro.mcw.edu/mcword/definitions.html
# Number of Letters:
#
# LEN: This is simply the number of letters within the string.
# Frequency of Orthographic Form:
#
# FREQ: Frequency is a measure of how often a wordform is encountered in 1,000,000 presentations of text. It is based on the raw frequency of the string in the modified CELEX database (see the main page), and converted to frequency per million. If a string has a frequency of 0, then it is not in the CELEX database.
# Orthographic Neighborhood Statistics (Coltheart's N):
#
# Orth: This is the number of orthographic neighbors that a string has. An orthographic neighbor is defined as a word of the same length that differs from the original string by only one letter. For example, given the word 'cat', the words 'bat', 'fat', 'mat', 'cab', etc. are considered orthographic neighbors.
# Orth_F: This is the averaged frequency (per million) of the orthographic neighbors.
# Constrained Unigram Statistics:
#
# N1_F: This is the averaged frequency (per million) of the constrained unigrams for the wordform. A constrained unigram is defined as a specific letter in a specific position, in a specific length of word. That is, the 'c' in 'cat' is considered the same 'c' as in 'cot', but not in 'act', or 'catch'.
# N1_C: This is a count of the number of wordforms that share the same constrained unigrams (analogous to a Coltheart N measure).
# Constrained Bigram Statistics:
#
# N2_F: This is the averaged frequency (per million) of the constrained bigrams for the wordform. A constrained bigram is defined as a specific two letter combination (bigram) in a specific position, in a specific length of word. That is, the 'ca' in 'cat' is considered the same 'ca' as in 'can', but not in 'act' ('ac' is different than 'ca'), or 'catch'. Note that single letters do not have bigram statistics, and therefore are listed as NA for this measure.
# N2_C: This is a count of the number of wordforms that share the same constrained bigrams (analogous to a Coltheart N measure).
# Constrained Trigram Statistics:
#
# N3_F: This is the averaged frequency (per million) of the constrained trigrams for the wordform. A constrained trigram is defined as a specific three letter combination (trigram) in a specific position, in a specific length of word. That is, the 'sta' in 'stage' is considered the same 'sta' as in 'staff', but is different from the trigram in 'stay' . Note that single letters and two letter stings do not have trigram statistics, and therefore are listed as NA for this measure.
# N3_C: This is a count of the number of wordforms that share the same constrained trigrams (analogous to a Coltheart N measure).
# Unconstrained Unigram Statistics:
#
# UN1_F: This is the averaged frequency (per million) of the unconstrained unigrams for the wordform. An unconstrained unigram is defined as a specific letter within a word, regardless of its position, or the wordlength.
# UN1_C: This is a count of the number of wordforms that share the same unigrams (analogous to a Coltheart N measure).
# Unconstrained Bigram Statistics:
#
# UN2_F: This is the averaged frequency (per million) of the constrained bigrams for the wordform. An unconstrained bigram is defined as a specific two letter combination (bigram) within a word, regardless of its position, or the wordlength. For example, the 'ba' in 'bat' is considered the same as the 'ba' in 'tabasco'. Note that single letters do not have bigram statistics, and therefore are listed as NA for this measure.
# UN2_C: This is a count of the number of wordforms that share the same bigrams (analogous to a Coltheart N measure).
# Unconstrained Trigram Statistics:
#
# UN3_F: This is the averaged frequency (per million) of the constrained trigrams for the wordform. An unconstrained trigram is defined as a specific three letter combination (trigram) within a word, regardless of its position, or the wordlength. For example, the trigram 'sta' in 'stand' is considered the same as the trigram 'sta' in 'standard' and in 'estate'. Note that single letters and two letter stings do not have trigram statistics, and therefore are listed as NA for this measure.
# UN3_C: This is a count of the number of wordforms that share the same trigrams (analogous to a Coltheart N measure).

def process_data(target_file):
    with open(target_file, 'r') as f:
        reader = csv.DictReader(f)
        # headers
        # STRING	LEN	FREQ	Orth	Orth_F	N1_F	N1_C	N2_F	N2_C	N3_F	N3_C	UN1_F	UN1_C	UN2_F	UN2_C	UN3_F	UN3_C
        records = []
        for row in reader:
            record = {}
            record['word'] = replace_NA(row['STRING']).lower()
            record['frequency'] = replace_NA(row['FREQ'])
            record['ortho_neighbours'] = replace_NA(row['Orth'])
            record['ortho_frequency'] = replace_NA(row['Orth_F'])
            record['unigram_freq'] = replace_NA(row['N1_F'])
            record['unigram_count'] = replace_NA(row['N1_C'])
            record['bigram_freq'] = replace_NA(row['N2_F'])
            record['bigram_count'] = replace_NA(row['N2_C'])
            record['trigram_freq'] = replace_NA(row['N3_F'])
            record['trigram_count'] = replace_NA(row['N3_C'])
            record['unconstrained_unigram_freq'] = replace_NA(row['UN1_F'])
            record['unconstrained_unigram_count'] = replace_NA(row['UN1_C'])
            record['unconstrained_bigram_freq'] = replace_NA(row['UN2_F'])
            record['unconstrained_bigram_count'] = replace_NA(row['UN2_C'])
            record['unconstrained_trigram_freq'] = replace_NA(row['UN3_F'])
            record['unconstrained_trigram_count'] = replace_NA(row['UN3_C'])
            records.append(record)
    db_setup(orthographic=True)
    save_orthographic_records(records)


def replace_NA(item):
    if item == 'NA':
        return None
    return item


def save_orthographic_records(records):
    fieldnametuple = tuple(records[0].keys())
    execute_values_wrap('orthographic', fieldnametuple, [list(x.values()) for x in records])


def setup():
    main()


if __name__ == "__main__":
    setup()
