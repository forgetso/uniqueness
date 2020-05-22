import backend.settings as settings
import backend.db.db as db
from backend.lib.setup import db_setup
import csv
import argparse
import os
import pandas
import re


def process(wordassoc=None, responses=None, affix=None, norms=None, prevalences=None, concreteness=None, age=None,
            emotions=None):
    # TODO Merge all of the ugent files into one DB table. Emotions, age of acquisition, concreteness, prevalence, etc.

    db_setup(responses=responses, affixes=affix, norms=norms, prevalences=prevalences,
             concreteness=concreteness, age=age, emotions=emotions)
    if affix:
        affixes = []
        with open(settings.AFFIX_FILE, mode='r') as affixfile:
            reader = csv.DictReader(affixfile)
            for row in reader:
                row_affixes = row['Root, Prefix or Suffix'].replace(' ', '').split(',')
                for affix in row_affixes:
                    affix_type = 'prefix'
                    if '-' in affix:
                        affix_type = 'suffix'
                        affix = affix.replace('-', '')
                    affixes.append([affix, affix_type])

        db.execute_values_wrap('affixes', ('affix', 'type'), affixes)

    if norms:
        normsvalues = []
        readers = []

        for filename in os.listdir(settings.NORMS_FOLDER):
            if len(filename) > 1:
                with open('/'.join([settings.NORMS_FOLDER, filename]), mode='r', encoding='ISO-8859-1') as normsfile:
                    reader = pandas.read_csv(normsfile, skiprows=3, skipfooter=3, sep=", ")
                    reader = reader.replace(to_replace='¥', value='0')
                    readers.append(reader)

        reader = pandas.concat(readers)
        # we use two lists as execute_values does not work with dicts
        keys = ['cue', 'response', 'normed', 'groupsize', 'participants_producing_response', 'forward_strength',
                'cue_set_size', 'cue_frequency', 'cue_concreteness', 'cue_homograph', 'cue_part_of_speech',
                'cue_connectivity', 'cue_resonant_probability', 'cue_resonant_strength', 'cue_use_code',
                'response_set_size', 'response_frequency', 'response_concreteness', 'response_homograph',
                'response_part_of_speech', 'response_connectivity', 'response_resonant_probability',
                'response_resonant_strength', 'response_use_code']

        # fill NaNs with zeroes
        reader.fillna(0)
        # ZUCCHINI, NASTY, YES, 148, 3, .020, 0.000, 0.0000, .0020, 0, 0, 2.0, 0, 13, 0, �, , N, 2.50, .17, .007, 1, 12, 5, 2.63, , AJ, 2.58, .42, .030, 1
        for row in reader.iterrows():
            newrow = []
            newrow.append(str(row[1][reader.columns.get_loc('CUE')]).lower())
            newrow.append(str(row[1][reader.columns.get_loc('TARGET')]).lower())
            newrow.append(True if row[1][reader.columns.get_loc('NORMED?')] == 'YES' else False)
            newrow.append(int(row[1][reader.columns.get_loc('#G')]))
            newrow.append(int(row[1][reader.columns.get_loc('#P')]))
            newrow.append(round(float(row[1][reader.columns.get_loc('FSG')]), 3))

            # cue metrics
            newrow.append(round(float(row[1][reader.columns.get_loc('QSS')]), 3))
            newrow.append(round(float(row[1][reader.columns.get_loc('QFR')]), 3))
            newrow.append(round(float(row[1][reader.columns.get_loc('QCON')]), 3))
            newrow.append(True if row[1][reader.columns.get_loc('QH')] == 'Y' else False)
            newrow.append(True if row[1][reader.columns.get_loc('QPS')] == 'Y' else False)
            newrow.append(round(float(row[1][reader.columns.get_loc('QMC')]), 3))
            newrow.append(round(float(row[1][reader.columns.get_loc('QPR')]), 3))
            newrow.append(round(float(row[1][reader.columns.get_loc('QRSG')]), 3))
            newrow.append(bool(int(row[1][reader.columns.get_loc('QUC')])))

            # response metrics
            newrow.append(round(float(row[1][reader.columns.get_loc('TSS')]), 3))
            newrow.append(round(float(row[1][reader.columns.get_loc('TFR')]), 3))
            newrow.append(round(float(row[1][reader.columns.get_loc('TCON')]), 3))
            newrow.append(True if row[1][reader.columns.get_loc('TH')] == 'Y' else False)
            newrow.append(True if row[1][reader.columns.get_loc('TPS')] == 'Y' else False)
            newrow.append(round(float(row[1][reader.columns.get_loc('TMC')]), 3))
            newrow.append(round(float(row[1][reader.columns.get_loc('TPR')]), 3))
            newrow.append(round(float(row[1][reader.columns.get_loc('TRSG')]), 3))
            newrow.append(bool(int(row[1][reader.columns.get_loc('TUC')])))
            normsvalues.append(newrow)

        db.execute_values_wrap('norms', keys, normsvalues)

    if prevalences:
        prevalences = []
        with open(settings.PREVALENCES_FILE, mode='r', encoding='ISO-8859-1') as prevalences_file:
            # Word,Pknown,Nobs,Prevalence,FreqZipfUS
            reader = csv.DictReader(prevalences_file)
            for row in reader:
                prevalences.append(list(row.values()))
            db.execute_values_wrap('prevalences', ('word', 'pknown', 'nobs', 'prevalence', 'freqzipfus'),
                                   prevalences)

    if concreteness:
        concretenesses = []
        with open(settings.CONCRETENESS_FILE, mode='r', encoding='ISO-8859-1') as concreteness_file:
            # Word	Bigram	Conc.M	Conc.SD	Unknown	Total	Percent_known	SUBTLEX	Dom_Pos
            reader = csv.DictReader(concreteness_file, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                concretenesses.append(list(row.values()))
            db.execute_values_wrap('concreteness', (
                'word', 'bigram', 'concreteness_measure', 'concreteness_standard_dev', 'unknown', 'total',
                'percent_known', 'subtlex', 'dom_pos'), concretenesses)

    if age:
        age_of_acquisitions = []
        # this is to match up with the sense2vec data
        part_of_speech_mapping = {
            'adjective': 'adj',
            'adposition': 'adp',
            'adverb': 'adv',
            'auxiliary': 'aux',
            'conjunction': 'conj',
            'determiner': 'det',
            'interjection': 'intj',
            'numeral': 'num',
            'particle': 'part',
            'pronoun': 'pron',
            'proper noun': 'propn',
            'punctuation': 'punct',
            'subordinating conjunction': 'sconj',
            'symbol': 'sym',
            'to': 'adp',
            'ex': 'adv',
            'article': 'det',
            'not': 'adv'
        }
        with open(settings.AGE_FILE, mode='r', encoding='ISO-8859-1') as age_file:
            # Word	Alternative.spelling	Freq_pm	Dom_PoS_SUBTLEX	Nletters	Nphon	Nsyll	Lemma_highest_PoS	AoA_Kup	Perc_known	AoA_Kup_lem	Perc_known_lem
            reader = csv.DictReader(age_file, delimiter=',')
            for row in reader:
                row['Freq_pm'] = row['Freq_pm'].replace('#N/A', '0')
                row['Dom_PoS_SUBTLEX'] = re.sub(r'(#N\/A)*(\d+\.\d+)*', '', row['Dom_PoS_SUBTLEX']).lower()
                if row['Dom_PoS_SUBTLEX'] in part_of_speech_mapping:
                    row['Dom_PoS_SUBTLEX'] = part_of_speech_mapping[row['Dom_PoS_SUBTLEX']]
                row['AoA_Kup_lem'] = row['AoA_Kup_lem'].replace('NA', '0')
                rowlist = list(row.values())
                edited_row = rowlist[0:8]
                edited_row.extend(rowlist[10:12])
                age_of_acquisitions.append(tuple(edited_row))
            db.execute_values_wrap('acquisition_age', (
                'word', 'alternative_spelling', 'freq_pm', 'dom_pos_subtlex', 'num_letters', 'num_phon',
                'num_syllables', 'lemma_highest_pos', 'aoa_kup_lemma', 'percent_known_lemma',), age_of_acquisitions)

    if emotions:
        emotions = []
        # TODO import segments of emotions like male/female split
        with open(settings.EMOTIONS_FILE, mode='r', encoding='ISO-8859-1') as emotions_file:
            # Word	V.Mean.Sum	V.SD.Sum	V.Rat.Sum	A.Mean.Sum	A.SD.Sum	A.Rat.Sum	D.Mean.Sum
            reader = csv.DictReader(emotions_file, delimiter=',')
            for row in reader:
                final_row = [row['Word'], row['V.Mean.Sum'], row['A.Mean.Sum'], row['D.Mean.Sum']]
                emotions.append(final_row)
            db.execute_values_wrap('emotions', (
                'word', 'valence', 'arousal', 'dominance'), emotions)


def setup():
    parser = argparse.ArgumentParser(description='Import various types of word data')
    parser.add_argument('--wordassoc', action='store_true', help='Small world of words data')
    parser.add_argument('--responses', action='store_true', help='Responses table refresh')
    parser.add_argument('--affix', action='store_true', help='Affixes')
    parser.add_argument('--norms', action='store_true', help='Word association norms')
    parser.add_argument('--prevalences', action='store_true', help='Word prevalences')
    parser.add_argument('--concreteness', action='store_true', help='Word concretenesses')
    parser.add_argument('--age', action='store_true', help='Word age if acquisition')
    parser.add_argument('--emotions', action='store_true', help='Word valence, arousal, and dominance ratings')

    args = parser.parse_args()
    process(args.wordassoc, args.responses, args.affix, args.norms, args.prevalences, args.concreteness, args.age,
            args.emotions)


if __name__ == "__main__":
    setup()
