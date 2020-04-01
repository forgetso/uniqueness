from zipfile import ZipFile
import backend.settings as settings
import backend.lib.db as db
import io
import csv
import datetime
import argparse
import os
import pandas


def process(wordassoc=None, responses=None, affix=None, norms=None, prevalences=None, concreteness=None):
    conn = db.connect()
    db.setup(wordassoc=wordassoc, responses=responses, affixes=affix, norms=norms, prevalences=prevalences,
             concreteness=concreteness)
    cur = conn.cursor()
    if wordassoc:
        with ZipFile(settings.ZIP_FILE) as zf:
            infile = zf.open(settings.DATA_FILE, mode='r')
            infile = io.TextIOWrapper(infile, encoding='utf8', newline='')

            reader = csv.DictReader(infile)
            formatted = []
            for row in reader:
                row.pop('')
                row['id'] = int(row['id'])
                row['participantID'] = int(row['participantID'])
                row['age'] = int(row['age'])
                row['created_at'] = datetime.datetime.strptime(row['created_at'], '%Y-%m-%d %H:%M:%S')
                formatted.append(list(row.values()))

        db.execute_values_wrap(cur, 'wordassoc', reader.fieldnames[1:], formatted)
        conn.commit()

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

        db.execute_values_wrap(cur, 'affixes', ('affix', 'type'), affixes)
        conn.commit()

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

        db.execute_values_wrap(cur, 'norms', keys, normsvalues)
        conn.commit()

    if prevalences:
        prevalences = []
        with open(settings.PREVALENCES_FOLDER, mode='r', encoding='ISO-8859-1') as prevalences_file:
            # Word,Pknown,Nobs,Prevalence,FreqZipfUS
            reader = csv.DictReader(prevalences_file)
            for row in reader:
                prevalences.append(list(row.values()))
            db.execute_values_wrap(cur, 'prevalences', ('word', 'pknown', 'nobs', 'prevalence', 'freqzipfus'),
                                   prevalences)
            conn.commit()

    if concreteness:
        concreteness = []
        with open(settings.PREVALENCES_FOLDER, mode='r', encoding='ISO-8859-1') as concreteness_file:
            # Word,Pknown,Nobs,Prevalence,FreqZipfUS
            reader = csv.DictReader(concreteness_file)
            for row in reader:
                prevalences.append(list(row.values()))
            db.execute_values_wrap(cur, 'prevalences', ('word', 'pknown', 'nobs', 'prevalence', 'freqzipfus'),
                                   prevalences)
            conn.commit()


def setup():
    parser = argparse.ArgumentParser(description='Import various types of word data')
    parser.add_argument('--wordassoc', action='store_true', help='Small world of words data')
    parser.add_argument('--responses', action='store_true', help='Responses table refresh')
    parser.add_argument('--affix', action='store_true', help='Affixes')
    parser.add_argument('--norms', action='store_true', help='Word association norms')
    parser.add_argument('--prevalences', action='store_true', help='Word prevalences')
    parser.add_argument('--concreteness', action='store_true', help='Word concreteness')
    args = parser.parse_args()
    process(args.wordassoc, args.responses, args.affix, args.norms, args.prevalences)


if __name__ == "__main__":
    setup()
