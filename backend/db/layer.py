import psycopg2 as pg
from backend.settings_private import PG_PASSWORD, PG_USER
from backend.settings import FREQUENCIES_FILE, WORD_FREQUENCIES, SYNONYMS_ANTONYMS_FILE, MERONYMS_FOLDER
from psycopg2.extras import execute_values
import pprint
import sys
import logging
import numpy as np
import pandas as pd
import sqlalchemy
from backend.lib.helpers.df import unpack_lists
import re
from os import listdir
from pathlib import Path

SOURCES = ['humanbraincloud', 'smallworldofwords', 'bot']


def connect():
    conn = pg.connect(host='localhost', user=PG_USER, password=PG_PASSWORD, database="backend")
    return conn


def engine():
    engine = sqlalchemy.create_engine(
        'postgresql://{user}:{password}@localhost:5432/backend'.format(user=PG_USER, password=PG_PASSWORD))
    return engine


def execute_values_wrap(table, fieldnametuple, values, returning=None):
    # TODO make this thread safe
    # TODO make this SQL injection safe
    # user sql.Identifier?
    conn = connect()
    return_list = []
    page_size = 1000
    returning_len = 0
    try:

        if not isinstance(values[0], list) and not isinstance(values[0], tuple):
            raise TypeError(
                'values must be a list of tuples or lists, not {} [{}]'.format(type(values), type(values[0])))
        fieldnames = '(%s)' % ','.join(fieldnametuple).lower()
        query = "INSERT INTO %(table)s %(fieldnames)s VALUES" % {'table': table, 'fieldnames': fieldnames}
        query = ' '.join([query, "%s RETURNING id"])
        if returning:
            returning_len = len(returning)
            for return_var in returning:
                query = ' '.join([query, ", {}".format(return_var)])
        logging.debug(query)

        # execute_values batches the queries into page size page_size so we need to do the same to catch all of the ids
        value_chunks = chunks(values, page_size)
        number_of_variables = 1 + returning_len
        for value_chunk in value_chunks:
            cur = conn.cursor()
            execute_values(cur, query, value_chunk, page_size=page_size)
            return_values = [[row[i] for i in range(0, number_of_variables)] for row in cur.fetchall()]
            return_list.extend(return_values)

        conn.commit()
    except (Exception, pg.DatabaseError) as error:
        err_type, err_obj, traceback = sys.exc_info()
        print(err_type, err_obj, 'line number {}'.format(traceback.tb_lineno))
        line_num = traceback.tb_lineno
        print(error.diag)
        pprint.pprint(dict(zip(fieldnametuple, values[line_num])))
        print(query)


    finally:
        if conn is not None:
            conn.close()
        return return_list


def chunks(lst, n):
    # Yield successive n-sized chunks from list
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


################################################### DB METHODS #########################################################

def get_cues(min_length=None, limit=None):
    conn = connect()
    cur = conn.cursor(cursor_factory=pg.extras.DictCursor)
    query = """ SELECT DISTINCT cue
                    FROM wordassoc"""
    if limit:
        query += """ LIMIT %s""" % str(int(limit))
    cur.execute(query)
    return [x for x in cur]


def get_cue_response_pairs(limit=None, source=None):
    sourcestr = ""
    if source and source in SOURCES:
        sourcestr = """WHERE r.source = '{}'\n""".format(source)

    conn = connect()
    cur = conn.cursor(cursor_factory=pg.extras.DictCursor)

    query = """
        SELECT 
          r.id AS rowid, 
          r.user_id, 
          r.cue AS cue_raw, 
          r.cue_clean, 
          c_corrections.correction AS cue_correction, 
          LOWER(c_sense.sense) as cue_sense, 
          r.response AS response_raw, 
          r.response_clean, 
          r_corrections.correction AS response_correction, 
          LOWER(r_sense.sense) as response_sense, 
          r.source 
        FROM 
          responses r 
          LEFT JOIN s2vsenses AS c_sense ON r.cue_sense_id = c_sense.id 
          LEFT JOIN s2vsenses AS r_sense ON r.response_sense_id = r_sense.id 
          LEFT JOIN corrections AS c_corrections ON r.id = c_corrections.response_id 
          AND c_corrections.type = 'c' 
          LEFT JOIN corrections AS r_corrections ON r.id = r_corrections.response_id 
          AND c_corrections.type = 'r' 
        {sourcestr}
          ORDER BY cue_clean, response_clean ASC
        """.format(sourcestr=sourcestr)

    if limit:
        query += """ LIMIT %s""" % str(int(limit))
    # logging.debug(query)
    cur.execute(query)

    return cur


def get_cue_response_pairs_coded(limit=None, source=None):
    sourcestr = ""
    if source and source in SOURCES:
        sourcestr = """WHERE r.source = '{}'\n""".format(source)

    conn = connect()
    cur = conn.cursor(cursor_factory=pg.extras.DictCursor)
    query = """
        SELECT 
          r.id AS rowid, 
          r.user_id, 
          r.cue AS cue_raw, 
          r.cue_clean, 
          r.response AS response_raw, 
          r.response_clean, 
          r.source, 
          u.age, 
          u.nativelanguage, 
          u.gender, 
          u.education, 
          u.city, 
          u.country, 
          cod.*
        FROM 
          responses r 
          LEFT JOIN users u ON u.id = r.user_id 
          INNER JOIN coding cod ON cod.response = r.response 
          AND cod.cue = r.cue
        {sourcestr}
        ORDER BY 
          cue_clean, 
          response_clean ASC
""".format(sourcestr=sourcestr)

    if limit:
        query += """ LIMIT %s""" % str(int(limit))
    logging.debug(query)
    cur.execute(query)

    return cur


def get_affixes():
    conn = connect()
    cur = conn.cursor(cursor_factory=pg.extras.DictCursor)
    query = """ SELECT affix, type
                FROM affixes"""
    cur.execute(query)
    return [dict(x) for x in cur]


def get_common_norms():
    conn = connect()
    cur = conn.cursor(cursor_factory=pg.extras.DictCursor)
    query = """ SELECT cue, response
                FROM norms
                WHERE forward_strength > 0.4"""
    cur.execute(query)
    cue_pair_dict = {}
    for pair in cur:
        cue_pair_dict.setdefault(pair['cue'], [])
        cue_pair_dict[pair['cue']].append(pair['response'])
    conn.close()
    return cue_pair_dict


def get_prevalences():
    conn = connect()
    cur = conn.cursor(cursor_factory=pg.extras.DictCursor)
    query = """ SELECT word, prevalence
                FROM prevalences
    """
    cur.execute(query)
    return {x['word']: x['prevalence'] for x in cur}


def get_concretenesses():
    conn = connect()
    cur = conn.cursor(cursor_factory=pg.extras.DictCursor)
    query = """ SELECT word, concreteness_measure as concreteness
                FROM concreteness
    """
    cur.execute(query)
    return {x['word']: x['concreteness'] for x in cur}


def get_acquisition_ages():
    conn = connect()
    cur = conn.cursor(cursor_factory=pg.extras.DictCursor)
    query = """ SELECT word,
                        alternative_spelling as word_alternative,
                        lemma_highest_pos as word_lemma,
                        aoa_kup_lemma as age,
                        dom_pos_subtlex as part_of_speech
                FROM acquisition_age
    """
    cur.execute(query)
    age_dict = {}
    for x in cur:
        word_set = {x['word'], x['word_alternative'], x['word_lemma']}
        for w in list(word_set):
            age_dict[w] = x['age']
    conn.close()
    return age_dict


def get_part_of_speech():
    conn = connect()
    cur = conn.cursor(cursor_factory=pg.extras.DictCursor)
    query = """ SELECT word,
                        alternative_spelling as word_alternative,
                        lemma_highest_pos as word_lemma,
                        aoa_kup_lemma as age,
                        dom_pos_subtlex as part_of_speech
                FROM acquisition_age
    """
    cur.execute(query)
    pos_dict = {}
    for x in cur:
        word_set = {x['word'], x['word_alternative'], x['word_lemma']}
        for w in list(word_set):
            pos_dict[w] = x['part_of_speech']
    conn.close()
    return pos_dict


def get_emotions():
    conn = connect()
    cur = conn.cursor(cursor_factory=pg.extras.DictCursor)
    query = """ SELECT word, valence, arousal, dominance
                FROM emotions
    """
    cur.execute(query)
    emotions = {x['word']: {'valence': x['valence'], 'arousal': x['arousal'], 'dominance': x['dominance']} for x in
                cur}
    conn.close()
    return emotions


def get_orthographic():
    # TODO some formatting of orthographic names before returning>
    conn = connect()
    cur = conn.cursor(cursor_factory=pg.extras.DictCursor)
    query = """ SELECT *
                FROM orthographic
    """
    cur.execute(query)
    orthographic = {x['word']: x for x in cur}
    conn.close()
    return orthographic


def get_phonographic():
    # TODO some formatting of phonographic names before returning>
    conn = connect()
    cur = conn.cursor(cursor_factory=pg.extras.DictCursor)
    query = """ SELECT word, nphon, unsdens, strdens, unsbpav, strbpav, unstpav, strtpav, unspospav, strpospav, unslcpospav, strlcpospav, sfreq, scdcnt
                FROM phonographic
    """
    cur.execute(query)
    phonographic = {x['word']: {k: v for k, v in x.items() if k not in ['word']} for x in cur}
    conn.close()
    return phonographic


def get_frequencies():
    df = pd.read_csv(FREQUENCIES_FILE.absolute(), delimiter="\t", encoding='ISO-8859-1')
    df = df.rename(columns={'Word': 'word', 'Lg10WF': 'lg10wf'})
    df = df[['word', 'lg10wf']]
    df['word'] = df['word'].str.encode('utf8')
    df.set_index('word', inplace=True)
    return df


def get_coded_pairs():
    conn = connect()
    cur = conn.cursor(cursor_factory=pg.extras.DictCursor)
    query = """SELECT * FROM coding"""
    cur.execute(query)
    return {(x.get('cue'), x.get('response')): x for x in cur}


def get_all_words():
    # TODO should I be returning cursors and letting the script evaluate them when it needs them?
    conn = connect()
    cur = conn.cursor()
    query = """SELECT word FROM all_words ORDER BY word ASC"""
    cur.execute(query)
    result = set([x[0] for x in cur])
    conn.close()
    return result


def get_word_filter(bands=None, pos=None):
    # Take only words in the 2k to 3k band from the British National Corpus as per Fitzgerald et. al
    df = pd.read_csv(WORD_FREQUENCIES, sep='\t')
    if bands == 'all':
        word_filter = df
    else:
        word_filter = df[2000:3999].copy()
    if pos:
        if isinstance(pos, str):
            pos = [pos]
        word_filter = word_filter[word_filter['PoS'].isin(pos)].copy()
    word_filter['word'] = word_filter['Word'].str.lower().str.encode('utf8')
    return word_filter['word'].values


def get_all_measures_raw():
    conn = connect()
    cur = conn.cursor(cursor_factory=pg.extras.DictCursor)
    query = """SELECT
                    
                    LOWER(a.lemma_highest_pos) as word_lemma,
                    a.dom_pos_subtlex as part_of_speech,
                    LOWER(c.word) AS c_word,
                    LOWER(a.word) AS a_word,
                    LOWER(o.word) AS o_word,
                    LOWER(e.word) AS e_word,
                    LOWER(p.word) AS p_word,
                    LOWER(phon.word) as phon_word,
                    a.aoa_kup_lemma as age,
                    e.arousal AS arousal,
                    e.dominance AS dominance,
                    e.valence AS valence,
                    o.bigram_count AS bigram_count,
                    o.bigram_freq AS bigram_freq,
                    o.ortho_frequency AS ortho_frequency,
                    o.ortho_neighbours AS ortho_neighbours,
                    o.trigram_count AS trigram_count,
                    o.trigram_freq AS trigram_freq,
                    o.unconstrained_bigram_count AS unconstrained_bigram_count,
                    o.unconstrained_bigram_freq AS unconstrained_bigram_freq,
                    o.unconstrained_trigram_count AS unconstrained_trigram_count,
                    o.unconstrained_trigram_freq AS unconstrained_trigram_freq,
                    o.unconstrained_unigram_count AS unconstrained_unigram_count,
                    o.unconstrained_unigram_freq AS unconstrained_unigram_freq,
                    o.unigram_count AS unigram_count,
                    o.unigram_freq AS unigram_freq,
                    p.pknown AS pknown,
                    p.prevalence AS prevalence,
                    phon.nphon AS nphon,
                    phon.scdcnt AS scdcnt,
                    phon.sfreq AS sfreq,
                    phon.strbpav AS strbpav,
                    phon.strdens AS strdens,
                    phon.strlcpospav AS strlcpospav,
                    phon.strpospav AS strpospav,
                    phon.strtpav AS strtpav,
                    phon.unsbpav AS unsbpav,
                    phon.unsdens AS unsdens,
                    phon.unslcpospav AS unslcpospav,
                    phon.unspospav AS unspospav,
                    phon.unstpav AS unstpav,
                    c.concreteness_measure AS concreteness_measure
                FROM orthographic o
                FULL OUTER JOIN concreteness c ON o.word=c.word
                FULL OUTER JOIN prevalences p ON o.word=p.word
                FULL OUTER JOIN emotions e ON o.word=e.word
                FULL OUTER JOIN acquisition_age a ON o.word=a.word
                FULL OUTER JOIN phonographic phon ON o.word=phon.word
                ORDER BY c_word, a_word, o_word, e_word, p_word, phon_word  ASC;"""

    cur.execute(query)
    return cur


def get_all_measures_processed():
    conn = connect()
    cur = conn.cursor(cursor_factory=pg.extras.DictCursor)
    query = """SELECT * FROM measures ORDER BY word ASC"""
    cur.execute(query)
    return cur


def get_senses():
    conn = connect()
    cur = conn.cursor()
    query = """SELECT id as senseid, sense FROM s2vsenses ORDER BY sense ASC"""
    cur.execute(query)
    result = {x[1]: int(x[0]) for x in cur}
    conn.close()
    return result


def get_synonyms_antonyms():
    # TODO move these utility functions to somewhere more general
    df = pd.read_csv(SYNONYMS_ANTONYMS_FILE)

    def listify(wordstr):
        flat = wordstr
        if isinstance(wordstr, str):
            x = wordstr
            nested = [re.sub('\s\((or|also\s(\w+))\)', ',\\2', y).split(',') for y in x.split(',')]
            flat = [re.sub('\s+\[\w+\]', '', item) for sublist in nested for item in sublist]
        return flat

    def get_related_list(df, relation):
        unpacked = pd.DataFrame(unpack_lists(df['{}_list'.format(relation)]), columns=[relation, 'index'], dtype=str)
        unpacked['index'] = unpacked['index'].astype(int)
        unpacked[relation] = unpacked[relation].astype(str)
        unpacked.set_index('index', inplace=True)
        df.index.name = 'index'
        relations = pd.merge(df[['word']].astype(str), unpacked, how='right', left_index=True, right_index=True)
        return relations

    df['synonym_list'] = df['synonyms'].apply(listify)
    df['antonym_list'] = df['antonyms'].apply(listify)
    synonyms = get_related_list(df, 'synonym')
    antonyms = get_related_list(df, 'antonym')
    return synonyms, antonyms


def get_meronyms():
    df = pd.DataFrame()
    for file in listdir(MERONYMS_FOLDER):
        if 'meronyms' in file:
            meronym_type = str(file).split('-')[1].split(']')[0].strip()
            mdf = pd.read_csv(Path(MERONYMS_FOLDER).joinpath(file), delimiter='\t', names=['word', 'relations'])
            mdf['type'] = meronym_type
            df = pd.concat([df, mdf])
    df['relations'] = df['relations'].str.split('/')
    df.reset_index(inplace=True, drop=True)
    unpacked = pd.DataFrame(unpack_lists(df['relations'], length=1), columns=['relation', 'index']).set_index('index')
    unpacked.index = unpacked.index.astype(int)
    meronyms = df[['word', 'type']].merge(unpacked, how='inner', left_index=True, right_index=True)
    return meronyms


def save_all_measures(df):
    logging.debug('Saving measures to db')
    conn = engine()
    df.to_sql('measures', con=conn, if_exists='replace', index=False)




def save_cue_response_pair(cue_response_pair):
    cue = cue_response_pair['cue']
    response = cue_response_pair['response']
    user_id = cue_response_pair['user_id']
    time_duration = cue_response_pair['time_duration']
    corrections = cue_response_pair['corrections']
    created = cue_response_pair['created']
    source = cue_response_pair['source']
    conn = connect()
    cur = conn.cursor()
    query = """INSERT INTO responses (user_id, cue, response, time_duration, created, source)
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING id"""

    cur.execute(query, (user_id, cue, response, time_duration, created, source))
    id_of_new_row = cur.fetchone()[0]
    correction_ids = []
    for original_word, corrections in corrections.items():
        correction_ids.append(save_corrections([original_word, corrections, None, id_of_new_row]))
    # # put the newly created correction ids in the responses table, for good measure
    # query = """UPDATE responses SET correction_ids = %s WHERE id = %s"""
    # cur.execute(query, (correction_ids, id_of_new_row))
    conn.commit()

    if 'correction' in cue_response_pair:
        update_correction(response_id=id_of_new_row, original_word=response, correction=cue_response_pair['correction'])

    return id_of_new_row


def save_cue_relations(fingerprint, cue, relation_type, relations, time_duration, created):
    conn = connect()
    cur = conn.cursor()
    for relation, score in relations.items():
        query = """INSERT INTO relations (fingerprint, cue, relation_type, relation, score, time_duration, created)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)"""
        cur.execute(query, (fingerprint, cue, relation_type, relation, score, time_duration, created))
    conn.commit()

    return


def save_cue_response_pairs(cue_response_pairs):
    response_values = []
    for cue_response_pair in cue_response_pairs:
        response_values.append(tuple(cue_response_pair.values()))
    fieldnamelist = list(cue_response_pairs[0].keys())
    fieldnametuple = tuple(fieldnamelist)
    response_ids = execute_values_wrap('responses', fieldnametuple, response_values)
    rid_len = len(response_ids)
    rval_len = len(response_values)
    if rid_len != rval_len:
        raise ValueError(
            'Number of database rows inserted does not equal original number of records {} != {}'.format(rid_len,
                                                                                                         rval_len))

    # cleanup old corrections
    delete_orphaned_corrections()
    return response_ids


def save_corrections(corrections_values):
    # TODO better checks.
    # TODO maybe pass in dict instead
    delete_orphaned_corrections()
    try:
        # original word
        assert (type(corrections_values[0][0]) is str)
        # corrections
        # assert (type(corrections_values[0][1]) is list)
        # # correction
        assert (type(corrections_values[0][1]) is str or corrections_values[0][2] is None)
        # response id
        assert (type(corrections_values[0][2]) is int)
        # type
        assert (corrections_values[0][3] in ('c', 'r'))
    except IndexError as e:
        print(e, corrections_values)
        exit(0)
    fieldnametuple = ('original_word', 'correction', 'response_id', 'type')
    correction_ids = execute_values_wrap('corrections', fieldnametuple, corrections_values)
    if len(corrections_values) != len(correction_ids):
        raise ValueError('Number of database rows inserted does not equal original number of records')
    return correction_ids


def update_senses(df):
    # later, if generalising use sql.Identifier(c) for passing in columns
    conn = connect()
    cur = conn.cursor()
    senses = get_senses()
    df['cue_sense_id'] = df['cue_sense'].map(senses)  # replace({np.nan: -1}).astype(int)
    df['response_sense_id'] = df['response_sense'].map(senses)  # .replace({np.nan: -1}).astype(int)
    values = df[['cue_sense_id', 'response_sense_id', 'rowid']].replace({np.nan: None}).values.tolist()
    query = """UPDATE responses AS r 
                  SET cue_sense_id = CAST(e.cue_sense_id AS int),
                  response_sense_id = CAST(e.response_sense_id AS int)
                  FROM (VALUES %s) AS e(cue_sense_id,response_sense_id, id) 
                  WHERE e.id = r.id;"""
    pg.extras.execute_values(cur, query, values, template=None, page_size=100)
    conn.commit()


def update_correction(response_id, original_word, correction):
    conn = connect()
    cur = conn.cursor()
    query = """UPDATE corrections SET correction = %s WHERE original_word = %s AND response_id = %s"""
    cur.execute(query, (correction, original_word, response_id))
    conn.commit()


def save_coded_pairs(coded_pairs):
    logging.debug('Saving coded pairs')
    conn = engine()
    # TODO manage creation or appending
    # columns = set(conn.execute('SELECT * FROM coding').keys())
    # new_columns = set(coded_pairs.columns.values.tolist())
    # print(columns, new_columns)
    coded_pairs.to_sql('coding', con=conn, if_exists='append')


def delete_coded_pairs():
    conn = connect()
    cur = conn.cursor()
    query = """DROP TABLE  IF EXISTS coding"""
    cur.execute(query)
    conn.commit()


def delete_responses(source):
    conn = connect()
    cur = conn.cursor()
    query = """DELETE FROM responses WHERE source = %(source)s RETURNING id"""
    cur.execute(query, {'source': source})
    # nasty
    ids = [x[0] for x in cur.fetchall()]
    if ids:
        query = """DELETE FROM corrections WHERE response_id IN (SELECT * FROM UNNEST(%s))"""
        cur.execute(query, (ids,))
    conn.commit()


def delete_users(source):
    conn = connect()
    cur = conn.cursor()
    query = """DELETE FROM users WHERE source = %(source)s"""
    cur.execute(query, {'source': source})
    conn.commit()


def delete_wordassoc():
    conn = connect()
    cur = conn.cursor()
    query = """DELETE FROM wordassoc"""
    cur.execute(query)
    conn.commit()


def delete_corrections(ids):
    conn = connect()
    cur = conn.cursor()
    query = """DELETE
                FROM  corrections c 
                WHERE response_id IN (SELECT * FROM UNNEST(%s))"""
    cur.execute(query, (ids,))
    conn.commit()


def delete_orphaned_corrections():
    conn = connect()
    cur = conn.cursor()
    query = """DELETE
                FROM   corrections c 
                WHERE  NOT EXISTS (
                    SELECT  -- SELECT list mostly irrelevant; can just be empty in Postgres
                    FROM   responses
                    WHERE  id = c.response_id
                );"""
    cur.execute(query)
    conn.commit()


def delete_measures():
    conn = connect()
    cur = conn.cursor()
    query = """DELETE FROM  measures"""
    cur.execute(query)
    conn.commit()
