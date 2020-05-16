import psycopg2 as pg
from backend.settings import PG_PASSWORD, PG_USER
from psycopg2.extras import execute_values
from nltk import PorterStemmer
from backend.helpers.dataframe import create_dataframe_in_parallel

ps = PorterStemmer()


def connect():
    conn = pg.connect(host='localhost', user=PG_USER, password=PG_PASSWORD, database="backend")
    return conn


def execute_values_wrap(table, fieldnametuple, values):
    conn = connect()
    ids = []
    page_size = 1000
    try:

        cur = conn.cursor()
        if not isinstance(values[0], tuple):
            raise TypeError('values must be a list of tuples, not {} [{}]'.format(type(values), type(values[0])))
        fieldnames = '(%s)' % ','.join(fieldnametuple).lower()
        query = "INSERT INTO %(table)s %(fieldnames)s VALUES" % {'table': table, 'fieldnames': fieldnames}
        query = ' '.join([query, "%s RETURNING id"])

        # execute_values batches the queries into page size page_size so we need to do the same to catch all of the ids
        value_chunks = chunks(values, page_size)

        for value_chunk in value_chunks:
            result = execute_values(cur, query, value_chunk, page_size=page_size)
            ids.extend([row[0] for row in cur.fetchall()])
        conn.commit()
    except (Exception, pg.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
        return ids


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


def get_cue_response_pairs(limit=None, coded=None):
    conn = connect()
    cur = conn.cursor(cursor_factory=pg.extras.DictCursor)
    # TODO make this a RIGHT JOIN instead of SELECT IN ?
    query = """ SELECT * FROM (
                    SELECT  r.user_id AS user_id, 
                            r.cue AS cue,
                            r.response AS response_raw,
                            r.cue_clean,
                            r.response_clean,
                            CASE WHEN c.correction IS NOT NULL THEN c.correction ELSE r.response END AS response,
                            CASE WHEN c.correction IS NOT NULL THEN TRUE ELSE FALSE END AS incorrect_spelling,
                            COUNT(r.response) AS count,
                            source AS source
                        FROM responses r
                        LEFT JOIN corrections c ON r.id = c.response_id
                """
    if coded:
        query += """RIGHT JOIN coding cod ON cod.response=r.response_clean AND cod.cue=r.cue_clean\n"""

    query += """ WHERE LENGTH(r.response) > 3
                AND LENGTH(r.cue) > 3
                AND r.response != 'Unknown word'
                GROUP BY r.user_id, r.cue, r.cue_clean, r.response_clean, r.response, c.correction, r.source
                ORDER BY r.cue, r.response ASC
                ) responses_group"""

    if limit:
        query += """ LIMIT %s""" % str(int(limit))
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


def get_all_measures():
    conn = connect()
    cur = conn.cursor(cursor_factory=pg.extras.DictCursor)
    # TODO words with multiple parts are smushed together. perhaps these should be stored differently on import
    query = """SELECT
                    
                    a.lemma_highest_pos as word_lemma,
                    AVG(a.aoa_kup_lemma) as age,
                    string_agg(a.dom_pos_subtlex, ',') as part_of_speech,
                    string_agg(o.word, ',') as word,
                    AVG(COALESCE(e.arousal, 0)) AS arousal,
                    AVG(COALESCE(e.dominance, 0)) AS dominance,
                    AVG(COALESCE(e.valence, 0)) AS valence,
                    AVG(COALESCE(o.bigram_count, 0)) AS bigram_count,
                    AVG(COALESCE(o.bigram_freq, 0)) AS bigram_freq,
                    AVG(COALESCE(o.frequency, 0)) AS frequency,
                    AVG(COALESCE(o.ortho_frequency, 0)) AS ortho_frequency,
                    AVG(COALESCE(o.ortho_neighbours, 0)) AS ortho_neighbours,
                    AVG(COALESCE(o.trigram_count, 0)) AS trigram_count,
                    AVG(COALESCE(o.trigram_freq, 0)) AS trigram_freq,
                    AVG(COALESCE(o.unconstrained_bigram_count, 0)) AS unconstrained_bigram_count,
                    AVG(COALESCE(o.unconstrained_bigram_freq, 0)) AS unconstrained_bigram_freq,
                    AVG(COALESCE(o.unconstrained_trigram_count, 0)) AS unconstrained_trigram_count,
                    AVG(COALESCE(o.unconstrained_trigram_freq, 0)) AS unconstrained_trigram_freq,
                    AVG(COALESCE(o.unconstrained_unigram_count, 0)) AS unconstrained_unigram_count,
                    AVG(COALESCE(o.unconstrained_unigram_freq, 0)) AS unconstrained_unigram_freq,
                    AVG(COALESCE(o.unigram_count, 0)) AS unigram_count,
                    AVG(COALESCE(o.unigram_freq, 0)) AS unigram_freq,
                    AVG(COALESCE(p.pknown, 0)) AS pknown,
                    AVG(COALESCE(p.prevalence, 0)) AS prevalence,
                    AVG(COALESCE(phon.nphon, 0)) AS nphon,
                    AVG(COALESCE(phon.scdcnt, 0)) AS scdcnt,
                    AVG(COALESCE(phon.sfreq, 0)) AS sfreq,
                    AVG(COALESCE(phon.strbpav, 0)) AS strbpav,
                    AVG(COALESCE(phon.strdens, 0)) AS strdens,
                    AVG(COALESCE(phon.strlcpospav, 0)) AS strlcpospav,
                    AVG(COALESCE(phon.strpospav, 0)) AS strpospav,
                    AVG(COALESCE(phon.strtpav, 0)) AS strtpav,
                    AVG(COALESCE(phon.unsbpav, 0)) AS unsbpav,
                    AVG(COALESCE(phon.unsdens, 0)) AS unsdens,
                    AVG(COALESCE(phon.unslcpospav, 0)) AS unslcpospav,
                    AVG(COALESCE(phon.unspospav, 0)) AS unspospav,
                    AVG(COALESCE(phon.unstpav, 0)) AS unstpav,
                    AVG(COALESCE(c.concreteness_measure, 0)) AS concreteness_measure
                FROM orthographic o
                LEFT JOIN concreteness c ON o.word=c.word
                LEFT JOIN prevalences p ON o.word=p.word
                LEFT JOIN emotions e ON o.word=e.word
                LEFT JOIN acquisition_age a ON o.word=a.word
                LEFT JOIN phonographic phon ON o.word=phon.word
                GROUP BY a.lemma_highest_pos
                ORDER BY a.lemma_highest_pos ASC;"""
    cur.execute(query)

    item_dtype = [
        ('word', '|S'),
        ('age', 'f'),
        ('part_of_speech', 'category'),
        ('word_lemma', '|S'),
        ('arousal', 'f'),
        ('dominance', 'f'),
        ('valence', 'f'),
        ('bigram_count', 'f'),
        ('bigram_freq', 'f'),
        ('frequency', 'f'),
        ('ortho_frequency', 'f'),
        ('ortho_neighbours', 'f'),
        ('trigram_count', 'f'),
        ('trigram_freq', 'f'),
        ('unconstrained_bigram_count', 'f'),
        ('unconstrained_bigram_freq', 'f'),
        ('unconstrained_trigram_count', 'f'),
        ('unconstrained_trigram_freq', 'f'),
        ('unconstrained_unigram_count', 'f'),
        ('unconstrained_unigram_freq', 'f'),
        ('unigram_count', 'f'),
        ('unigram_freq', 'f'),
        ('pknown', 'f'),
        ('prevalence', 'f'),
        ('nphon', 'f'),
        ('scdcnt', 'f'),
        ('sfreq', 'f'),
        ('strbpav', 'f'),
        ('strdens', 'f'),
        ('strlcpospav', 'f'),
        ('strpospav', 'f'),
        ('strtpav', 'f'),
        ('unsbpav', 'f'),
        ('unsdens', 'f'),
        ('unslcpospav', 'f'),
        ('unspospav', 'f'),
        ('unstpav', 'f'),
        ('concreteness_measure', 'f')
    ]
    item_list = explode_word_stems(cur)
    df = create_dataframe_in_parallel(item_list, index='word', columns_types=item_dtype)

    return df


def explode_word_stems(cur):
    item_list = []
    for item in cur:
        row_dict = dict(item)
        # row_dict.pop('word_alternative',None)
        row_dict.pop('word_lemma')
        word_set = set(item['word'].split(','))
        word_set.add(item['word_lemma'])
        word_set.discard(None)
        pos_set = None
        try:
            pos_set = set(item['part_of_speech'].split(','))
            pos_set.discard(None)
            pos_set = ','.join(sorted(list(pos_set)))
        except AttributeError:
            pass
        for w in list(word_set):
            row = row_dict.copy()
            row['word'] = w
            row['part_of_speech'] = pos_set
            item_list.append(row)
    return item_list


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


def save_cue_response_pairs_and_corrections(cue_response_pairs):
    response_values = []
    corrections_values = []
    for cue_response_pair in cue_response_pairs:
        corrections_values.append([cue_response_pair.get('response'),
                                   cue_response_pair.get('corrections'),
                                   cue_response_pair.get('correction')])
        cue_response_pair.pop('corrections', [])
        cue_response_pair.pop('correction', None)
        response_values.append(tuple(cue_response_pair.values()))
    fieldnamelist = list(cue_response_pairs[0].keys())
    fieldnametuple = tuple(fieldnamelist)
    response_ids = execute_values_wrap('responses', fieldnametuple, response_values)
    if len(response_ids) != len(response_values):
        raise ValueError('Number of database rows inserted does not equal original number of records')
    for idx, response_id in enumerate(response_ids):
        corrections_values[idx].append(response_id)
        if len(corrections_values[idx]) != 4:
            raise ValueError('Field(s) missing from corrections record')
        # cast to tuple to avoid this https://github.com/psycopg/psycopg2/issues/537
        corrections_values[idx] = list(corrections_values[idx])
    # get rid of empty list corrections
    corrections_values = [tuple(c) for c in corrections_values if len(c[1])]
    correction_ids = []
    if corrections_values:
        correction_ids = save_corrections(corrections_values)
    # cleanup old corrections
    delete_corrections()
    return response_ids, correction_ids


def save_corrections(corrections_values):
    # todo better checks or maybe pass in dict instead, check all values...
    # original word
    try:
        assert (type(corrections_values[0][0]) is str)
        # corrections
        assert (type(corrections_values[0][1]) is list)
        # correction
        assert (type(corrections_values[0][2]) is str or corrections_values[0][2] is None)
        # response id
        assert (type(corrections_values[0][3]) is int)
    except IndexError:
        print(corrections_values)
        exit(0)
    fieldnametuple = ('original_word', 'corrections', 'correction', 'response_id')
    correction_ids = execute_values_wrap('corrections', fieldnametuple, corrections_values)
    if len(corrections_values) != len(correction_ids):
        raise ValueError('Number of database rows inserted does not equal original number of records')
    return correction_ids


def update_correction(response_id, original_word, correction):
    conn = connect()
    cur = conn.cursor()
    query = """UPDATE corrections SET correction = %s WHERE original_word = %s AND response_id = %s"""
    cur.execute(query, (correction, original_word, response_id))
    conn.commit()


def save_coded_pairs(coded_pairs, fieldnames):
    conn = connect()
    cur = conn.cursor()
    values = []
    for key, x in coded_pairs.items():
        value = tuple([key[0], key[1]] + list(x.values()))
        values.append(value)
        if len(value) != len(fieldnames):
            import pprint
            pprint.pprint(value)
            print(len(value))
            pprint.pprint(fieldnames)
            print(len(fieldnames))
            print('error')
            exit(0)
    execute_values_wrap('coding', fieldnames, values)
    # execute_values_wrap(cur, 'coding', fieldnames,
    #                b      [[key[0], key[1]] + list(x.values()) for key, x in coded_pairs.items()])
    conn.commit()


def delete_coded_pairs():
    conn = connect()
    cur = conn.cursor()
    query = """DELETE FROM coding"""
    cur.execute(query)
    conn.commit()


def delete_responses(source):
    conn = connect()
    cur = conn.cursor()
    query = """DELETE FROM responses WHERE source = %s"""
    cur.execute(query, (source,))
    conn.commit()


def delete_corrections():
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
