import psycopg2 as pg
from backend.settings import PG_PASSWORD, PG_USER
from psycopg2.extras import execute_values


def connect():
    conn = pg.connect(host='localhost', user=PG_USER, password=PG_PASSWORD, database="backend")
    return conn


def execute_values_wrap(cur, table, fieldnametuple, values):
    fieldnames = '(%s)' % ','.join(fieldnametuple).lower()
    query = "INSERT INTO %(table)s %(fieldnames)s VALUES" % {'table': table, 'fieldnames': fieldnames}
    query = ' '.join([query, "%s"])
    result = execute_values(cur, query, values)
    return result


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


def get_cue_response_pairs(min_length=None, limit=None, responses=None):
    conn = connect()
    cur = conn.cursor(cursor_factory=pg.extras.DictCursor)
    if responses:
        query = """ SELECT user_id as participantID, cue, response AS response_raw, response AS response, count(response) AS count
                        FROM responses
                        WHERE LENGTH(response) > 3
                        AND LENGTH(cue) > 3
                        AND user_id IN ('325696','325422','325809','325631' ,'325599','325388','325869','325795','325359','325449')
                        GROUP BY user_id, cue, response
                        ORDER BY cue, response ASC"""
    else:
        query = """ SELECT participantID, age, education,  cue, r1raw AS response_raw, r1 AS response, count(r1) AS count
                        FROM wordassoc
                        WHERE r1 != 'Unknown word'
                        AND LENGTH(r1) > 3
                        AND LENGTH(cue) > 3
                        GROUP BY participantID, age, education, cue, r1, r1raw
                        ORDER BY cue, response ASC"""

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
    query = """ SELECT word, alternative_spelling as word_alternative, lemma_highest_pos as word_lemma, aoa_kup_lemma as age
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
    query = """ SELECT *
                FROM phonographic
    """
    cur.execute(query)
    phonographic = {x['word']: x for x in cur}
    conn.close()
    return phonographic


def get_coded_pairs():
    conn = connect()
    cur = conn.cursor(cursor_factory=pg.extras.DictCursor)
    query = """SELECT * FROM coding"""
    cur.execute(query)
    return {(x.get('cue'), x.get('response')): x for x in cur}


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
        correction_ids.append(save_corrections(conn, id_of_new_row, original_word, corrections))
    # put the newly created correction ids in the responses table, for good measure
    query = """UPDATE responses SET correction_ids = %s WHERE id = %s"""
    cur.execute(query, (correction_ids, id_of_new_row))
    conn.commit()
    return id_of_new_row


def save_cue_response_pairs(cue_response_pairs):
    conn = connect()
    cur = conn.cursor()
    values = [list(x.values()) for x in cue_response_pairs]
    fieldnametuple = tuple(cue_response_pairs[0].keys())
    execute_values_wrap(cur, 'responses', fieldnametuple, values)
    conn.commit()


def save_corrections(conn, response_id, original_word, corrections):
    cur = conn.cursor()
    query = """INSERT INTO corrections (response_id, original_word, corrections)
                VALUES (%s, %s, %s) RETURNING id"""
    cur.execute(query, (response_id, original_word, corrections))
    id_of_new_row = cur.fetchone()[0]
    return id_of_new_row


def update_correction(response_id, original_word, correction):
    conn = connect()
    cur = conn.cursor()
    query = """UPDATE corrections SET correction = %s WHERE original_word = %s AND response_id = %s"""
    cur.execute(query, (correction, original_word, response_id))
    conn.commit()


def save_coded_pairs(coded_pairs, fieldnames):
    conn = connect()
    cur = conn.cursor()
    print(fieldnames)
    execute_values_wrap(cur, 'coding', fieldnames,
                        [[key[0], key[1]] + list(x.values()) for key, x in coded_pairs.items()])
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
