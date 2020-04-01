from backend.lib.db import get_cue_response_pairs, get_affixes, get_common_norms
from nltk.corpus import wordnet as wn
import nltk
import jellyfish
import unidecode
from sense2vec import Sense2Vec
from backend.settings import SENSE2VEC_FOLDER
import pprint
import re
from backend.lib.wordnet_relations import WordnetRelations

AFFIXES = [x.get('affix') for x in get_affixes()]

# The following categories could be used to describe a cue response pair
CUE_RESPONSE_VECTOR = {
    # these are the ways in which the response could be related to the cue
    'relations': {
        'exact': False,
        'antonym': False,
        'associated': False,
        'co-hyponym': False,
        'holonym': False,
        'hypernym': False,
        'hyponym': False,
        'instance': False,
        'meronym': False,
        'phonology': False,
        'synonym': False,
        'syntagmatic': False,
        'positional': False,
        'affix': False,
    },
    # these are basic attributes of the response
    'basic': {
        'length': None,
        'response_time': None,
        'syllables': None,
        'capitalised': False,
        'punctuation': False,
        'number_of_words': None,
        'incorrect_spelling': False,
    },
    # these are other more complicated, derived metrics
    'other': {
        'prevalence': None,
        'corpus_similarity': None,
        'dictionary_similarity': None,
        'concreteness': None,
        'age_of_acquisition': None,
        'emotion': None,
        'subtitle_frequency': None,
        'orthographic_neighbourhood': None,
        'phonograpic_neighbourhood': None,
        'part_of_speech': None,
        'morpheme': False,
        'cultural_reference': None,
        'active_brain_region': None,
    }
}


def main():
    # Returns the category (1,2,3,4) of a cue response pair, based on the following framework
    #
    # 1 Meaning-based association
    # 'Defining Synonym                x means the same as y
    # 'Specific Synonym                x can mean y in some specific contexts
    # 'Lexical set/context related     x and y same lexicalset/coordinates/meronyms/superordinates/provide context
    # 'Conceptual association          x and y have some other conceptual link
    #
    # 2 Position-based association
    # 'Consecutive xy collocation      y follows x directly (includes compounds)
    # 'Consecutive yx collocation      y precedes x directly (includes compounds)
    # 'Alphabet collocation            y follows x, literally TODO does this depend too much on the cue?
    # 'Other collocational             y follows/precedes x in phrase with words between them
    #
    # 3 Form-based associations
    # 'Change of affix                 y is x plus or minus affix
    #                                   of a word with a similar form to x
    # 'Similar form not meaning        y looks or sounds similar to x but has no clear meaning link or is an associate
    #
    # 4 Erratic association
    # 'No link/blank                   y has no decipherable link to x or no response given

    # TODO consider lists of films, books and music
    # TODO consider the categories found in the Gallant Research:
    # TODO (visual, tactile, outdoor, bodypart, number, place, violence, person, social, mental, time)

    common_norms = get_common_norms()
    s2v = Sense2Vec().from_disk(SENSE2VEC_FOLDER)

    cue_response_pairs = []
    distinct_cue_response_pairs = set()
    for pair in get_cue_response_pairs(limit=100):
        pair = dict(pair)
        cue = pair['cue'].lower()
        response = pair['response'].replace(' ', '_').replace('.', '').lower()
        pair['cue'] = cue
        pair['response'] = response
        cue_response_pairs.append(pair)
        distinct_cue_response_pairs.add((pair['cue'], pair['response']))

    participants = {}
    totals = dict()

    erratic = set()
    # code distinct list of cue response pairs
    coded_pairs = {}
    for pair in distinct_cue_response_pairs:
        cue = pair[0]
        response = pair[1]
        print(cue, response)
        category = 0
        match_type = None
        depth = 0

        if len(cue) > 3:
            relation_obj = WordnetRelations(cue, response)
            coded_pairs[(cue, response)] = relation_obj.relations
            for type in relation_obj.relation_types:
                if relation_obj.relations[type]:
                    totals.setdefault(type, 0)
                    totals[type] += 1
    print(totals)
    print(sum(totals.values()))


def create_s2v_keys(word):
    keys = []
    word_types = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN',
                  'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
    for word_type in word_types:
        keys.append(''.join([word, '|', word_type]))
    return keys


def check_similarity(w1, w2, depth=None):
    if depth is None:
        depth = 0
    w1set = wn.synsets(w1)
    w2set = wn.synsets(w2)
    max_similarity = {'similarity': 0, 'w1': None, 'w2': None, 'depth': 0}
    end_loop = False
    similar_threshold = 0.8

    # TODO check all the individual synsets first and only then check for the nouns?
    if depth < 3:
        for w1s in w1set:
            for w2s in w2set:
                similarity = w1s.wup_similarity(w2s)
                if similarity:
                    max_similarity = get_new_max_similarity(max_similarity, similarity, w1s, w2s)
                    if max_similarity['similarity'] < similar_threshold:
                        w1noun, w2noun = get_nouns(w1s, w2s, w1, w1)
                        if w1noun and w2noun and w1noun != w2noun:
                            w1n = list(w1noun)[0].lemmas()[0].name()
                            w2n = list(w2noun)[0].lemmas()[0].name()
                            # print('increasing depth')
                            depth += 1
                            w1sn, w2sn, similarity, depth = check_similarity(w1n, w2n, depth)
                            max_similarity = get_new_max_similarity(max_similarity, similarity, w1sn, w2sn)
                            if max_similarity['similarity'] > similar_threshold:
                                end_loop = True
                                break
                    else:
                        # print('similarity over {}'.format(similar_threshold))
                        end_loop = True
                        break
            if end_loop:
                break

    return max_similarity['w1'], max_similarity['w2'], max_similarity['similarity'], depth


def get_nouns(w1s, w2s, w1, w2):
    w1noun = None
    w2noun = None
    if w1s.pos() == 'v' and w2s.pos() == 'v':
        w1noun = nounify(w1)
        w2noun = nounify(w2)
    elif w1s.pos() == 'v' and w2s.pos() == 'n':
        w1noun = nounify(w1)
        w2noun = {w2s}
    elif w1s.pos() == 'n' and w2s.pos() == 'v':
        w1noun = {w1s}
        w2noun = nounify(w2)
    return w1noun, w2noun


def nounify(verb_word):
    set_of_related_nouns = set()

    # TODO look into morphy. It does something with affixes to find a version of a word that is in wordnet
    # https://wordnet.princeton.edu/documentation/morphy7wn
    for lemma in wn.lemmas(wn.morphy(verb_word, wn.VERB), pos="v"):
        for related_form in lemma.derivationally_related_forms():
            for synset in wn.synsets(related_form.name(), pos=wn.NOUN):
                if wn.synset('person.n.01') in synset.closure(lambda s: s.hypernyms()):
                    set_of_related_nouns.add(synset)

    return set_of_related_nouns


def get_new_max_similarity(max_similarity, new, w1s, w2s):
    if new:
        # print(w1s, w2s, new)
        if max_similarity['similarity'] < new:
            max_similarity = {'similarity': new, 'w1': w1s, 'w2': w2s}
    return max_similarity


def get_lemmas(synsets):
    lemmas = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            lemmas.add(lemma)
    return lemmas


def position_based(cue, response):
    if '_' in response:
        response_split = response.split('_')
        if cue in response:
            cue_index = response.index(cue)

            if cue_index == 0 and response.index(response) == len(cue) + 1:
                return 2, 'consecutivexy'
            elif cue_index > 0 and len(response_split) == 2:
                return 2, 'consecutiveyx'
            else:
                return 2, 'othercollocational'

    return 0, None


def form_based(cue, response):
    if cue == response:
        return 3, 'exact'

    # check to see if y is x plus or minux an affix
    if affix_check(cue, response):
        return 3, 'affix'

    remainder = response.split(cue)
    if len(remainder) == 2:
        print(cue, response, 'other_form')
        return 3, 'other_form'

    return 0, None


# Soundex similarity check
def sound_based(cue, response):
    # Try to get a cue soundex
    cue_decoded = unidecode.unidecode(cue)
    cue_soundex = jellyfish.soundex(cue_decoded)
    # cue_letters = re.sub(r'[0-9]', '', cue_soundex)
    # cue_numbers = cue_soundex.replace(cue_letters, '')

    # Try to get a response soundex
    response_decoded = unidecode.unidecode(response.replace('_', ''))
    response_soundex = jellyfish.soundex(response_decoded)
    # response_letters = re.sub(r'[0-9]', '', response_soundex)
    # response_numbers = response_soundex.replace(response_letters, '')

    if cue_soundex == response_soundex:
        return 3, 'soundex'

    if jellyfish.levenshtein_distance(cue, response) < 3:
        print(cue, response, 'levenshtein')
        return 3, 'levenshtein'

    return 0, None


def erratic_association(cue, response):
    return


def get_wn_relations(wn_obj_set, existing=None):
    if existing is None:
        existing = set()
    if not isinstance(wn_obj_set, set):
        raise Exception('wn_obj_list must be a set got {}'.format(type(wn_obj_set)))
    final_obj_list = []
    prune_list = []
    for wn_obj in wn_obj_set:
        prune = False
        for relation in wn_methods:
            if hasattr(wn_obj, relation):
                sub_wn_obj_list = getattr(wn_obj, relation)()
                if sub_wn_obj_list:
                    if set(sub_wn_obj_list) not in existing:
                        final_obj_list.extend(sub_wn_obj_list)
                    else:
                        prune = True

        if prune:
            prune_list.append(wn_obj)

    return set(final_obj_list), set(prune_list)


def affix_check(cue, response):
    if cue in response:
        remainder = response.split(cue)
        for r in remainder:
            if r in AFFIXES:
                return True


if __name__ == "__main__":
    main()
