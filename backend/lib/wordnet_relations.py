from nltk.corpus import wordnet as wn
import nltk

synset_relations = [
    'derivationally_related_forms',
    'pertainyms',
    'instance_hypernyms',
    'hyponyms',
    'instance_hyponyms',
    'member_holonyms',
    'substance_holonyms',
    'part_holonyms',
    'member_meronyms',
    'substance_meronyms',
    'part_meronyms',
    'topic_domains',
    'region_domains',
    'usage_domains',
    'attributes',
    'entailments',
    'causes',
    'also_sees',
    'verb_groups',
    'similar_tos',
    'hypernyms',
    'root_hypernyms',
]

lemma_relations = [
    'antonyms',
    'derivationally_related_forms',
    'pertainyms'
]


class WordnetRelations():

    def __init__(self, cue, response):
        nltk.download('wordnet')
        self.cue = cue.lower()
        self.response = response.replace(' ', '_').replace('.', '').lower()
        self.synsets_cue = wn.synsets(self.cue)
        self.synsets_response = wn.synsets(self.response)
        # set the relations
        self.relations = {x: False for x in synset_relations}
        self.relations['synonym'] = self.get_synonym()

        for relation in synset_relations:
            self.relations[relation] = self.get_related(relation, response)
        for relation in lemma_relations:
            self.relations[relation] = self.get_related_lemmas(relation, response)
        if not self.relations['antonyms']:
            self.relations['antonyms'] = self.get_antonym(self.cue, self.response)
        self.relation_types = synset_relations + ['synonym'] + lemma_relations
        #TODO disallow root hypernym [Synset('entity.n.01')]

    def affix(self):
        return

    def get_synonym(self):
        intersection = self.check_intersection(self.synsets_cue, self.synsets_response)
        if intersection:
            return True
        return False

    def get_antonym(self, cue, response):
        # response has form X not e.g. have not, maybe_not, etc.
        if response in [' '.join([cue, 'not']), '_'.join([cue, 'not'])]:
            return True

        # response has form 'opposite of Y'
        try:
            if response.index('opposite of ') == 0:
                return True
        except ValueError:
            pass
        return False

    # lowest common hypernym could be a measure of relatedness
    # >> > wn.synsets('village')[1].hypernyms()[0].hypernyms()[0].hypernyms()
    # [Synset('region.n.03')]
    #
    # >> > wn.synsets('town')[0].hypernyms()[0].hypernyms()[0].hypernyms()[0].hypernyms()
    # [Synset('region.n.03')]

    def get_related(self, relation, response):
        result = False
        for cue_synset in self.synsets_cue:
            if hasattr(cue_synset, relation):
                cue_related = getattr(cue_synset, relation)()
                for response_synset in self.synsets_response:
                    response_related = getattr(response_synset, relation)()
                    intersection = self.check_intersection(cue_related, response_related)
                    if intersection:
                        result = True
                    if self.check_word_in_synsets(response, cue_related):
                        result = True
                if cue_related:
                    print('{} {}'.format(relation, cue_related))
        return result

    def get_related_lemmas(self, relation, response):
        for cue_synset in self.synsets_cue:
            for cue_lemma in cue_synset.lemmas():
                cue_related_lemmas = getattr(cue_lemma, relation)()
                if self.check_word_in_lemmas(response, cue_related_lemmas):
                    return True
                for response_synset in self.synsets_response:
                    for response_lemma in response_synset.lemmas():
                        response_related_lemmas = getattr(response_lemma, relation)()
                        if cue_related_lemmas and response_related_lemmas:
                            if self.check_intersection(getattr(cue_lemma, relation)(),
                                                       getattr(response_lemma, relation)()):
                                return True

    def check_word_in_synsets(self, word, synsets):
        for synset in synsets:
            word_in_lemmas = self.check_word_in_lemmas(word, synset.lemmas())
            if word_in_lemmas:
                return word_in_lemmas
        return None

    def check_word_in_lemmas(self, word, lemmas):
        for lemma in lemmas:
            if word == lemma.name():
                return lemma.name()
        return None

    def check_intersection(self, s1, s2):
        s1set = set(s1)
        s2set = set(s2)
        intersection = s1set.intersection(s2set)
        return intersection

    #########################~~TODO work out if any of the below is required ##########################################

    # def check_similarity(w1, w2, depth=None):
    #     if depth is None:
    #         depth = 0
    #     w1set = wn.synsets(w1)
    #     w2set = wn.synsets(w2)
    #     max_similarity = {'similarity': 0, 'w1': None, 'w2': None, 'depth': 0}
    #     end_loop = False
    #     similar_threshold = 0.8
    #
    #     # TODO check all the individual synsets first and only then check for the nouns?
    #     if depth < 3:
    #         for w1s in w1set:
    #             for w2s in w2set:
    #                 similarity = w1s.wup_similarity(w2s)
    #                 if similarity:
    #                     max_similarity = get_new_max_similarity(max_similarity, similarity, w1s, w2s)
    #                     if max_similarity['similarity'] < similar_threshold:
    #                         w1noun, w2noun = get_nouns(w1s, w2s, w1, w1)
    #                         if w1noun and w2noun and w1noun != w2noun:
    #                             w1n = list(w1noun)[0].lemmas()[0].name()
    #                             w2n = list(w2noun)[0].lemmas()[0].name()
    #                             # print('increasing depth')
    #                             depth += 1
    #                             w1sn, w2sn, similarity, depth = check_similarity(w1n, w2n, depth)
    #                             max_similarity = get_new_max_similarity(max_similarity, similarity, w1sn, w2sn)
    #                             if max_similarity['similarity'] > similar_threshold:
    #                                 end_loop = True
    #                                 break
    #                     else:
    #                         # print('similarity over {}'.format(similar_threshold))
    #                         end_loop = True
    #                         break
    #             if end_loop:
    #                 break
    #
    #     return max_similarity['w1'], max_similarity['w2'], max_similarity['similarity'], depth
    #
    #
    # def get_nouns(w1s, w2s, w1, w2):
    #     w1noun = None
    #     w2noun = None
    #     if w1s.pos() == 'v' and w2s.pos() == 'v':
    #         w1noun = nounify(w1)
    #         w2noun = nounify(w2)
    #     elif w1s.pos() == 'v' and w2s.pos() == 'n':
    #         w1noun = nounify(w1)
    #         w2noun = {w2s}
    #     elif w1s.pos() == 'n' and w2s.pos() == 'v':
    #         w1noun = {w1s}
    #         w2noun = nounify(w2)
    #     return w1noun, w2noun
    #
    #
    # def nounify(verb_word):
    #     set_of_related_nouns = set()
    #
    #     # TODO look into morphy. It does something with affixes to find a version of a word that is in wordnet
    #     # https://wordnet.princeton.edu/documentation/morphy7wn
    #     for lemma in wn.lemmas(wn.morphy(verb_word, wn.VERB), pos="v"):
    #         for related_form in lemma.derivationally_related_forms():
    #             for synset in wn.synsets(related_form.name(), pos=wn.NOUN):
    #                 if wn.synset('person.n.01') in synset.closure(lambda s: s.hypernyms()):
    #                     set_of_related_nouns.add(synset)
    #
    #     return set_of_related_nouns
    #
    #
    # def get_new_max_similarity(max_similarity, new, w1s, w2s):
    #     if new:
    #         # print(w1s, w2s, new)
    #         if max_similarity['similarity'] < new:
    #             max_similarity = {'similarity': new, 'w1': w1s, 'w2': w2s}
    #     return max_similarity
    #
    #
    # def get_lemmas(synsets):
    #     lemmas = set()
    #     for synset in synsets:
    #         for lemma in synset.lemmas():
    #             lemmas.add(lemma)
    #     return lemmas