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
