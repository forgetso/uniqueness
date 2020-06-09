from symspellpy import SymSpell, Verbosity
import re
import jellyfish
import pkg_resources

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_bigram_dictionary(dictionary_path, 0, 2)
CORRECTIONS_LIMIT = 10


class CueResponseSpelling:
    def __init__(self, cue, response):
        # the word we're trying to correct
        self.response = response
        # optional cue word for reducing the number of corrections returned. e.g. if someone is presented with
        # foggy as a cue word and they responded with grogy then the cue will help to identify that the response
        # is misspelled and should probably be groggy
        self.cue = cue
        if not isinstance(self.response, str):
            raise TypeError('Response must be a string: {}'.format(response))

    def get_corrections(self):
        corrections_list = []
        if len(self.response) == 0:
            return corrections_list
        corrections = self.check_spelling(self.response)
        if corrections:
            # this class can be used to check individual words as well, the cue is optional
            if isinstance(self.cue, str) and len(self.cue) > 0:
                corrections = self.reduce_correction_options(self.cue, self.response, corrections)
            corrections_list = sorted(list(corrections))
        return corrections_list

    def check_spelling(self, word):
        # checks the spelling of a word and returns the spelling suggestions if a mistake was detected
        word_reduced = self.remove_extra_chars(word)
        suggestions = sym_spell.lookup(word_reduced, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            if suggestions[0].term != word:
                return suggestions
        return None

    def remove_extra_chars(self, text):
        # replaces any instance of 3 or more of the same character with a single character
        result = text
        for char in set(text):
            raw_regex = ''.join([re.escape(char), '{3,}'])
            pattern = re.compile(raw_regex)
            match_result = re.findall(pattern, text)
            if match_result:
                match_result_len = len(match_result[0])
                if match_result_len > 2:
                    result = text.replace(char * match_result_len, char)
        return result

    # TODO restrict the number of spelling corrections returned to the page (perhaps phonetically?)
    def reduce_correction_options(self, cue, response, corrections):
        new_corrections = corrections
        if cue[0] == response[0]:
            new_corrections = [x for x in corrections if x.term[0] == cue[0]]
        if len(new_corrections) > CORRECTIONS_LIMIT:
            cue_phonology = jellyfish.soundex(cue)
            new_corrections = [x for x in new_corrections if cue_phonology == jellyfish.soundex(x.term)]
        return new_corrections

    def sort_corrections(self, corrections):
        # place any corrections that begin with the same character at the beginning of the list in alphabetical order
        top_of_list = []
        bottom_of_list = []
        for correction in corrections:
            if correction[0] == self.cue[0]:
                top_of_list.append(correction)
            else:
                bottom_of_list.append(correction)
        return sorted(top_of_list).extend(sorted(bottom_of_list))
