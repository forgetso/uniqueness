from spellchecker import SpellChecker
import re
import jellyfish

spell = SpellChecker()
CORRECTIONS_LIMIT = 10


# TODO add all known words from wordassoc cues and other sources
# TODO work out if the response has spaces and check each word individually

class Spelling():
    def __init__(self, cue, response):
        self.cue = cue
        self.response = response

    def get_corrections(self):
        corrections = self.check_spelling(self.response)
        corrections_list = []
        if corrections:
            corrections = self.reduce_correction_options(self.cue, self.response, corrections)
            corrections_list = sorted(list(corrections))
        return corrections_list

    def check_spelling(self, word):
        # checks the spelling of a word and returns the spelling candidates if a mistake was detected
        word_reduced = self.remove_extra_chars(word)
        mispelled = spell.unknown([word_reduced])
        for word in mispelled:
            if spell.correction(word):
                return spell.candidates(word)
        return None

    def remove_extra_chars(self, text):
        # replaces any instance of 3 or more of the same character with a single character
        result = text
        for char in set(text):
            pattern = re.compile(''.join([char, '{3,}']))
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
            new_corrections = [x for x in corrections if x[0] == cue[0]]
        if len(new_corrections) > CORRECTIONS_LIMIT:
            cue_phonology = jellyfish.soundex(cue)
            new_corrections = [x for x in new_corrections if cue_phonology == jellyfish.soundex(x)]
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
