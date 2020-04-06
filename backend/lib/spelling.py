from spellchecker import SpellChecker
import re

spell = SpellChecker()


def check_spelling(word):
    # checks the spelling of a word and returns the spelling candidates if a mistake was detected
    word_reduced = remove_extra_chars(word)
    mispelled = spell.unknown([word_reduced])
    for word in mispelled:
        if spell.correction(word):
            return spell.candidates(word)
    return None


def remove_extra_chars(text):
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
