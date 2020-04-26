from backend.app.coding import check_syntagmatic, check_phonology, check_affix, relation_coding, basic_coding


def test_check_syntagmatic():
    cue = 'back'
    response = 'and_forth'
    response_split = response.split('_')
    syntagmatic, type = check_syntagmatic(cue, response, response_split)
    assert (syntagmatic is True)
    assert (type == 'consecutivexy-cue_not_entered')

    cue = 'x'
    response = 'x_y'
    response_split = response.split('_')
    syntagmatic, type = check_syntagmatic(cue, response, response_split)
    assert (syntagmatic is True)
    assert (type == 'consecutivexy')

    cue = 'x'
    response = 'y_x'
    response_split = response.split('_')
    syntagmatic, type = check_syntagmatic(cue, response, response_split)
    assert (syntagmatic is True)
    assert (type == 'consecutiveyx')

    cue = 'cat'
    response = 'its_raining_cats_and_dogs'
    response_split = response.split('_')
    syntagmatic, type = check_syntagmatic(cue, response, response_split)
    assert (syntagmatic is True)
    assert (type == 'othercollocational')


def test_check_phonology():
    cue = 'elastic'
    response = 'plastic'
    similar, type = check_phonology(cue, response)
    assert (similar)
    assert (type == 'soundex')

    cue = 'goon'
    response = 'spoon'
    similar, type = check_phonology(cue, response)
    assert (similar)
    assert (type == 'soundex')

    cue = 'bored'
    response = 'bore'
    similar, type = check_phonology(cue, response)
    assert (similar)
    assert (type == 'levenshtein')


def test_check_affix():
    cue = 'respondent'
    response = 'responded'
    affix = check_affix(cue, response)
    assert (affix)

    cue = 'light'
    response = 'lighter'
    affix = check_affix(cue, response)
    assert (affix)


def test_relation_coding():
    cue = 'poor'
    response = 'rich'
    relations, totals = relation_coding(cue, response)
    assert (relations['antonyms'])
    assert (totals['antonyms'] == 1)


def test_basic_coding():
    cue = "Blahblahblah'"
    response = 'Chat_and_business'
    basic = basic_coding(cue, response, response_raw=response, response_split=response.split('_'))
    assert (basic['length'] == 17)
    assert (basic['syllables'] == 5)
    assert (basic['syllables_diff'] == 2)
    assert (basic['capitalised'])
    assert (basic['punctuation'])
    assert (basic['number_of_words'] == 3)


