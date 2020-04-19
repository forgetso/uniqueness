from backend.app.coding import check_syntagmatic, check_phonology, check_affix


def test_check_syntagmatic():
    # cue = 'back'
    # response = 'and_forth'
    # syntagmatic, type = check_syntagmatic(cue, response)
    # assert (syntagmatic is True)
    # assert (type == 'consecutivexy')

    cue = 'x'
    response = 'x_y'
    syntagmatic, type = check_syntagmatic(cue, response)
    assert (syntagmatic is True)
    assert (type == 'consecutivexy')

    cue = 'x'
    response = 'y_x'
    syntagmatic, type = check_syntagmatic(cue, response)
    assert (syntagmatic is True)
    assert (type == 'consecutiveyx')

    cue = 'cat'
    response = 'its_raining_cats_and_dogs'
    syntagmatic, type = check_syntagmatic(cue, response)
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
    assert(affix)

    cue = 'light'
    response = 'lighter'
    affix = check_affix(cue, response)
    assert(affix)

