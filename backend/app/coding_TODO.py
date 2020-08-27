# TODO wordnet stuff with pandas? probably means rewriting or wrapping relation_coding to accept dataframes
# TODO consider the categories found in the Gallant Research:
#   (visual, tactile, outdoor, bodypart, number, place, violence, person, social, mental, time)
# TODO what to do about words joined with hyphens?
# TODO use split col throughout instead of splitting multiple times
# TODO add s2v check for syntagmatic
# TODO add proverb check for syntagmatic
# TODO check if the part of speech changes from cue to response
# TODO add comparisons between the other metrics instead of the metrics themselves (i.e. concreteness difference)
#   this is probably more valid than modelling on the response metrics as they will be indicative of the words
# TODO  contractions (don't)
# TODO  word is in ALL_WORDS
# TODO  encyclopaedic associations - e.g. if the cue is war and someone responds with the name of a particular war
# TODO  obscene words
# TODO  sexual references
# TODO  slang
# TODO  morpheme
# TODO  years (1960, 2010, etc.)
# TODO  cultural_reference (encyclopaedic):
#  [movies,
#   music,
#   famous people,
#   books (1984, farenheit 451),
#   poems,
#   computer games,
#   tv shows,
#   brands,
#   historical dates,
#   characters,
#   superheroes,
#   religion,
#   web sites,
#   web status codes (404),
#   emergency services numbers,
#   scientific numbers (365, pi, gravity, phi, etc.)
#       ]
# TODO food references
#  [cocktails,
#   recipes,
#   etc.
#   ]
#   The part of speech (Person, etc.) is a good start for deriving further information from the full cue/
#   response pair. For example, (portal, chell). Chell is picked up as a person but is a character in the
#   game, Portal.
# TODO  company reference [walmart, asda, etc.]
# TODO  geographical_reference [country, state, city, town, village, post code]
# TODO  corpus_similarity
# TODO  dictionary_similarity
