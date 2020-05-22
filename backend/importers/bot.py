from backend.db.db import get_part_of_speech, get_cue_response_pairs, execute_values_wrap, delete_responses
import random
import argparse
from datetime import datetime as dt
from backend.helpers.string import clean_word
import numpy as np
from backend.settings import SEMANTIC_SPACE_FOLDER
PARTS_OF_SPEECH = get_part_of_speech()
semantic_space_data = np.load('{}/semantic_space.npz'.format(SEMANTIC_SPACE_FOLDER))
ENGLISH1000_VOCAB = [x.decode('utf8') for x in semantic_space_data['english1000_vocab']]

def main():
    all_pairs = [x for x in get_cue_response_pairs()]
    bot_data = create_bot_data(min(len(all_pairs), 20000))
    delete_responses(source='bot_rand')
    execute_values_wrap('responses', tuple(bot_data[0].keys()), [tuple(x.values()) for x in bot_data])


def create_bot_data(length):
    print('Creating {} dummy pairs'.format(length))
    bot_data = []
    words = ENGLISH1000_VOCAB
    for i in range(0, length - 1):
        template = {'cue': None,
                    'response': None,
                    'source': 'bot_rand',
                    'user_id': 'bot_rand',
                    'created': dt.utcnow(),
                    'time_duration': 0
                    }
        rand_x = words[random.randint(0, len(words) - 1)]
        rand_y = words[random.randint(0, len(words) - 1)]
        template['cue'] = rand_x
        template['response'] = rand_y
        template['cue_clean'] = clean_word(rand_x)
        template['response_clean'] = clean_word(rand_y)
        bot_data.append(template)
    return bot_data


def setup():
    parser = argparse.ArgumentParser(description='Import various types of word data')
    args = parser.parse_args()
    main()


if __name__ == "__main__":
    setup()
