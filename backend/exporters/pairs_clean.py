from backend.db.layer import get_cue_response_pairs
from backend.settings import DATA_FOLDER
import argparse
import random

def main(rand_amount=None):
    pairs = {''.join([x['cue_clean'], "\t", x['response_clean'], '\n']) for x in get_cue_response_pairs()}
    pairs = list(pairs)
    if rand_amount:
        pairs = random.choices(pairs, k=int(rand_amount))
    print(len(pairs))
    pairs = sorted(pairs)

    with open(''.join([DATA_FOLDER, '/cue_response_pairs.txt']), 'w') as f:
        for pair in sorted(list(pairs)):
            f.write(pair)

def setup():
    parser = argparse.ArgumentParser(description='Export pairs')
    parser.add_argument('--rand_amount', type=int, help="Export a random selection of cue response pairs")
    args = parser.parse_args()

    main(args.rand_amount)




if __name__ == "__main__":
    setup()
