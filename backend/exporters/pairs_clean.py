from backend.lib.db import get_cue_response_pairs
from backend.settings import DATA_FOLDER


def main():
    pairs = {''.join([x['cue_clean'], "\t", x['response_clean'], '\n']) for x in get_cue_response_pairs()}
    with open(''.join([DATA_FOLDER, '/cue_response_pairs.txt']), 'w') as f:
        for pair in sorted(list(pairs)):
            f.write(pair)


if __name__ == "__main__":
    main()
