import argparse
from backend.db.db import get_cue_response_pairs, save_corrections

def main(source=None):
    cur = get_cue_response_pairs(source=source)
    df = pd.DataFrame([x for x in cur], columns=[desc[0] for desc in cur.description])


def setup():
    parser = argparse.ArgumentParser(description='correct incorrectly spelled cues and responses')
    parser.add_argument('--source', default=None, help="code responses from a particular source only")
    args = parser.parse_args()
    if args.source:
        main(source=args.source)
    else:
        main()


if __name__ == "__main__":
    setup()
