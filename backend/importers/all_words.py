import json
from backend.db.layer import connect, execute_values_wrap
from backend.lib.setup import db_setup
from backend.settings import ALL_WORDS_FILE


def main():
    target_file = ALL_WORDS_FILE
    process_data(target_file)


def process_data(target_file):
    words = []
    with open(target_file, 'r') as f:
        json_data = json.loads(f.read())
        for k, v in json_data.items():
            words.append([k])
    db_setup(all_words=True)
    save_all_words(words)


def save_all_words(records):
    conn = connect()
    cur = conn.cursor()
    execute_values_wrap(cur, 'all_words', ('word',), records)
    conn.commit()
    conn.close()


def setup():
    main()


if __name__ == "__main__":
    setup()
