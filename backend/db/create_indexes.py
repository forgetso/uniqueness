from backend.db.db import connect


def main():
    conn = connect()
    cur = conn.cursor()
    query = """DROP INDEX IF EXISTS idx_source"""
    cur.execute(query)
    query = """DROP INDEX IF EXISTS idx_corrections"""
    cur.execute(query)
    query = """DROP INDEX IF EXISTS idx_coding"""
    cur.execute(query)
    query = """CREATE INDEX idx_source ON responses (source);"""
    cur.execute(query)
    query = """CREATE INDEX idx_corrections ON corrections (response_id, type);"""
    cur.execute(query)
    query = """CREATE INDEX idx_coding ON coding (cue, response);"""
    cur.execute(query)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
