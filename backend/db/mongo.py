import pymongo


def connect(database, server='127.0.0.1', port=27017):
    # server.local_bind_port
    connection = pymongo.MongoClient(server, port)
    db = connection[database]
    return db