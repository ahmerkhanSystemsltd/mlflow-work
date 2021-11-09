import sqlite3
from sqlite3 import Error
from sqlalchemy import create_engine
engine = create_engine('sqlite:///test.db')
database = "./test.db"

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    return conn

def select_all_tasks(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM names")
    rows = cur.fetchall()

    for row in rows:
        print(row)

def main():
    database = r"./test.db"
    # create a database connection
    conn = create_connection(database)
    with conn:
        print("Query 1 for testing")
        select_all_tasks(conn)

if __name__ == '__main__':
    main()


# def create_connection(db_file):
#     """ create a database connection to a SQLite database """

#     conn = None
#     try:
#         conn = sqlite3.connect(db_file)
#         print(sqlite3.version)
#     except Error as e:
#         print(e)
#     finally:
#         if conn:
#             conn.close()


# if __name__ == '__main__':
#     create_connection(r"./test.db")