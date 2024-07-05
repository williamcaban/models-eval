#
##
# pip install psycopg2-binary
##
import psycopg2
import sys

PG_DBNAME = "qa_dataset"
PG_DBUSER = "llm_eval"
PG_DBPASS = "llm_eval"
PG_DBHOST = "127.0.0.1"
PG_DBPORT = "5432"

DEBUG = False

# -- Create dbuser
# CREATE ROLE llm_eval WITH
# 	LOGIN
# 	NOSUPERUSER
# 	CREATEDB
# 	NOCREATEROLE
# 	INHERIT
# 	NOREPLICATION
# 	CONNECTION LIMIT -1
# 	PASSWORD 'xxxxxx';

def conn():
    """
    Connects to evaluation database and return a connection object
    """
    ## Try connecting to database
    try:
        conn = psycopg2.connect(user=PG_DBUSER,
                                password=PG_DBPASS,
                                host=PG_DBHOST,
                                port=PG_DBPORT,
                                database=PG_DBNAME)
        conn.autocommit = True
    except psycopg2.OperationalError as e:
        if e.args[0].find("does not exist") >= 0:
            # db does not exist
            print(f"Error database {PG_DBNAME} does not exist.\n")
            sys.exit()
        if (conn):
            #cur.close()
            conn.close()
            print("PostgreSQL connection is closed")

    # Fetch version information
    cur = conn.cursor()
    cur.execute("SELECT version();")
    record = cur.fetchone()
    if DEBUG is True:
        print(f"DEBUG: Connecting to {record[0].split(',')[0]}")
    cur.close()

    return conn

#
# END Of FILE
#