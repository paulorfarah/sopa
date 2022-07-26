import datetime

import mysql.connector as connection
import pandas as pd

def read_commits():
    try:
        mydb = connection.connect(host="localhost", database='bcel', user="root", passwd="password", use_pure=True)
        # query = "Select * from commits;"
        query = "SELECT c.committer_date, commit_hash, f.name AS classname, m.name AS methodName, AVG(m.own_duration) AS duration "
        query += "FROM bcel.commits AS c "
        query += "INNER JOIN bcel.files AS f ON f.commit_id=c.id "
        query += "INNER JOIN bcel.methods AS m ON m.file_id=f.id "
        query += "GROUP BY commit_hash, classname "
        query += "ORDER BY c.committer_date, f.name ;"
        result_dataFrame = pd.read_sql(query, mydb)
        mydb.close() #close the connection
        return result_dataFrame
    except Exception as e:
        mydb.close()
        print(str(e))

def read_methods():
    try:
        mydb = connection.connect(host="localhost", database="perfrt", user="root", passwd="password", use_pure=True)
        query = "SELECT c.committer_date, commit_hash, f.name AS classname, r.id AS run_id, m.name AS method_name, "
        query += "m.caller_id, m.own_duration, m.cumulative_duration, m.created_at AS method_started_at "
        query += "FROM perfrt.commits AS c INNER JOIN perfrt.files AS f ON f.commit_id = c.id "
        query += "INNER JOIN perfrt.methods AS m ON m.file_id = f.id INNER JOIN perfrt.runs as r ON m.run_id = r.id "
        query += "WHERE m.finished = true ORDER BY c.committer_date, f.name"
        ds = pd.read_sql(query, mydb)
        mydb.close()
        return ds
    except Exception as e:
        mydb.close()
        print(str(e))

def read_resources_of_method(run_id, start_time, end_time):

    try:
        mydb = connection.connect(host="localhost", database="perfrt", user="root", passwd="password", use_pure=True)
        query = "SELECT * FROM perfrt.resources WHERE run_id=" + run_id + " AND created_at BETWEEN '" + start_time + "' AND '" + end_time +"';"
        ds = pd.read_sql(query, mydb)
        mydb.close()
        return ds
    except Exception as e:
        mydb.close()
        print(str(e))
def mock_ds():
    # initialise data of lists.
    data = {
        'committer_date': ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01', '2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01'],
        'commit_hash': ['1', '2', '3', '4', '1', '2', '3', '4', '1', '2', '3', '4', '1', '2', '3', '4'],
        'cpu_percent': [20, 21, 19, 18, 32, 99, 7, 17, 39, 182, 14, 19, 7, 1, 135, 19],
        'mem_percent': [20, 21, 19, 18, 32, 99, 7, 17, 39, 182, 14, 19, 7, 1, 135, 19],
        'changed': [True, False, False, False, True, False, False, False, False, True, False, False, False, True, True, False]
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    return df


def read_tsv(file):
    df = pd.read_csv(file, sep="\t")
    return df
