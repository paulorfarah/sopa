import datetime

import mysql.connector as connection
import pandas as pd

def read_commits(db):
    try:
        mydb = connection.connect(host="localhost", database=db, user="root", passwd="password", use_pure=True)
        # query = "Select * from commits;"
        query = "SELECT c.committer_date, commit_hash, f.name AS classname, m.name AS methodName, AVG(m.own_duration) AS duration "
        query += "FROM commits AS c "
        query += "INNER JOIN files AS f ON f.commit_id=c.id "
        query += "INNER JOIN methods AS m ON m.file_id=f.id "
        query += "GROUP BY commit_hash, classname "
        query += "ORDER BY c.committer_date, f.name ;"
        result_dataFrame = pd.read_sql(query, mydb)
        mydb.close() #close the connection
        return result_dataFrame
    except Exception as e:
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
        print(str(e))

def read_resources_of_method(run_id, start_time, end_time):
    ds = pd.DataFrame()
    try:
        mydb = connection.connect(host="localhost", database="perfrt", user="root", passwd="password", use_pure=True)
        query = "SELECT * FROM perfrt.resources WHERE run_id=" + str(run_id) + " AND created_at BETWEEN '" + str(start_time) + "' AND '" + str(end_time) +"';"
        ds = pd.read_sql(query, mydb)
        mydb.close()
        return ds
    except Exception as e:
        mydb.close()
        print("Error querying resources: " + str(e))
        return ds

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


def averages():
    try:
        mydb = connection.connect(host="localhost", database='perfrt', user="root", passwd="password", use_pure=True)
        query = ("SELECT c.committer_date, commit_hash, f.name AS classname, m.name AS methodName, "
                "AVG(m.own_duration), STD(m.own_duration) AS duration, "
                "AVG(res.cpu_percent), STD(res.cpu_percent), AVG(res.mem_percent), STD(res.mem_percent), "
                "AVG(res.rss), STD(res.rss), AVG(res.hwm), STD(res.hwm), AVG(res.data), STD(res.data), AVG(res.stack), STD(stack), AVG(res.locked), "
                "STD(res.locked), AVG(res.swap), STD(res.swap), "
                "AVG(res.read_count), STD(res.read_count), AVG(res.write_count), STD(res.write_count), AVG(res.read_bytes), "
                 "STD(res.read_bytes), AVG(res.write_bytes), STD(res.write_bytes), "
                "AVG(res.minor_faults), STD(res.minor_faults), AVG(res.major_faults), STD(res.major_faults), "
                "AVG(res.child_minor_faults), STD(res.child_minor_faults), AVG(res.child_major_faults), STD(res.child_major_faults) "
                "FROM perfrt.commits AS c "
                "INNER JOIN perfrt.files AS f ON f.commit_id=c.id "
                "INNER JOIN perfrt.methods AS m ON m.file_id=f.id "
                "INNER JOIN perfrt.runs AS r ON m.run_id = r.id "
                "INNER JOIN perfrt.resources res ON res.run_id = r.id "
                "GROUP BY commit_hash, classname, methodName "
                "ORDER BY c.committer_date, f.name  ")

        ds = pd.read_sql(query, mydb)
        mydb.close() #close the connection
        return ds
    except Exception as e:
        print(str(e))