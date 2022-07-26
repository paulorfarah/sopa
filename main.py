import datetime
import sys

import dataset
import pandas as pd

def resources():
    df = dataset.read_methods()
    print(df.columns)
    # print(df.columns)
    c = 0
    print(df['cumulative_duration'].head())
    df['method_ended_at'] = df['method_started_at'] + pd.to_timedelta(df['cumulative_duration'], unit='s')

    for idx in range(len(df)):
        print(df['run_id'].iloc[idx], df['method_started_at'].iloc[idx], df['method_ended_at'].iloc[idx])
        res = dataset.read_resources_of_method(df['run_id'].iloc[idx], df['method_started_at'].iloc[idx], df['method_ended_at'].iloc[idx])
        if res:
            print(res.head())
            try:
                df.iloc[idx]['cpu_percent'] = res['cpu_percent'].mean()
            except:
                sys.exc_info()
        # df['mem_percent'].iloc[idx] = res['mem_percent'].mean()
        # df['rss'].iloc[idx] = res['rss'].mean()
        # df['hwm'].iloc[idx] = res['hwm'].mean()
        # df['data'].iloc[idx] = res['data'].mean()
        # df['stack'].iloc[idx] = res['stack'].mean()
        # df['locked'].iloc[idx] = res['locked'].mean()
        # df['swap'].iloc[idx] = res['swap'].mean()
        # df['read_count'].iloc[idx] = res['read_count'].mean()
        # df['write_count'].iloc[idx] = res['write_count'].mean()
        # df['read_bytes'].iloc[idx] = res['read_bytes'].mean()
        # df['write_bytes'].iloc[idx] = res['write_bytes'].mean()
        # df['minor_faults'].iloc[idx] = res['minor_faults'].mean()
        # df['major_faults'].iloc[idx] = res['major_faults'].mean()
        # df['child_minor_faults'].iloc[idx] = res['child_minor_faults'].mean()
        # df['child_major_faults'].iloc[idx] = res['child_major_faults'].mean()
        # df['child_major_faults'].iloc[idx] = res['child_major_faults'].mean()
        c += 1
        print(datetime.datetime.now(), c)
    df.to_csv('resources.csv')

if __name__ == '__main__':
    resources()

    # df.dropna(inplace=True)
    # df.to_csv('results/commits.csv')
    # charts.boxplot(df)
    # charts.boxplot(dataset.mock_ds())
    # charts.correlation(dataset.mock_ds())

    # df = dataset.read_tsv("data/bcel.tsv")
    # df.dropna(inplace=True)
    # print(df)

    # desc = df.groupby(['commit_hash']).mean()
    # perc =[.20, .40, .60, .80]

    # include=['object', 'float', 'int']
    # desc = df.groupby(['commit_hash']).describe(percentiles=perc, include=include)
    # print(desc)
    # desc.to_csv('results/bcel.csv')
    # charts.correlation()


