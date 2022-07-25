import datetime

import dataset
import charts

def resources():
    df = dataset.read_methods()
    # print(df.columns)
    c = 0
    for idx, row in df.iterrows():
        res = dataset.read_resources_of_method(row['run_id'], row['start_time'], row['duration'])
        row['cpu_percent'] = res['cpu_percent'].mean()
        row['mem_percent'] = res['mem_percent'].mean()
        row['rss'] = res['rss'].mean()
        row['hwm'] = res['hwm'].mean()
        row['data'] = res['data'].mean()
        row['stack'] = res['stack'].mean()
        row['locked'] = res['locked'].mean()
        row['swap'] = res['swap'].mean()
        row['read_count'] = res['read_count'].mean()
        row['write_count'] = res['write_count'].mean()
        row['read_bytes'] = res['read_bytes'].mean()
        row['write_bytes'] = res['write_bytes'].mean()
        row['minor_faults'] = res['minor_faults'].mean()
        row['major_faults'] = res['major_faults'].mean()
        row['child_minor_faults'] = res['child_minor_faults'].mean()
        row['child_major_faults'] = res['child_major_faults'].mean()
        row['child_major_faults'] = res['child_major_faults'].mean()
        c += 1
        print(datetime.datetime.now(), c)

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


