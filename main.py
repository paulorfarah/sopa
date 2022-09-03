import datetime
import sys

import pandas

import charts
import dataset
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def resources():
    db = "maven-project"
    df = dataset.read_commits(db)



    # col = 'duration'
    # charts.pdf_cdf(df, col)
    # df = dataset.averages()
    # print(df.columns)
    # print(df.columns)
    # c = 0
    # print(df['cumulative_duration'].head())
    # df['method_ended_at'] = df['method_started_at'] + pd.to_timedelta(df['cumulative_duration'], unit='s')

    # cpu = []
    # for idx in range(len(df)):
        # print("idx:" + str(idx))
        # print(df['run_id'].iloc[idx], df['method_started_at'].iloc[idx], df['method_ended_at'].iloc[idx])
        # res = dataset.read_resources_of_method(df['run_id'].iloc[idx], df['method_started_at'].iloc[idx], df['method_ended_at'].iloc[idx])
        # print(len(res))
        # try:
        #     # df.iloc[idx]['cpu_percent'] = res['cpu_percent'].mean()
        #     cpu.append(res['cpu_percent'].mean())
        # except:
        #     print('error in mean')
        #     sys.exc_info()
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
        # c += 1
        # print(datetime.datetime.now(), c)

    # df['cpu_percent'] = cpu
    # fig, ax = plt.subplots(figsize=(10, 8))
    # df.plot.line(x='committer_date', y='cpu_percent', color='crimson', ax=ax)
    # plt.ylabel("CPU Percent")
    # plt.show()

    # dfg = df['classname', 'committer_date'].groupby(['classname']).mean()#.T.plot.line(x='committer_date', color='crimson', ax=ax)
    # print(dfg.head())
    df.to_csv('resources.csv')

def commits_avg(csv, col):
    df = pandas.read_csv(csv)
    df_methods = df.groupby(['committer_date', 'class_name', 'method_name'])
    # print(df_methods.head())
    df_methods = df_methods.aggregate('mean')
    # print(df_methods)

    df_classes = df_methods.groupby(['committer_date', 'class_name'])
    df_classes = df_classes.aggregate(col['measure'])
    print(df_classes)
    df_res = pd.pivot_table(df_classes, values=col['name'],
                   index=['class_name'],
                   columns='committer_date')
    return df_res

if __name__ == '__main__':
#     cols = ['own_duration', 'cumulative_duration', 'AVG(active)', 'AVG(available)', 'AVG(buffers)', 'AVG(cached)',
#             'AVG(child_major_faults)', 'AVG(child_minor_faults)', 'AVG(commit_limit)', 'AVG(committed_as)',
# 'AVG(cpu_percent)', 'AVG(data)', 'AVG(dirty)', 'AVG(free)', 'AVG(high_free)', 'AVG(high_total)', 'AVG(huge_pages_total)', 'AVG(huge_pages_free)', 'AVG(huge_pages_total)',
# 'AVG(hwm)', 'AVG(inactive)', 'AVG(laundry)', 'AVG(load1)', 'AVG(load5)', 'AVG(load15)', 'AVG(locked)', 'AVG(low_free)', 'AVG(low_total)', 'AVG(major_faults)', 'AVG(mapped)', 'AVG(mem_percent)',
# 'AVG(minor_faults)', 'AVG(page_tables)', 'AVG(pg_fault)', 'AVG(pg_in)', 'AVG(pg_maj_faults)', 'AVG(pg_out)', 'AVG(read_bytes)', 'AVG(read_count)', 'AVG(rss)', 'AVG(shared)', 'AVG(sin)', 'AVG(slab)',
# 'AVG(sout)', 'AVG(sreclaimable)', 'AVG(stack)', 'AVG(sunreclaim)', 'AVG(swap)', 'AVG(swap_cached)', 'AVG(swap_free)', 'AVG(swap_total)', 'AVG(swap_used)', 'AVG(swap_used_percent)',
# 'AVG(total)', 'AVG(used)', 'AVG(used_percent)', 'AVG(vm_s)', 'AVG(vmalloc_chunk)', 'AVG(vmalloc_total)', 'AVG(vmalloc_used)', 'AVG(wired)', 'AVG(write_back)', 'AVG(write_back_tmp)',
# 'AVG(write_bytes)', 'AVG(write_count)'
# ]
    cols = [
        {'name': 'own_duration', 'unit': 'ns', 'measure': 'sum'},
        {'name': 'cumulative_duration', 'unit': 'ns', 'measure': 'sum'},
        {'name': 'AVG(cpu_percent)', 'unit': '%', 'measure': 'mean'},
        {'name': 'AVG(mem_percent)', 'unit': '%', 'measure': 'mean'},
        ]
    easymock = 'data/maven-project.csv'
    for col in cols:
        df = commits_avg(easymock, col)
        charts.violin(df, col)
        # df.boxplot()
        # plt.savefig('results/' + col + '.pdf')
    # charts.gantt()
    # commits_avg('data/nailgun.csv')
    # resources()

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

    plt.show()

