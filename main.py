import datetime
import sys

import matplotlib
import pandas

import charts
import dataset
import pandas as pd
import matplotlib.pyplot as plt

from diff import methods_diff
from stats import get_group_statistics, get_statistics
from statsmodels.graphics.tsaplots import plot_acf

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

def commits_avg(df, col):

    df_methods = df.groupby(['commit_hash', 'class_name', 'method_name'])
    # print(df_methods.head())
    df_methods = df_methods.aggregate('mean')
    # print(df_methods)

    df_classes = df_methods.groupby(['commit_hash', 'class_name'])
    df_classes = df_classes.aggregate(col['measure'])
    # print(df_classes)
    df_res = pd.pivot_table(df_classes, values=col['name'],
                   index=['class_name'],
                   columns='commit_hash')
    return df_res


def calls(df):
    # conns = {}
    calls = pd.DataFrame(columns=['commit_hash', 'class_name_source', 'method_name_source', 'class_name_dest',
                                  'method_name_dest', 'own_duration', 'cumulative_duration'])
    # print(df.head())
    for index, row in df.iterrows():
        commit_hash = row['commit_hash']
        class_name_dest = row['class_name']
        method_name_dest = row['method_name']
        caller_id = row['caller_id']
        if caller_id:
            caller = df.loc[(df['commit_hash'] == commit_hash) & (df['id'] == caller_id)]
            if len(caller):
                class_name_source = caller['class_name'].iloc[0]
                method_name_source = caller['method_name'].iloc[0]
            calls.loc[len(calls.index)] = [commit_hash, class_name_source, method_name_source, class_name_dest,
                                    method_name_dest, row['own_duration'], row['cumulative_duration']]

    calls.to_csv('results/calls.csv', index=False)
    calls_gb = calls.groupby(['commit_hash', 'class_name_source', 'class_name_dest'])['class_name_source', 'class_name_dest'].count()
    calls_gb.to_csv('results/calls_gb.csv', index=False)




if __name__ == '__main__':
    pd.options.display.max_colwidth
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
    csv = 'data/bcel5.csv'
    df = pandas.read_csv(csv)
    df['commit_hash_short'] = df.commit_hash.str[:6]
    df['class_name_short'] = df.class_name.str[30:] #replace('src/main/java/org/apache/bcel/', '')
    df['method_name_short'] = df.method_name.str.split('org.apache.bcel.').str[-1]
    df.method_name_short = df.method_name_short.str.split(' throws').str[0]

    # for i, v in df.loc[df['class_name'] == 'src/main/java/org/apache/bcel/util/AbstractClassPathRepository.java'].iterrows():
    #     print(v['method_name'])


    # for col in cols:
        # print(col['name'])
        # print(df[[col['name']]].describe())
        # get_statistics(df, col['name'])
        # get_group_statistics(df, 'commit_hash', col['name'])
        # df_col = dataset.commits_avg(df, col)
        # charts.violin(df_col, col)
        # print(df2.head())
        # df3 = df2
        # plot_acf(df2)

        #surface

        # df_commits_classes = dataset.commits_classes_avg(df)
        # print(df_commits_classes.columns)
        # print(df_commits_classes.head())
        # charts.surface(df_commits_classes, col)
        # charts.lines(df[['commit_hash_short', 'class_name', col['name']]], col)


        # charts.multiple_area(df[['commit_hash_short', 'class_name_short', col['name']]], col)

        # df.boxplot()
        # plt.savefig('results/' + col + '.pdf')


    # waterfall chart
    commit = 'fa271c5d7ed94dd1d9ef31c52f32d1746d5636dc'

    # ok:
    # class_name = 'src/main/java/org/apache/bcel/classfile/AnnotationElementValue.java'
    # method_name = 'final int org.apache.bcel.classfile.ElementValue.getType()'

    # class_name = 'src/test/java/org/apache/bcel/util/ClassPathRepositoryTestCase.java'
    # method_name = 'private void org.apache.bcel.util.ClassPathRepositoryTestCase.verifyCaching(org.apache.bcel.util.AbstractClassPathRepository) throws java.lang.ClassNotFoundException'

    # generic/MethodGenTestCase.java  # generic.MethodGenTestCase.testRemoveLocalVariables()
    class_name = 'src/test/java/org/apache/bcel/generic/MethodGenTestCase.java'
    method_name = 'generic.MethodGenTestCase.testRemoveLocalVariables()'
    #error:
    # class_name = 'src/main/java/org/apache/bcel/BCELBenchmark.java'
    # method_name = 'public void baseline(Blackhole bh) throws IOException'

    # class_name = 'src/test/java/org/apache/bcel/util/ClassPathRepositoryTestCase.java'
    # method_name = 'public void org.apache.bcel.util.ClassPathRepositoryTestCase.testClassPathRepository() throws java.lang.ClassNotFoundException,java.io.IOException'

    # class_name = 'src/test/java/org/apache/bcel/verifier/VerifierReturnTestCase.java'
    # method_name = 'public void org.apache.bcel.verifier.VerifierReturnTestCase.testInvalidReturn()'
    # method = df[(df['commit_hash'] == commit) & (df['class_name'] == class_name) & (df['method_name'].str.startswith(method_name))]
    # print(method['method_name'])
    # method_name = str(method['method_name'])
    #
    # print(method_name)
    # if method['method_name'].any():
    #     charts.waterfall_method(df, commit, class_name, method)
    # else:
    #     print('No method found!')

    # calls(df)


    class_name_df = df[(df['class_name_short'].str.contains("TestCase.java")) & (df['commit_hash'] == commit)].drop_duplicates(subset=['class_name', 'method_name'])
    for idx, tc in class_name_df.iterrows():
        charts.waterfall_method(df, commit, tc['class_name'], tc['method_name'])

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


    # plt.show()

    # methods_diff()
