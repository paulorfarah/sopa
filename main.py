import datetime
import sys

import matplotlib
import pandas

import charts
import dataset
import pandas as pd
import matplotlib.pyplot as plt

import resources
from diff import methods_diff
from stats import get_group_statistics, get_statistics
from statsmodels.graphics.tsaplots import plot_acf

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def commits_avg(df, col):

    df_methods = df.groupby(['commit_hash', 'class_name', 'method_name'])
    df_methods = df_methods.aggregate('mean')

    df_classes = df_methods.groupby(['commit_hash', 'class_name'])
    df_classes = df_classes.aggregate(col['measure'])
    df_res = pd.pivot_table(df_classes, values=col['name'],
                   index=['class_name'],
                   columns='commit_hash')
    return df_res

def testcases_by_commit_waterfall(df, commit):
    class_name_df = df[(df['class_name_short'].str.contains("TestCase.java")) & (df['commit_hash'] == commit)].drop_duplicates(subset=['class_name', 'method_name'])
    for idx, tc in class_name_df.iterrows():
        charts.testcase_waterfall(df, commit, tc['class_name'], tc['method_name'])


if __name__ == '__main__':
    pd.options.display.max_colwidth

    cols = [
        {'name': 'own_duration', 'unit': 'ns', 'measure': 'sum'},
        {'name': 'cumulative_duration', 'unit': 'ns', 'measure': 'sum'},
        {'name': 'AVG(cpu_percent)', 'unit': '%', 'measure': 'mean'},
        {'name': 'AVG(mem_percent)', 'unit': '%', 'measure': 'mean'},
        {'name': 'AVG(swap)', '': 'Bytes', 'measure': 'mean'},
        {'name': 'AVG(swap_cached)', '': 'Bytes', 'measure': 'mean'},
        {'name': 'AVG(swap_free)', '': 'Bytes', 'measure': 'mean'},
        {'name': 'AVG(swap_total)', '': 'Bytes', 'measure': 'mean'},
        {'name': 'AVG(swap_used)', 'unit': '%', 'measure': 'mean'},
        {'name': 'AVG(swap_used_percent)', 'unit': '%', 'measure': 'mean'},
        {'name': 'AVG(rss)', 'unit': 'Bytes', 'measure': 'mean'},
        {'name': 'AVG(hwm)', 'unit': 'Bytes', 'measure': 'mean'},
        {'name': 'AVG(load1)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(load5)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(load15)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(mapped)', 'unit': 'Bytes', 'measure': 'mean'},
        {'name': 'AVG(locked)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(cached)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(read_bytes)', 'unit': 'Bytes', 'measure': 'mean'},
        {'name': 'AVG(read_count)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(write_bytes)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(write_count)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(data)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(dirty)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(free)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(high_total)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(data)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(huge_pages_total)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(huge_pages_free)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(active)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(inactive)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(available)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(buffers)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(data)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(major_faults)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(minor_faults)', '': 'Bytes', 'measure': 'mean'},
        {'name': 'AVG(child_major_faults)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(child_minor_faults)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(commit_limit)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(committed_as)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(laundry)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(low_free)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(low_total)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(page_tables)', '': '', 'measure': 'mean'},
        {'name': 'AVG(pg_fault)', '': '', 'measure': 'mean'},
        {'name': 'AVG(pg_in)', '': '', 'measure': 'mean'},
        {'name': 'AVG(pg_maj_faults)', '': '', 'measure': 'mean'},
        {'name': 'AVG(pg_out)', '': '', 'measure': 'mean'},
        {'name': 'AVG(shared)', '': 'Bytes', 'measure': 'mean'},
        {'name': 'AVG(sin)', '': '', 'measure': 'mean'},
        {'name': 'AVG(slab)', '': '', 'measure': 'mean'},
        {'name': 'AVG(sout)', '': '', 'measure': 'mean'},
        {'name': 'AVG(sreclaimable)', '': '', 'measure': 'mean'},
        {'name': 'AVG(stack)', '': 'Bytes', 'measure': 'mean'},
        {'name': 'AVG(sunreclaim)', '': '', 'measure': 'mean'},
        {'name': 'AVG(sunreclaim)', '': '', 'measure': 'mean'},
        {'name': 'AVG(total)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(used)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(used_percent)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(vm_s)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(data)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(vmalloc_chunk)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(vmalloc_total)', 'unit': 'Bytes', 'measure': 'mean'},
        {'name': 'AVG(vmalloc_used)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(data)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(wired)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(write_back)', 'unit': '', 'measure': 'mean'},
        {'name': 'AVG(write_back_tmp)', 'unit': '', 'measure': 'mean'},
    ]

    df = resources.avg()
    df['commit_hash_short'] = df.commit_hash.str[:7]
    df['class_name_short'] = df.class_name.str[30:] #replace('src/main/java/org/apache/bcel/', '')
    df['method_name_short'] = df.method_name.str.split('org.apache.bcel.').str[-1]
    df.method_name_short = df.method_name_short.str.split(' throws').str[0]

    #violin chart
    commit = 'a9c13ede0e565fae0593c1fde3b774d93abf3f71'
    class_name = 'src/main/java/org/apache/bcel/classfile/AnnotationElementValue.java'
    method_name = 'final int org.apache.bcel.classfile.ElementValue.getType()'

    for col in cols:
        get_statistics(df, col['name'])
        get_group_statistics(df, 'commit_hash', col['name'])

        ### violin chart by commit
        df_commits = dataset.commits_avg(df, col)
        charts.violin(df_commits, col)

        ### violin chart by method
        df_method = dataset.method_avg(df, class_name, method_name, col)
        charts.violin(df_method, col)

        ### testcases multiple area
        target_class = "PLSETestCase.java"
        tc_df = df[
            (df['class_name_short'].str.contains(target_class))]
        charts.multiple_area(tc_df[['commit_hash_short', 'method_name_short', col['name']]], col)

    # waterfall chart
    commit = 'a9c13ede0e565fae0593c1fde3b774d93abf3f71'
    class_name = 'src/test/java/org/apache/bcel/generic/MethodGenTestCase.java'
    method_name = 'generic.MethodGenTestCase.testRemoveLocalVariables()'

    ### testcases by commit waterfall chart
    testcases_by_commit_waterfall(df, commit)

