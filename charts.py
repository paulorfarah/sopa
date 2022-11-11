from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import datetime as dt
def violin(df, col):
    sns.set(style='whitegrid')

    # sns.boxplot(x='commit_hash', y='cpu_percent', data=df, palette="deep")

    fig, axs = plt.subplots(figsize=(10, 6))
    # sns.boxplot(x='commit_hash', y='duration', data=df, ax=axs[0], color='plum', width=0.5)
    sns.violinplot(data=df, color='plum', width=0.5)
    plt.title(col['name'], fontsize=14)
    plt.xlabel('date')
    plt.ylabel(col['unit'])

    locs, labels = plt.xticks()
    # x_ticks = []
    new_xticks = []
    for l in labels:
        new_xticks.append(str(l)[12:-15])
        # print(str(l)[12:-15])

    # new_xticks = [d[:11] for d in locs]
    plt.xticks(locs, new_xticks, rotation=45, horizontalalignment='right')

    sns.despine()
    # sns.boxplot(x='commit_hash', y='mem_percent', data=df, ax=axs[1])
    plt.tight_layout()
    # plt.show()
    plt.savefig('results/' + col['name'] + '.pdf')



def multiple_area(df, col):
    sns.set(font_scale=0.9)

    g = sns.FacetGrid(df, col='method_name_short', hue='method_name_short', col_wrap=4, )

    # Add the line over the area with the plot function
    g = g.map(plt.plot, 'commit_hash_short', col['name'])

    # Fill the area with fill_between
    g = g.map(plt.fill_between, 'commit_hash_short', col['name'], alpha=0.2).set_titles("{col_name} ")

    # Control the title of each facet
    g = g.set_titles("{col_name}")

    # plt.xticks(rotation=30)
    # Add a title for the whole plot
    # plt.subplots_adjust(top=0.75)
    # g = g.fig.suptitle(col['name'])

    # Show the graph
    # plt.show()
    plt.savefig('results/area/area_' + col['name'] + '.pdf')


def testcase_waterfall(df, commit_hash, class_name, method_name):
    method = df[(df['commit_hash'] == commit_hash) & (df['class_name'] == class_name) & (
            df['method_name'] == method_name)]
    run_id = method.run.iloc[0]
    methods_df = df.loc[df['run'] == run_id].sort_values('method_started_at')

    # convert string to datetime
    methods_df['method_started_at'] = pd.to_datetime(methods_df['method_started_at'],
                                                     format='%Y-%m-%d %H:%M:%S.%f')  # 2022-09-08 17:25:02.890
    methods_df['method_ended_at'] = pd.to_datetime(methods_df['method_ended_at'],
                                                   format='%Y-%m-%d %H:%M:%S.%f')  # 2022-09-08 17:25:02.890
    # waterfall_chart(methods_df)
    root = methods_df[methods_df['caller_id'].isnull()]
    methods = pd.DataFrame()
    methods = tree_node(methods_df, methods, root)
    # print(methods[['id',  'own_duration', 'cumulative_duration', 'completion']]) #'method_started_at', 'method_ended_at',
    waterfall_chart(methods)


def waterfall_chart(data):
    font = {'size': 20}

    matplotlib.rc('font', **font)
    ##### DATA PREP #####

    df = pd.DataFrame(data)

    proj_start = df.method_started_at.min()

    df['start_num'] = (df.method_started_at - proj_start).dt.total_seconds()#.dt.microseconds
    df['end_num'] = (df.method_ended_at - proj_start).dt.total_seconds()#.dt.microseconds
    df['time_start_to_end'] = df.end_num - df.start_num
    df['total_time'] = df.start_num + df.end_num
    # df = df[df["time_start_to_end"] > 0].head(50)

    # method id
    counter = 1
    for i, row in df.iterrows():
        df.at[i, 'method_label'] = str(counter) + '-' + row['method_name_short']
        counter += 1

    # df['current_num'] = (df.time_start_to_end * df.completion)

    # print(df[['method_label', 'method_started_at', 'method_ended_at', 'start_num', 'end_num', 'time_start_to_end']])
    if df['class_name_short'].any():
        class_name_short = df['class_name_short'].iloc[0]
        commit_hash_short = df['commit_hash_short'].iloc[0]
        method_name_short = df['method_name_short'].iloc[0]

        # create a column with the color for each department
        def color(row):
            list_colors = list(matplotlib.colors.cnames.values())[9:]
            colors = dict(zip(df['class_name_short'].unique(),
                              (f'{c}' for c in list_colors)))
            return colors[row['class_name_short']]

        df['color'] = df.apply(color, axis=1)

        ##### PLOT #####
        plt.rcParams.update({'figure.autolayout': True})
        # plt.style.use('tableau-colorblind10')
        fig, (ax, ax1) = plt.subplots(2, figsize=(28, 15), gridspec_kw={'height_ratios': [len(df.index), 1]})

        # bars
        hbars1 = ax.barh(df.method_label, df.time_start_to_end, left=df.start_num, color=df.color)  # , color=df.color
        # hbars2 = ax.barh(df.method_label, df.time_start_to_end, left=df.start_num, alpha=0.5, color=df.color)  # color=df.color,

        # grid lines
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.2, which='both')
        ax.set_xlabel('cumulative duration (seconds)')

        plt.gca().invert_yaxis()

        # # ticks
        xticks = np.arange(0, df.end_num.max() + 1, 1)
        ax.set_xticks(xticks)

        plt.suptitle(
            'Cumulative duration of ' + method_name_short + ' in ' + commit_hash_short)

        ##### LEGENDS #####
        legend_elements = []
        classes = df.drop_duplicates('class_name_short')

        for i, cl in classes.iterrows():
            patch = Patch(facecolor=str(cl['color']), label=str(cl['class_name_short']))
            legend_elements.append(patch)

        ax1.legend(handles=legend_elements, loc='upper center', ncol=3, frameon=False)

        # clean second axis
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # # Label with specially formatted floats
        ax.bar_label(hbars1, labels=df.time_start_to_end.round(2).tolist(), fmt='%.2f')
        # ax.bar_label(hbars2, fmt='%.f')

        plt.gcf().subplots_adjust(left=0.3)

        Path("results/" + commit_hash_short).mkdir(parents=True, exist_ok=True)
        plt.savefig(
            "results/" + commit_hash_short + '/' + commit_hash_short + '_' + class_name_short.replace('/',
                                                                            '_') + '_' + method_name_short + ".pdf")
    else:
        print('Attention: commit hash, class and method not found!')