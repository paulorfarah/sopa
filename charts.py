import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import datetime as dt


def ridgeline(df):
    # getting the data
    temp = pd.read_csv(
        'https://raw.githubusercontent.com/plotly/datasets/master/2016-weather-data-seattle.csv')  # we retrieve the data from plotly's GitHub repository
    temp['month'] = pd.to_datetime(temp['Date']).dt.month  # we store the month in a separate column

    # we define a dictionnary with months that we'll use later
    month_dict = {1: 'january',
                  2: 'february',
                  3: 'march',
                  4: 'april',
                  5: 'may',
                  6: 'june',
                  7: 'july',
                  8: 'august',
                  9: 'september',
                  10: 'october',
                  11: 'november',
                  12: 'december'}

    # we create a 'month' column
    temp['month'] = temp['month'].map(month_dict)

    # we generate a pd.Serie with the mean temperature for each month (used later for colors in the FacetGrid plot), and we create a new column in temp dataframe
    month_mean_serie = temp.groupby('month')['Mean_TemperatureC'].mean()
    temp['mean_month'] = temp['month'].map(month_mean_serie)

    # we generate a color palette with Seaborn.color_palette()
    pal = sns.color_palette(palette='coolwarm', n_colors=12)

    # in the sns.FacetGrid class, the 'hue' argument is the one that is the one that will be represented by colors with 'palette'
    g = sns.FacetGrid(temp, row='month', hue='mean_month', aspect=15, height=0.75, palette=pal)

    # then we add the densities kdeplots for each month
    g.map(sns.kdeplot, 'Mean_TemperatureC',
          bw_adjust=1, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)

    # here we add a white line that represents the contour of each kdeplot
    g.map(sns.kdeplot, 'Mean_TemperatureC',
          bw_adjust=1, clip_on=False,
          color="w", lw=2)

    # here we add a horizontal line for each plot
    g.map(plt.axhline, y=0,
          lw=2, clip_on=False)

    # we loop over the FacetGrid figure axes (g.axes.flat) and add the month as text with the right color
    # notice how ax.lines[-1].get_color() enables you to access the last line's color in each matplotlib.Axes
    for i, ax in enumerate(g.axes.flat):
        ax.text(-15, 0.02, month_dict[i + 1],
                fontweight='bold', fontsize=15,
                color=ax.lines[-1].get_color())

    # we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
    g.fig.subplots_adjust(hspace=-0.3)

    # eventually we remove axes titles, yticks and spines
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    plt.setp(ax.get_xticklabels(), fontsize=15, fontweight='bold')
    plt.xlabel('Temperature in degree Celsius', fontweight='bold', fontsize=15)
    g.fig.suptitle('Daily average temperature in Seattle per month',
                   ha='right',
                   fontsize=20,
                   fontweight=20)

    plt.show()


def boxplot(df, col):
    sns.set(style='whitegrid')

    # sns.boxplot(x='commit_hash', y='cpu_percent', data=df, palette="deep")

    fig, axs = plt.subplots(figsize=(10, 6))
    # sns.boxplot(x='commit_hash', y='duration', data=df, ax=axs[0], color='plum', width=0.5)
    sns.boxplot(data=df, color='plum', width=0.5, orient='v')
    plt.title(col, fontsize=14)
    plt.xlabel('date')
    sns.despine()
    # sns.boxplot(x='commit_hash', y='mem_percent', data=df, ax=axs[1])
    plt.tight_layout()
    plt.show()


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


def correlation(df):
    # X, y = datasets.make_regression(n_samples=200, n_features=5, n_informative=2, random_state=42)
    # df = pd.DataFrame(X)
    # df.columns = ['ftre1', 'ftre2', 'ftre3', 'ftre4', 'ftre5']
    # df['target'] = y
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(9, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, mask=mask, cmap=cmap)
    plt.show()


def pdf_cdf(df, col):
    # https: // stackoverflow.com / questions / 25577352 / plotting - cdf - of - a - pandas - series - in -python
    # print(df.columns)
    s = pd.Series(df[col], name='value')
    df = pd.DataFrame(s)
    # print(df.head)

    # Get the frequency, PDF and CDF for each value in the series

    # Frequency
    stats_df = df.groupby('value')['value'].agg('count').pipe(pd.DataFrame).rename(columns={'value': 'frequency'})

    # PDF
    stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])

    # CDF
    stats_df['cdf'] = stats_df['pdf'].cumsum()
    stats_df = stats_df.reset_index()
    print(stats_df)
    # Plot the discrete Probability Mass Function and CDF.
    # Technically, the 'pdf label in the legend and the table the should be 'pmf'
    # (Probability Mass Function) since the distribution is discrete.

    # If you don't have too many values / usually discrete case
    stats_df.plot.bar(x='value', y=['pdf', 'cdf'], grid=True)

    # Define your series
    s = pd.Series(np.random.normal(loc=10, scale=0.1, size=1000), name='value')

    # Plot
    stats_df.plot(x='value', y=['pdf', 'cdf'], grid=True)

    stats_df.plot(x='value', y=['pdf'], grid=True)

    stats_df.plot(x='value', y=['cdf'], grid=True)
    plt.show()


# def gantt():
#     df = [dict(Task="Job-1", Start='2017-01-01', Finish='2017-02-02', Resource='Complete'),
#           dict(Task="Job-1", Start='2017-02-15', Finish='2017-03-15', Resource='Incomplete'),
#           dict(Task="Job-2", Start='2017-01-17', Finish='2017-02-17', Resource='Not Started'),
#           dict(Task="Job-2", Start='2017-01-17', Finish='2017-02-17', Resource='Complete'),
#           dict(Task="Job-3", Start='2017-03-10', Finish='2017-03-20', Resource='Not Started'),
#           dict(Task="Job-3", Start='2017-04-01', Finish='2017-04-20', Resource='Not Started'),
#           dict(Task="Job-3", Start='2017-05-18', Finish='2017-06-18', Resource='Not Started'),
#           dict(Task="Job-4", Start='2017-01-14', Finish='2017-03-14', Resource='Complete')]
#
#     colors = {'Not Started': 'rgb(220, 0, 0)',
#               'Incomplete': (1, 0.9, 0.16),
#               'Complete': 'rgb(0, 255, 100)'}
#
#     fig = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True,
#                           group_tasks=True)
#     fig.show()


def surface(df, col):
    # Get the data (csv file is hosted on the web)
    # url = 'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/volcano.csv'
    # data = pd.read_csv(url)
    #
    # # Transform it to a long format
    # df = data.unstack().reset_index()
    # df.columns = ["X", "Y", "Z"]
    # print(df.columns)
    # print(df.head())

    # And transform the old column name in something numeric
    df['X'] = pd.Categorical(df['X'])
    df['X'] = df['X'].cat.codes

    # # Make the plot
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
    # plt.show()
    #
    # # to Add a color bar which maps values to colors.
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()
    #
    # # Rotate it
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
    # ax.view_init(30, 45)
    # plt.show()

    # Other palette
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.jet, linewidth=0.01)
    plt.show()


def lines(df, col):
    print(df.head())
    sns.set_theme(style="whitegrid")

    # rs = np.random.RandomState(365)
    # values = rs.randn(365, 4).cumsum(axis=0)
    # dates = pd.date_range("1 1 2016", periods=365, freq="D")
    # data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])
    # data = data.rolling(7).mean()

    # sns.lineplot(data=data, palette="tab10", linewidth=2.5)

    # pivot the data into the correct shape
    df = df.pivot_table(index='commit_hash_short', columns='class_name', values=col['name'], aggfunc=col['measure'])
    # plot the pivoted dataframe; if the column names aren't colors, remove color=df.columns
    df.plot(figsize=(15, 10), title=col['name']).legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    # dataframe_var.plot.bar().legend(loc='center left', bbox_to_anchor=(1.0, 0.5));


def multiple_area(df, col):
    # Create a dataset
    # my_count = ["France", "Australia", "Japan", "USA", "Germany", "Congo", "China", "England", "Spain", "Greece",
    #             "Marocco", "South Africa", "Indonesia", "Peru", "Chili", "Brazil"]
    # df = pd.DataFrame({
    #     "country": np.repeat(my_count, 10),
    #     "years": list(range(2000, 2010)) * 16,
    #     "value": np.random.rand(160)
    # })

    # Create a grid : initialize it
    g = sns.FacetGrid(df, col='class_name_short', hue='class_name_short', col_wrap=10, )

    # Add the line over the area with the plot function
    g = g.map(plt.plot, 'commit_hash_short', col['name'])

    # Fill the area with fill_between
    g = g.map(plt.fill_between, 'commit_hash_short', col['name'], alpha=0.2).set_titles("{col_name} class_name_short")

    # Control the title of each facet
    g = g.set_titles("{col_name}")

    # Add a title for the whole plot
    plt.subplots_adjust(top=0.92)
    g = g.fig.suptitle('Evolution of the ' + col['name'])

    # Show the graph
    plt.show()


def tree_node(df, methods_df, root):
    # print(root.id)
    # root['completion'] = root['own_duration'] * 100.0 / root['cumulative_duration']
    methods_df = pd.concat([methods_df, root])
    # print(df.head())
    # print(root['run'])
    # print(df['run'])
    # find call tree of methods
    leafs = df.loc[(df.caller_id == float(root.id)) & (df.run == int(root.run))]
    for leaf_id in leafs['id']:
        # print('tree: ', leaf_id)
        leaf = df.loc[df['id'] == float(leaf_id)]
        methods_df = tree_node(df, methods_df, leaf)
        # print(m['id'])
        # completion = method_df['own_duration']*100/method_df['cumulative_duration']
        # methods.append({'commit_hash': method_df['commit_hash'], 'class_name': method_df['class_name'],
        #                 'method_name': method_df['method_name'], 'own_duration': method_df['own_duration'],
        #                 'cumulative_duration': method_df['cumulative_duration'], 'method_started_at':
        #                     method_df['method_started_at'], 'method_ended_at': method_df['method_ended_at'],
        #                 'completion': completion})
    return methods_df




# def waterfall_method(df, commit, class_name, method_name):
#     # https: // towardsdatascience.com / gantt - charts -
#     # with-pythons - matplotlib - 395b7af72d72
#
#     methods = pd.DataFrame()
#     root = df[(df['commit_hash'] == commit) & (df['class_name'] == class_name) & (
#                 df['method_name'] == method_name)]  # .iloc[0]
#
#     if len(root.index) > 0:
#         # find root testcase that calls all submethods
#         while root['caller_id'].any():
#             root = df.loc[(df.id == int(root['caller_id'].iloc[0])) & (df.run == int(root['run'].iloc[0]))]
#         methods = tree_node(df, methods, root)
#
#         # convert string to datetime
#         methods['method_started_at'] = pd.to_datetime(methods['method_started_at'],
#                                                       format='%Y-%m-%d %H:%M:%S.%f')  # 2022-09-08 17:25:02.890
#         methods['method_ended_at'] = pd.to_datetime(methods['method_ended_at'],
#                                                     format='%Y-%m-%d %H:%M:%S.%f')  # 2022-09-08 17:25:02.890
#
#         # print(methods.head())
#         # project start date
#
#         # print(methods.head())
#         methods.to_csv('results/methods.csv', index=False)
#         # waterfall_chart(methods)
#
#         # fig, ax = plt.subplots(1, figsize=(16, 10))
#         # ax.barh(methods.method_name, methods.time_start_to_end, left=methods.start_num)
#         # plt.show()
#     else:
#         print('Attention: method ' + method_name + ' not found!!!')


# def waterfall_method2(df):
#     calls = pd.DataFrame(columns=['commit_hash', 'class_name', 'method_name_source', 'class_name_dest',
#                                   'method_name_dest', 'own_duration', 'cumulative_duration'])
#     # print(df.head())
#     for index, row in df.iterrows():
#         commit_hash = row['commit_hash']
#         class_name_dest = row['class_name']
#         method_name_dest = row['method_name']
#         caller_id = row['caller_id']
#         if caller_id:
#             caller = df.loc[(df['commit_hash'] == commit_hash) & (df['id'] == caller_id)]
#             if len(caller):
#                 class_name_source = caller['class_name'].iloc[0]
#                 method_name_source = caller['method_name'].iloc[0]
#             calls.loc[len(calls.index)] = [commit_hash, class_name_source, method_name_source, class_name_dest,
#                                            method_name_dest, row['own_duration'], row['cumulative_duration']]


def waterfall_method(df, commit_hash, class_name, method_name):
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

        # x_ticks = [i for i in range(int(proj_end.strftime('%S')) + 1)]
        # print(df.start_num)
        # print(dt.timedelta(seconds=i))
        # x_labels = [(df.start_num + dt.timedelta(seconds=i)).strftime('%S.%f')
        #             for i in x_ticks]

        fig, (ax, ax1) = plt.subplots(2, figsize=(28, 15), gridspec_kw={'height_ratios': [len(df.index), 1]})

        # bars
        hbars1 = ax.barh(df.method_label, df.time_start_to_end, left=df.start_num, color=df.color)  # , color=df.color
        # hbars2 = ax.barh(df.method_label, df.time_start_to_end, left=df.start_num, alpha=0.5, color=df.color)  # color=df.color,

        # for idx, row in df.iterrows():
        #     ax.text(row.end_num + 0.1, idx, f"{int(row.completion * 100)}%", va='center', alpha=0.8)
            # ax.text(row.start_num - 0.1, idx, row.method_name, va='center', ha='right', alpha=0.8)

        # grid lines
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.2, which='both')
        ax.set_xlabel('cumulative duration (seconds)')

        plt.gca().invert_yaxis()

        # # ticks
        xticks = np.arange(0, df.end_num.max() + 1, 1)
        # xticks_labels = pd.date_range(proj_start, end=df.method_ended_at.max()).strftime("%S")
        # xticks_minor = np.arange(0, df.end_num.max() + 1, 1)
        ax.set_xticks(xticks)
        # ax.set_xticks(xticks_minor, minor=True)
        # ax.set_xticklabels(xticks_labels[::3])
        # ax.set_yticks([])
        #
        # # ticks top
        # # create a new axis with the same y
        # ax_top = ax.twiny()
        #
        # # align x axis
        # ax.set_xlim(0, df.end_num.max())
        # ax_top.set_xlim(0, df.end_num.max())
        #
        # # top ticks (markings)
        # xticks_top_minor = np.arange(0, df.end_num.max() + 1, 7)
        # ax_top.set_xticks(xticks_top_minor, minor=True)
        # # top ticks (label)
        # xticks_top_major = np.arange(3.5, df.end_num.max() + 1, 7)
        # ax_top.set_xticks(xticks_top_major, minor=False)
        #
        # # week labels
        # xticks_top_labels = [f"Week {i}" for i in np.arange(1, len(xticks_top_major) + 1, 1)]
        # ax_top.set_xticklabels(xticks_top_labels, ha='center', minor=False)
        #
        # # hide major tick (we only want the label)
        # ax_top.tick_params(which='major', color='w')
        # # increase minor ticks (to marks the weeks start and end)
        # ax_top.tick_params(which='minor', length=8, color='k')
        #
        # # remove spines
        # ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.spines['left'].set_position(('outward', 10))
        # ax.spines['top'].set_visible(False)
        #
        # ax_top.spines['right'].set_visible(False)
        # ax_top.spines['left'].set_visible(False)
        # ax_top.spines['top'].set_visible(False)

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
        # ax.set_xlim(right=15)  # adjust xlim to fit labels

        # rects = ax.patches
        #
        # # Make some labels.
        # labels = [f"label{i}" for i in range(len(rects))]
        #
        # for rect, label in zip(rects, labels):
        #     height = rect.get_height()
        #     ax.text(
        #         rect.get_x() + rect.get_width() / 2, height + 5, label, ha="center", va="bottom"
        #     )

        plt.gcf().subplots_adjust(left=0.3)
        # plt.tight_layout()
        # plt.show()
        plt.savefig(
            "results/" + commit_hash_short + '_' + class_name_short.replace('/',
                                                                            '_') + '_' + method_name_short + ".pdf")
    else:
        print('Attention: commit hash, class and method not found!')