import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

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
    sns.boxplot(data=df, color='plum',  width=0.5, orient='v')
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
    plt.show()




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

def gantt():
    df = [dict(Task="Job-1", Start='2017-01-01', Finish='2017-02-02', Resource='Complete'),
          dict(Task="Job-1", Start='2017-02-15', Finish='2017-03-15', Resource='Incomplete'),
          dict(Task="Job-2", Start='2017-01-17', Finish='2017-02-17', Resource='Not Started'),
          dict(Task="Job-2", Start='2017-01-17', Finish='2017-02-17', Resource='Complete'),
          dict(Task="Job-3", Start='2017-03-10', Finish='2017-03-20', Resource='Not Started'),
          dict(Task="Job-3", Start='2017-04-01', Finish='2017-04-20', Resource='Not Started'),
          dict(Task="Job-3", Start='2017-05-18', Finish='2017-06-18', Resource='Not Started'),
          dict(Task="Job-4", Start='2017-01-14', Finish='2017-03-14', Resource='Complete')]

    colors = {'Not Started': 'rgb(220, 0, 0)',
              'Incomplete': (1, 0.9, 0.16),
              'Complete': 'rgb(0, 255, 100)'}

    fig = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True,
                          group_tasks=True)
    fig.show()
