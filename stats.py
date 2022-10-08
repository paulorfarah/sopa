import numpy as np

def get_statistics(df, column_name):
    df_copy = df.copy()
    print(f"Mean: ", np.round(df_copy[column_name].mean(), 2))
    print(f"Standard Deviation: ", np.round(df_copy[column_name].std()))
    print(f"Median: ", np.round(df_copy[column_name].median(), 2))
    print(f"Max: ", df_copy[column_name].max())
    print(f"Min: ", df_copy[column_name].min())

def get_group_statistics(df, categorical, numerical):
    print('---- Mean ----')
    group_df = df[[categorical, numerical]].groupby(categorical).mean().dropna()
    print(group_df.head())
    print('---- Standard Deviation ----')
    group_df = df[[categorical, numerical]].groupby(categorical).std().dropna()
    print(group_df.head())
    print('---- Median ----')
    group_df = df[[categorical, numerical]].groupby(categorical).median().dropna()
    print(group_df.head())
    print('---- Max ----')
    group_df = df[[categorical, numerical]].groupby(categorical).max().dropna()
    print(group_df.head())
    print('---- Min ----')
    group_df = df[[categorical, numerical]].groupby(categorical).min().dropna()
    print(group_df.head())
