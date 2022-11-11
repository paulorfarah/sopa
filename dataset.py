import pandas as pd


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


def method_avg(df, class_name, method_name, col):
    df2 = df.query("method_name == '" + method_name + "' and class_name == '" + class_name + "'")

    # df_methods = df2.groupby(['commit_hash'])
    # df_methods = df_methods.aggregate('mean')
    df_res = pd.pivot_table(df2, values=col['name'],
                            index=['class_name'],
                            columns='commit_hash')
    print(df_res.head())
    return df_res
