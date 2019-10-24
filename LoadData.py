import pandas as pd

# Read the train data and set it to a pd DataFrame
col_names = list(range(1,501))
filename = "madelon_train.data"
df = pd.read_csv(filename, sep=' ', names=col_names, index_col=False, header=None)

# print(df)

def get_max_min_value_of_col(col_num):
    global df
    # use axis-1 as a parameter in max to get max value of row
    # DataFrame.idxmax(axis=0) use this to get id of highest row in a given column
    return df[col_num].max(),df[col_num].min()

def get_min_max_norm_of_value(x_value, col_num):
    max,min = get_max_min_value_of_col(col_num)
    norm = (x_value - min)/(max-min)
    return norm

def normalize(dataframe):
    normalized_df = (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())
    return(normalized_df)

def covarianza():
    global df

    x = normalize(df)
    y = x.cov()
    z = x.corr()

    return y.sum(axis=0, skipna=True), x.sum(axis=0, skipna=True)


# print(get_max_min_value_of_col(2))
# print(get_min_max_norm_of_value(485, 1))
print(covarianza())
