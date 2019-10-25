import pandas as pd

# Read the train data and set it to a pd DataFrame
col_names = list(range(1,501))
filename = "madelon_train.data"
df = pd.read_csv(filename, sep=' ', names=col_names, index_col=False, header=None)
print(df)

def get_min_value_of_df():
    global df
    maxVals = df.max(axis = 1)
    return maxVals
    print(maxVals.max())

def get_max_norm_of_df():
    global df
    minVals = df.min(axis = 1)
    return minVals
    print(minVals.min())

def normalize(dataframe):
    normalized_df = (dataframe - get_min_value_of_df()) / (get_max_norm_of_df() - get_min_value_of_df())
    return(normalized_df)

def covarianza():
    global df

    x = normalize(df)
    z = x.corr()
    print(x)
    #print(z.nlargest(10,1)[1])

#covarianza()
