import pandas as pd

# Read the train data and set it to a pd DataFrame
col_names = list(range(1,501))
filename = "madelon_train.data"
df = pd.read_csv(filename, sep=' ', names=col_names, index_col=False, header=None)
#print(df)

def get_min_value_of_df(data):
    global df
    minVals = data.min(axis = 1)
    minVals = minVals.min()
    print("Min:")
    print(minVals)
    return minVals

def get_max_norm_of_df(data):
    global df
    maxVals = data.max(axis=1)
    maxVals = maxVals.max()
    print("Max:")
    print(maxVals)
    return maxVals

def normalize(dataframe):
    #dataframe = dataframe[(dataframe < 650).all(axis=1)]
    result = dataframe
    # print("Data Frame:")
    # print(result)
    max_value = get_max_norm_of_df(result)
    min_value = get_min_value_of_df(result)

    for feature_name in df.columns:
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    result.to_csv(r'./results.csv')
    return result


def covarianza():
    global df
    largest = []

    x = normalize(df)
    z = x.corr(method='pearson')
    z.to_csv(r'./correlated.csv')
    # print("Data Frame Normalized:")
    # print(x)
    # print("Data Frame Correlated:")
    # print(z)

    tuple_list = []
    print("Largest Values: ")
    for i in range(1, 500):
        for j in range(1, 500):
            if 0.1 < abs(z.at[i, j]) < 0.6 and z.at[i,j] != 1:
                if z.at[i, j] not in largest:
                    tuple_list.append((i,j, z.at[i,j]))
                    largest.append(z.at[i,j])

    # we have a list of tuples with the format (Column, row, value)
    # the column and row can be interchangeable
    print(len(largest))
    print(len(tuple_list))
    print(tuple_list)
    #res_dictionary = {}
   # for col, row, val in tuple_list:
       # if col in res_dictionary.keys():
       #     res_dictionary[col].append(row)
       # else:
        #    res_dictionary[col] = []
    # remove_cols([319, 443, 452, 473, 379, 337, 456, 476, 129, 339, 282, 434, 242, 494])


def remove_cols(int_array):
    global df

    for val in int_array:
        df = df.drop(val, 1)
    # df.to_csv(r'./df_after_first_corr.csv')
    print(df)

def get_final_df():
    global df

    to_keep_list = [17, 20, 23, 24, 32, 36, 40, 46, 47, 48, 51, 59, 63, 73, 77, 87, 92, 93, 95, 101, 106, 107, 108, 121,
                    122, 123, 124, 126, 131, 139, 142, 143, 146, 147, 148, 152, 154, 155, 156, 159, 162, 164, 178, 180,
                    183, 184, 188, 189, 191, 193, 197, 201, 204, 208, 215, 217, 220, 229, 230, 231, 240, 245, 247, 248,
                    250, 251, 253, 258, 262, 270, 281, 284, 286, 287, 296, 299, 300, 302, 303, 305, 306, 307, 310, 311,
                    318, 326, 327, 328, 335, 336, 338, 341, 342, 345, 354, 360, 361, 363, 366, 372, 373, 375, 394, 400,
                    403, 405, 410, 411, 417, 418, 423, 431, 438, 439, 445, 448, 450, 462, 465, 467, 469, 471, 477, 478,
                    479, 485, 487, 489, 494, 499, 29, 49, 65, 106, 154, 454]

    for i in range(1, 501):
        if i not in to_keep_list:
            df = df.drop(i, 1)

    df.to_csv(r'./CorrSubset.csv')
    print(df)


get_final_df()
