from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd
import numpy as py

# initial 2000x500 dataset
col_names = list(range(0, 500))
filename = "madelon_train.data"
df = pd.read_csv(filename, sep=' ', names=col_names, index_col=False, header=None)

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
    global df
    #dataframe = dataframe[(dataframe < 650).all(axis=1)]
    result = dataframe
    # print("Data Frame:")
    # print(result)
    max_value = get_max_norm_of_df(result)
    min_value = get_min_value_of_df(result)

    for feature_name in df.columns:
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    result.to_csv(r'./results.csv')
    df = result
    return result



best_attributes = [28, 48, 64, 105, 128, 153, 241, 281, 318, 336, 338, 378,
                    433, 442, 451, 453, 455, 472, 475]

    #[79, 81, 105, 128, 148, 153, 173, 210, 227, 241, 243, 281, 305, 318, 338, 378, 435, 451, 459, 493]

#[64, 197, 241, 281, 318, 338, 378, 455]
    # [28, 48, 100, 107, 128, 153, 241, 281, 283, 318, 336, 338, 378, 442, 453, 455, 64]
    # (39, 48, 128, 153, 154, 216, 219, 241, 252, 281, 309, 318, 336, 338, 372, 378, 442, 451, 453, 455)
    # (48, 64, 105, 128, 153, 241, 281, 318, 338, 472)
    # 197 241 281 318 338 378 455
    # [28, 48, 100, 107, 128, 142, 153, 241, 280, 281, 283, 317, 318, 336, 338, 378, 442, 451, 453, 455]  0.9 with 9 neighbors
    # [28, 35, 39, 48, 64, 76, 105, 128, 163, 179, 207, 241, 280, 281, 305, 318, 336, 338, 378, 404, 433, 442, 453, 455, 475]
    #39, 163, 404

for col in df.columns:
    if col not in best_attributes:
        df = df.drop(col, axis=1)


# get the target values for the training dataset
filename1 = "madelon_train.names"
training_target_values = pd.read_csv(filename1, sep=' ', index_col=None, header=None)


neighbors = 9
# train the knn algorithm
knn = KNeighborsClassifier(n_neighbors=neighbors, metric='minkowski')
knn.fit(df, training_target_values)

# Open validation data
validation_file_name = "madelon_valid.data"
validation_data = pd.read_csv(validation_file_name, sep=' ', names=col_names, index_col=False, header=None)
# keep only desired columns/ attributes
for col in validation_data.columns:
    if col not in best_attributes:
        validation_data = validation_data.drop(col, axis=1)

# validation target values
filename2 = "madelon_valid.labels"
validation_target_values = pd.read_csv(filename2, sep=' ', index_col=None, header=None)

print(best_attributes)
print("Neighbors: ", neighbors)


print("Accuracy: ", knn.score(X=validation_data, y=validation_target_values))


