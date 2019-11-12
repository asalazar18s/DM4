from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, RadiusNeighborsClassifier
from sklearn import metrics
import pandas as pd
import numpy as py



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



def check_score():
    # initial 2000x500 dataset
    col_names = list(range(0, 500))
    filename1 = "madelon_train.names"
    training_target_values = pd.read_csv(filename1, sep=' ', index_col=None, header=None)
    neighbors = 5
    m = 'minkowski'
    knn = KNeighborsClassifier(n_neighbors=neighbors,
                               weights='uniform',
                               metric=m,
                               )
    filename2 = "madelon_valid.labels"
    validation_target_values = pd.read_csv(filename2, sep=' ', index_col=None, header=None)
    best_attributes = [28, 90, 153, 288, 292, 318, 378, 455, 197, 475, 142, 100, 451, 472, 128, 7]
                       #151, 468, 482, 3, 0]
    for i in range(0, 500):
        validation_file_name = "madelon_valid.data"
        validation_data = pd.read_csv(validation_file_name, sep=' ', names=col_names, index_col=False, header=None)
        filename = "madelon_train.data"
        df = pd.read_csv(filename, sep=' ', names=col_names, index_col=False, header=None)
        if i not in best_attributes:
            best_attributes.append(i)
            for col in df.columns:
                if col not in best_attributes:
                    df = df.drop(col, axis=1)
            # train the knn algorithm
            knn.fit(df, training_target_values)
            for col in validation_data.columns:
                if col not in best_attributes:
                    validation_data = validation_data.drop(col, axis=1)
            # validation target values
            x = knn.score(X=validation_data, y=validation_target_values)
            if x > 0.93:
                print("Index: " + str(i), "Score: " + str(x))
                return(i)
            else:
                print("Best attributes: ", best_attributes)
                best_attributes.remove(i)
        print("Index: " + str(i), "Score: " + str(x))

print(check_score())
# print(best_attributes)
# print("Neighbors: ", neighbors)
# print("Metric: ", m)
#
#
# print("Accuracy: ", knn.score(X=validation_data, y=validation_target_values))

# 39 120 133 148 40 53 112 133
# 154 167 173 207 228 237 247 248 302 332 376 392 394 404 407 420 445
# 473
# 491 496
#