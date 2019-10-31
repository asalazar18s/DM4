from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

col_names = list(range(1, 501))
filename = "CorrSubset.csv"
df = pd.read_csv(filename, sep=',', index_col=0, header=0)

filename1 = "madelon_train.names"
training = pd.read_csv(filename1, sep=' ', index_col=None, header=None)


print(df)
print(training)
array = df.values
print(array)
X = array
print(X)
Y = training.values
print(Y)



# model = LogisticRegression(solver='lbfgs', max_iter=4000)
# rfe = RFE(model, 30)
#
# # where x is the data and y is the target given in the names
# fit = rfe.fit(X, Y)
# print("Num Features: %s" % (fit.n_features_))
# print("Selected Features: %s" % (fit.support_))
# print("Feature Ranking: %s" % (fit.ranking_))
#
# print(fit.support_.tolist())


# att_list = [17,20,23,24,29,32,36,40,46,47,48,49,51,59,63,65,73,77,87,92,93,95,101,106,107,108,121,122,123,124,126,131,
#             139,142,143,146,147,148,152,154,155,156,159,162,164,178,180,183,184,188,189,191,193,197,201,204,208,215,217,
#             220,229,230,231,240,245,247,248,250,251,253,258,262,270,281,284,286,287,296,299,300,302,303,305,306,307,310,
#             311,318,326,327,328,335,336,338,341,342,345,354,360,361,363,366,372,373,375,394,400,403,405,410,411,417,418,
#             423,431,438,439,445,448,450,454,462,465,467,469,471,477,478,479,485,487,489,494,499]
#
# tf_list = [False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, True, False, False, False, True, False, False, True, True, True, False, False, False, False, False, False, False, True, False, False, True, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, True, True, False, False, False, True, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False]
#
#
# result_list = []
# for i in range(len(att_list)):
#     if tf_list[i]:
#         result_list.append(att_list[i])
#
# print(result_list)
cols = list(df)

to_keep_list = [20, 49, 121, 139, 147, 155, 162, 164, 178, 197, 208, 229, 281, 307, 318, 400, 405, 410, 423, 467]
for i in cols:
    if int(i) not in to_keep_list:
        df = df.drop(i, 1)

#print(df)

filename2 = "madelon_valid.data"
valid_data = pd.read_csv(filename2, sep=' ', index_col=None, header=None)
valid_data = valid_data.drop(500, 1)
valid_data_dataframe = pd.DataFrame(valid_data, columns=valid_data.keys())
print(valid_data)

filename3 = "madelon_valid.labels"
valid_labels = pd.read_csv(filename3, sep=' ', index_col=None, header=None)
valid_labels_dataframe = pd.DataFrame(valid_labels, columns=valid_labels.keys())
print("valid labels")
print(valid_labels)
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(df, training)
x = knn.predict(valid_data)
print(x.tolist())
print(len(x.tolist()))
# print(knn.score(x,valid_labels))