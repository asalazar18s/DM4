from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

col_names = list(range(1, 501))
filename = "CorrSubset.csv"
df = pd.read_csv(filename, sep=',', index_col=0, header=0)

filename1 = "madelon_train.names"
training = pd.read_csv(filename1, sep='\n', index_col=None, header=None)


print(df)
print(training)
array = df.values
print(array)
X = array
print(X)
Y = training.values
print(Y)



model = LogisticRegression(solver='lbfgs', max_iter=4000)
rfe = RFE(model, 30)

# where x is the data and y is the target given in the names
fit = rfe.fit(X, Y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))
