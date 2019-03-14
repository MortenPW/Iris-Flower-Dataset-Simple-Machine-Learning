import pandas as pd
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Normalize
from sklearn.preprocessing import normalize

from sklearn import datasets, svm, tree, preprocessing, metrics
from sklearn.preprocessing import StandardScaler

# Iterators for search grid
import itertools

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion matrix
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(X_test, y_test, classes,
                          normalize=False,
                          title="Confusion matrix",
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = metrics.confusion_matrix(y_test, model.predict(X_test))
    
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    #print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()


# Score models
from sklearn import metrics

def score_report(X_test, y_test):
    print("Test accuracy: ", model.score(X_test, y_test))
    print("")
    print(metrics.classification_report(y_test, model.predict(X_test)))
    plot_confusion_matrix(X_test, y_test, classes="", normalize=True, title="")


# Load dataset
path = "datasets\\"
filename = "iris.data.txt"

df = pd.read_csv(path + filename, sep=",", decimal=",", header=None, names=None)

df.sample(10)

# Scatter plot
sns.pairplot(df, hue=4, size=2)

df_p = df.copy()

df_p.describe()

df_p.agg(["median", "nunique", "var", "prod"])

# Encode categorical data and convert rest to numberic
df_p = pd.get_dummies(df_p, columns=[4])
df_p = df_p.apply(pd.to_numeric)

#df_p[4] = preprocessing.LabelEncoder().fit_transform(df_p[4])

# Check correlations
corr = df_p.corr()
f, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

df = df.drop_duplicates()

y = df.loc[:, [4]]
X = df.drop([4], axis=1)

X = X.apply(pd.to_numeric)

# Convert to numpy
y = y.values
X = X.values

y = y.ravel()

X[:, 1]

rows, columns = X.shape
Z = np.zeros([rows, columns+6])

# Feature crosses
Z[:, 0] = X[:, 0]
Z[:, 1] = X[:, 1]
Z[:, 2] = X[:, 2]
Z[:, 3] = X[:, 3]

Z[:, 4] = X[:, 0] * X[:, 1]
Z[:, 5] = X[:, 0] * X[:, 2]
Z[:, 6] = X[:, 0] * X[:, 3]

Z[:, 7] = X[:, 1] * X[:, 2]
Z[:, 8] = X[:, 1] * X[:, 3]

Z[:, 9] = X[:, 2] * X[:, 3]
Z

# Normalize
#X = normalize(X, norm="l2", axis=1, copy=True, return_norm=False)
#X

# Manual normalization
#X[:, 0] = X[:, 0] / X[:, 0].max()
#X[:, 1] = X[:, 1] / X[:, 1].max()
#X[:, 2] = X[:, 2] / X[:, 2].max()
#X[:, 3] = X[:, 3] / X[:, 3].max()

# PCA
#from sklearn.decomposition import PCA

#pca = PCA(n_components = 8)
#principalComponents = pca.fit_transform(Z)
#Z = pd.DataFrame(data = principalComponents)

### Normalize after PCA
#Z = normalize(X, norm="l2", axis=1, copy=True, return_norm=False)

# Split data
# Random state set for experimentation with same sets!
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=1)

# Time function (prediction time)
import timeit, functools
t = timeit.Timer(functools.partial(model.predict, X_test))
print ("Avg. time:", t.timeit(1000), "seconds.")

# Modeller
from sklearn.naive_bayes import GaussianNB 

model = GaussianNB()
model = model.fit(X_train, y_train)

score_report(X_test, y_test)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5) 
model = model.fit(X_train, y_train)

score_report(X_test, y_test)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=3, random_state=1)
model = model.fit(X_train, y_train)

score_report(X_test, y_test)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=2, random_state=1)
model = model.fit(X_train, y_train)

score_report(X_test, y_test)

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=1, random_state=1)
model = model.fit(X_train, y_train)

score_report(X_test, y_test)

from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(n_estimators=5, random_state=1)
model = model.fit(X_train, y_train)

score_report(X_test, y_test)

from sklearn import svm

model = svm.LinearSVC(C=0.05)
model = model.fit(X_train, y_train)

score_report(X_test, y_test)

model = svm.SVC(kernel='linear', C=0.4)
model = model.fit(X_train, y_train)

score_report(X_test, y_test)

model = svm.SVC(kernel='rbf', gamma=1.0, C=0.13)
model = model.fit(X_train, y_train)

score_report(X_test, y_test)

model = svm.SVC(kernel='poly', degree=2, C=0.05)
model = model.fit(X_train, y_train)

score_report(X_test, y_test)

model = svm.NuSVC(probability=True)
model = model.fit(X_train, y_train)

score_report(X_test, y_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model = LinearDiscriminantAnalysis()
model = model.fit(X_train, y_train)

score_report(X_test, y_test)

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

model = QuadraticDiscriminantAnalysis()
model = model.fit(X_train, y_train)

score_report(X_test, y_test)

# ANN - MLP

hyperparams = [
        {
            "activation" : ["identity", "logistic", "tanh", "relu"],
            "solver" : ["lbfgs", "sgd", "adam"],
            "hidden_layer_sizes": [(128, 256, 128,), (1, 5, 1,), (10, 50, 20,), (10, 10, 10,), (512,), (256, 512),
                                     (128, 128, 128, 128, 128,), (10, 10, 10, 10, 10, 10, 10, 10,), (512, 512, 512,),
                                     (1024,), (1024, 1024, 1024,), (512, 128, 10,), (60, 30, 10,), (1512, 1128, 110, 10,),
                                     (100, 100, 100, 100, 10,), (256,), (512,), (1024,), (2048,), (4000,), (6000,), (8000,)]
        }
       ]

#[n for n in itertools.product([1, 2, 3, 4, 5, 6, 7, 8, 9], repeat=2)]

# Solver lbfgs anbefales for mindre datasett.
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
hyperparams = [
        {
            "activation" : ["identity", "logistic", "tanh", "relu"],
            "solver" : ["lbfgs", "sgd", "adam"],
            "hidden_layer_sizes": [n for n in itertools.product([1, 2, 3, 4, 5, 6, 7, 8, 9], repeat=1)]
        }
       ]

hyperparams = [
        {
            "activation" : ["relu"],
            "solver" : ["lbfgs"],
            "hidden_layer_sizes": [n for n in itertools.product([10, 20, 30, 40, 50, 60, 70, 80, 90], repeat=2)]
        }
       ]

hyperparams = [
        {
            "activation" : ["identity", "logistic", "tanh", "relu"],
            "solver" : ["lbfgs"],
            "hidden_layer_sizes": [(10, 70, 50,), (30, 60, 80,), (10, 90, 80,)]
        }
       ]

hyperparams = [
        {
            "activation" : ["relu"],
            "solver" : ["lbfgs"],
            "hidden_layer_sizes": [(6, 9,), (3, 9), (3, 3)]
        }
       ]

# Train model using grid search
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(verbose=10, learning_rate="adaptive", max_iter=2000, random_state=1)

# CV = Cross-validation. Verbose = how often should it report back elapsed time.
#model = GridSearchCV(mlp, hyperparams, verbose=10, n_jobs=-1, cv=5)
model = GridSearchCV(mlp, hyperparams, verbose=10, n_jobs=3, cv=30)
model.fit(X_train, y_train)

print("\nCompleted grid search with best mean cross-validated score: ", model.best_score_)
print("Best hyperparams appears to be: ", model.best_params_)

model = model.best_estimator_

print("Test accuracy: ", model.score(X_test, y_test))

score_report(X_test, y_test)

