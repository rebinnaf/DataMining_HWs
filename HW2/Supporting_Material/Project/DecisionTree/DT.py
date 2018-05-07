import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pandas as pd
pd.options.display.max_columns = 100

from matplotlib import pyplot as plt
import numpy as np

import seaborn as sns

import pylab as plot

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import  cross_val_score
from time import time
from operator import itemgetter
from scipy.stats import randint

def dropEmptyColumn():

    for l in list(combined):
        x=0
        for data in combined[l]:
            if data=="?" :
                x+=1
        if x>=combined.shape[0]:
            print("delete column:",l)
            combined.drop(l, axis=1, inplace=True)
    combined.drop("product-type", axis=1, inplace=True)
    if "packing" in list(combined):
         combined.drop("packing", axis=1, inplace=True)




def balanceData():
    global train
    tty = train[(train["classes"] == '3')]
    print(tty.shape)
    i = 0
    while i < 500:
        drop_indices = np.random.choice(tty.index, 1, replace=False)
        tty = tty.drop(drop_indices);
        i += 1
    train = train[train["classes"] != '3']
    train = train.append(tty)
    print(train.shape)
    train.reset_index(inplace=True)


def MappingContinues():
    global combined

    combined.loc[combined['hardness'] <= 30, 'hardness'] = 0
    combined.loc[(combined['hardness'] > 30) & (combined['hardness'] <= 75), 'hardness'] = 1
    combined.loc[(combined['hardness'] > 75) & (combined['hardness'] <= 90), 'hardness'] = 2

    combined.loc[combined['strength'] <= 250, 'strength'] = 0
    combined.loc[(combined['strength'] > 250) & (combined['strength'] <= 305), 'strength'] = 1
    combined.loc[(combined['strength'] > 305) & (combined['strength'] <= 380), 'strength'] = 2
    combined.loc[(combined['strength'] > 380) & (combined['strength'] <= 450), 'strength'] = 3
    combined.loc[(combined['strength'] > 450) & (combined['strength'] <= 650), 'strength'] = 4
    combined.loc[(combined['strength'] > 650) & (combined['strength'] <= 900), 'strength'] = 5

    combined.loc[combined['4'] <= 3, '4'] = 0
    combined.loc[(combined['4'] > 3) & (combined['4'] <= 8), '4'] = 1
    combined.loc[(combined['4'] > 8) & (combined['4'] <= 80), '4'] = 2

    combined.loc[combined['thick'] <= 0.77, 'thick'] = 0
    combined.loc[(combined['thick'] > 0.77) & (combined['thick'] <= 2.1), 'thick'] = 1
    combined.loc[(combined['thick'] > 2.1) & (combined['thick'] <= 5), 'thick'] = 2

    combined.loc[combined['width'] <= 150, 'width'] = 0
    combined.loc[(combined['width'] > 150) & (combined['width'] <= 240), 'width'] = 1
    combined.loc[(combined['width'] > 240) & (combined['width'] <= 350), 'width'] = 2
    combined.loc[(combined['width'] > 350) & (combined['width'] <= 600), 'width'] = 3
    combined.loc[(combined['width'] > 600) & (combined['width'] <= 625), 'width'] = 4
    combined.loc[(combined['width'] > 625) & (combined['width'] <= 910), 'width'] = 5
    combined.loc[(combined['width'] > 910) & (combined['width'] <= 1235), 'width'] = 6
    combined.loc[(combined['width'] > 1235) & (combined['width'] <= 1600), 'width'] = 7

    combined.loc[combined['len'] <= 600, 'len'] = 0
    combined.loc[(combined['len'] > 600) & (combined['len'] <= 1000), 'len'] = 1
    combined.loc[(combined['len'] > 1000) & (combined['len'] <= 5000), 'len'] = 2


def MapCategorical():
    global combined

    combined = combined.replace('Y', 1)
    combined = combined.replace('T', 1)
    combined['classes'] = combined['classes'].map({'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 'U': 6})
    combined['formability'] = combined['formability'].map({'1': 1, '2': 2, '3': 3, '5': 5, '?': 0})
    combined['enamelability'] = combined['enamelability'].map({'1': 1, '2': 2, '?': 0})
    # combined['packing'] = combined['packing'].map({'2': 2, '3': 3, '?': 0})

    # combined.loc[(combined['classes'] == 'U'), 'classes'] = 6
    combined['steel'] = combined['steel'].map({'A': 1, 'K': 2, 'M': 3, 'R': 4, 'S': 5, 'V': 6, 'W': 7, '?': 0})
    combined['family'] = combined['family'].map({'TN': 1, 'ZS': 2, '?': 0})
    combined['condition'] = combined['condition'].map({'A': 1, 'S': 2, '?': 0})
    combined['non-ageing'] = combined['non-ageing'].map({'N': 1, '?': 0})
    combined['surface-finish'] = combined['surface-finish'].map({'P': 1, '?': 0})
    combined['surface-quality'] = combined['surface-quality'].map({'D': 1, 'E': 2, 'F': 3, 'G': 4})
    combined['bw/me'] = combined['bw/me'].map({'B': 1, 'M': 2, '?': 0})
    combined['chrom'] = combined['chrom'].map({'C': 1, '?': 0})
    combined['phos'] = combined['phos'].map({'P': 1, '?': 0})
    combined['blue/bright/varn/clean'] = combined['blue/bright/varn/clean'].map({'B': 1, 'c': 2, 'V': 3, '?': 0})
    combined['shape'] = combined['shape'].map({'COIL': 1, 'SHEET': 2})
    combined['oil'] = combined['oil'].map({1: 1, 'N': 2, '?': 0})
    combined['bore'] = combined['bore'].map({500: 1, 600: 2, 0: 0})

def fill_surfaceq(row):
    global combined

    grouped_combined = combined.groupby(['temper_rolling','strength'])
    grouped_median_combined = grouped_combined.median()
    grouped_median_combined = grouped_median_combined.reset_index()[['temper_rolling','strength','surface-quality']]
    condition = (
        (grouped_median_combined['temper_rolling'] == row['temper_rolling']) &
        (grouped_median_combined['strength'] == row['strength'])
    )
    # print(grouped_median_combined[condition]['surface-quality'].values)
    if not np.isnan(grouped_median_combined[condition]['surface-quality'].values):
        return grouped_median_combined[condition]['surface-quality'].mode().iloc[0]
    elif row['strength'] > 0:
        return 4

    # a function that fills the missing values of the Age variable

def FillMissingValues():
    global combined

    combined = combined.replace('?', 0)
    combined['surface-quality'] = combined['surface-quality'].fillna(combined['surface-quality'].mode())
    combined['surface-quality'] = combined.apply(
        lambda row: fill_surfaceq(row) if np.isnan(row['surface-quality']) else row['surface-quality'], axis=1)
    combined.fillna(0)


def report(grid_scores, n_top=3):

    """Report top n_top parameters settings, default n_top=3.

    Args
    ----
    grid_scores -- output from grid or random search
    n_top -- how many to report, of top models

    Returns
    -------
    top_params -- [dict] top parameter settings found in
                  search
    """
    top_scores = sorted(grid_scores,
                        key=itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print(("Mean validation score: "
               "{0:.3f} (std: {1:.3f})").format(
               score.mean_validation_score,
               np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    return top_scores[0].parameters

def run_gridsearch(X, y, clf, param_grid, cv=5):
    """Run a grid search for best Decision Tree parameters.

    Args
    ----
    X -- features
    y -- targets (classes)
    cf -- scikit-learn Decision Tree
    param_grid -- [dict] parameter settings to test
    cv -- fold of cross-validation, default 5

    Returns
    -------
    top_params -- [dict] from report()
    """
    grid_search = GridSearchCV(clf,
                               param_grid=param_grid,
                               cv=cv)
    start = time()
    grid_search.fit(X, y)

    print(("\nGridSearchCV took {:.2f} "
           "seconds for {:d} candidate "
           "parameter settings.").format(time() - start,
                len(grid_search.grid_scores_)))

    top_params = report(grid_search.grid_scores_, 3)
    return  top_params



def get_code(tree, feature_names, target_names,
             spacer_base="    "):
    """Produce pseudo-code for decision tree.

    Args
    ----
    tree -- scikit-leant Decision Tree.
    feature_names -- list of feature names.
    target_names -- list of target (class) names.
    spacer_base -- used for spacing code (default: "    ").

    Notes
    -----
    based on http://stackoverflow.com/a/30104792.
    """
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            print(spacer + "if ( " + features[node] + " <= " + \
                  str(threshold[node]) + " ) {")
            if left[node] != -1:
                    recurse (left, right, threshold, features,
                             left[node], depth+1)
            print(spacer + "}\n" + spacer +"else {")
            if right[node] != -1:
                    recurse (left, right, threshold, features,
                             right[node], depth+1)
            print(spacer + "}")
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1],
                            target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print(spacer + "return " + str(target_name) + \
                      " ( " + str(target_count) + " examples )")

    recurse(left, right, threshold, features, 0, 0)


def visualize_tree(tree, feature_names, fn="dt"):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn Decision Tree.
    feature_names -- list of feature names.
    fn -- [string], root of filename, default `dt`.
    """
    dotfile = fn + ".dot"
    pngfile = fn + ".png"

    with open(dotfile, 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", dotfile, "-o", pngfile]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, "
             "to produce visualization")



# print(train.head())
#
# train.groupby('thick').agg('sum')[['class_1','class_2','class_3','class_5','class_U']].plot(kind='bar', figsize=(25, 7),
#                                                           stacked=True, colors=['g', 'r','y','b','w']);




params = {
    'axes.labelsize': "large",
    'xtick.labelsize': 'x-large',
    'legend.fontsize': 20,
    'figure.dpi': 150,
    'figure.figsize': [25, 7]
}
plot.rcParams.update(params)


train = pd.read_csv('Input/train.csv')
test = pd.read_csv('Input/test.csv')
print(train.shape)
train.head()
combined = train.append(test)
print(combined.shape)


balanceData()
dropEmptyColumn()
MappingContinues()
MapCategorical()
FillMissingValues()

train=combined.iloc[0:198]
test=combined.iloc[199:299]

param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 10, 20],
              "max_depth": [2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }

dt = DecisionTreeClassifier()
train = train.reset_index(drop=True)

y_train = train['classes']
targets = train["classes"].unique()
print(y_train.head())
print(list(train))

tt=train.drop(['classes'],axis=1)
# tt=tt.drop(['class_1'], axis=1)
# tt=tt.drop(['class_2'], axis=1)
# tt=tt.drop(['class_3'], axis=1)
# tt=tt.drop(['class_5'], axis=1)
# tt=tt.drop(['class_U'], axis=1)
# tt=tt.drop(['index'], axis=1)

tt = tt.reset_index(drop=True)
x_train = tt.values
features=list(tt)

print(x_train)

ts_gs = run_gridsearch(x_train, y_train, dt, param_grid, cv=10)

print("\n\n-- Testing best parameters [Grid]...")
dt_ts_gs = DecisionTreeClassifier(**ts_gs)
scores = cross_val_score(dt_ts_gs, x_train, y_train, cv=10)
print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),scores.std()),"\n\n" )


print("\n-- get_code for best parameters [Grid]:", "\n\n")
dt_ts_gs.fit(x_train,y_train)
get_code(dt_ts_gs, features, targets)
# visualize_tree(dt_ts_gs, features, fn="grid_best")


tt_test=test.drop(['classes'],axis=1)
# tt=tt.drop(['class_1'], axis=1)
# tt=tt.drop(['class_2'], axis=1)
# tt=tt.drop(['class_3'], axis=1)
# tt=tt.drop(['class_5'], axis=1)
# tt=tt.drop(['class_U'], axis=1)
# tt_test=tt_test.drop(['index'], axis=1)

tt_test = tt_test.reset_index(drop=True)
x_test = tt_test.values
y_pred = dt_ts_gs.predict(x_test)
submission = pd.DataFrame({
        "Classes": y_pred
    })
submission = submission.replace(6, 'U')
submission = submission.replace(1, '1')
submission = submission.replace(2, '2')
submission = submission.replace(3, '3')
submission = submission.replace(5, '5')


print(y_pred)

submission.to_csv('results.csv', index=True)