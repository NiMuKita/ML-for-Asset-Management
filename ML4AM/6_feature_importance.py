import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# code snippet 6.4 - clustered MDI
def groupMeanStd(df0, clstrs):
    """
    Create mean-std matrix

    :df0: initial DataFrame
    :clstrs: our clusters

    :return: out - DataFrame with mean and std values
    """

    out = pd.DataFrame(columns=['mean', 'std'])

    for i, j in clstrs.items():
        df1 = df0[j].sum(axis=1)
        out.loc['C_' + str(i), 'mean'] = df1.mean()
        out.loc['C_' + str(i), 'std'] = df1.std() * df1.shape[0] ** -.5

    return out


def featImpMDI_Clustered(fit, featNames, clstrs):
    """
    Get feature importance

    :fit: our model
    :featNames: names of features
    :clstrs: our clasters

    :return: feature importance
    """

    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan)  # because max_features=1
    imp = groupMeanStd(df0, clstrs)
    imp /= imp['mean'].sum()

    return imp


# code snippet 6.5 - clustered MDA
def featImpMDA_Clustered(clf, X, y, clstrs, n_splits=10):
    """
    return clastered features importants

    :clf: our classifier
    :X: features set
    :y: target set
    :clstrs: our clusters
    :n_splits: number of splits

    return feature importance
    """

    cvGen = KFold(n_splits=n_splits)
    scr0, scr1 = pd.Series(dtype='float64'), pd.DataFrame(columns=clstrs.keys())
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0, = X.iloc[train, :], y.iloc[train]
        X1, y1 = X.iloc[test, :], y.iloc[test]
        fit = clf.fit(X=X0, y=y0)
        prob = fit.predict_proba(X1)
        scr0.loc[i] = -log_loss(y1, prob, labels=clf.classes_)
        for j in scr1.columns:
            X1_ = X1.copy(deep=True)
            for k in clstrs[j]:
                np.random.shuffle(X1_[k].values)  # shuffle clusters
            prob = fit.predict_proba(X1_)
            scr1.loc[i, j] = -log_loss(y1, prob, labels=clf.classes_)
        imp = (-1 * scr1).add(scr0, axis=0)
        imp = imp / (-1 * scr1)
        imp = pd.concat({'mean': imp.mean(), 'std': imp.std() * imp.shape[0] ** -.5}, axis=1)
        imp.index = ['C_' + str(i) for i in imp.index]

    return imp