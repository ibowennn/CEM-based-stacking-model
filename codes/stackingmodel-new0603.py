# %%
import os
from operator import concat

from sklearn import feature_selection

os.chdir('/home/bowen/git/coding/python/CEM-stackingmodel/')
print(os.getcwd())
# %%
"""
该模型使用了person相关系数筛选组学特征，尝试更换特征筛选方法，即lasso回归结合嵌入法进行特征筛选
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

%matplotlib inline

import copy
import warnings
from itertools import product

import joblib
import matplotlib.style as style
import mglearn
import palettable
import pandas_profiling as ppf
import seaborn as sns
import xgboost as xgb
from scipy import interp
from sklearn import datasets, metrics, tree
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, brier_score_loss,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, Normalizer, OneHotEncoder,
                                   StandardScaler)
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import plot_importance
from xgboost.sklearn import XGBClassifier

warnings.filterwarnings("ignore")

# %%
data_quantatitivecesm = pd.read_csv(
    r'./data/data_quantatitivecesm_1107-65536.csv')

scaler = MinMaxScaler()

radiomics_low_original = pd.read_csv('./data/radiomics_low_original_1030.csv')
radiomics_low_wavelet = pd.read_csv('./data/radiomics_low_wavelet_1030.csv')
radiomics_low_squareroot = pd.read_csv(
    './data/radiomics_low_squareroot_1030.csv')
radiomics_low_square = pd.read_csv('./data/radiomics_low_square_1030.csv')
radiomics_low_exponential = pd.read_csv(
    './data/radiomics_low_exponential_1030.csv')
radiomics_low_logarithm = pd.read_csv(
    './data/radiomics_low_logarithm_1030.csv')
# radiomics_low_lbp_2d = pd.read_csv('/data/radiomics_low_lbp_2d_1030.csv')
radiomics_low_gradient = pd.read_csv('./data/radiomics_low_gradient_1030.csv')

radiomics_re_original = pd.read_csv('./data/radiomics_re_original_1030.csv')
radiomics_re_wavelet = pd.read_csv('./data/radiomics_re_wavelet_1030.csv')
radiomics_re_squareroot = pd.read_csv(
    './data/radiomics_re_squareroot_1030.csv')
radiomics_re_square = pd.read_csv('./data/radiomics_re_square_1030.csv')
radiomics_re_exponential = pd.read_csv(
    './data/radiomics_re_exponential_1030.csv')
radiomics_re_logarithm = pd.read_csv('./data/radiomics_re_logarithm_1030.csv')
# radiomics_re_lbp_2d = pd.read_csv('./data/radiomics_re_lbp_2d_1030.csv')
radiomics_re_gradient = pd.read_csv('./data/radiomics_re_gradient_1030.csv')

patient = pd.read_csv('./data/patient-only-1030.csv')

# data_final = pd.read_csv('data_final_0905.csv')

# radiomics_re_wavelet = pd.concat([data_final, radiomics_re_wavelet],
#                                  axis=1,
#                                  sort=False)
# %%
# ppf.ProfileReport(data_quantatitivecesm)


# %%
def reorder(data):
    data = data.drop([
        'Unnamed: 0', 'SeriesTime', 'time', 'ViewPosition', 'CompressionForce',
        'breast_background', 'lesion_val', 'image_background',
        'value difference', 'bpe'
    ],
                     axis=1)

    rank_1 = data[data['rank'] == 1]
    rank_2 = data[data['rank'] == 2]

    rank_1 = rank_1.rename(
        columns={
            # 'SeriesTime': 'SeriesTime_1',
            # 'ViewPosition': 'ViewPosition_1',
            #                                   'ImageLaterality':'ImageLaterality_1',
            # 'CompressionForce': 'CompressionForce_1',
            'rank': 'rank_1',
            # 'time': 'time_1',
            # 'breast_background': 'breast_background_1',
            # 'lesion_val': 'lesion_val_1',
            # 'image_background': 'image_background_1',
            # 'value difference': 'value_difference_1',
        })
    rank_1 = pd.DataFrame(rank_1)
    rank_2 = rank_2.rename(
        columns={
            # 'SeriesTime': 'SeriesTime_2',
            # 'ViewPosition': 'ViewPosition_2',
            # #                                   'ImageLaterality':'ImageLaterality_2',
            # 'CompressionForce': 'CompressionForce_2',
            'rank': 'rank_2',
            # 'time': 'time_2',
            # 'breast_background': 'breast_background_2',
            # 'lesion_val': 'lesion_val_2',
            # 'image_background': 'image_background_2',
            # 'value difference': 'value_difference_2',
        })
    rank_2 = pd.DataFrame(rank_2)

    result = pd.merge(rank_1, rank_2, on=['patientID', 'ImageLaterality'])
    result = pd.merge(patient, result, on=['patientID', 'ImageLaterality'])
    result = result.drop(['rank_1', 'rank_2'], axis=1)
    return result


def corr(data, threshold):
    columns = data[data.corr().columns]
    low_correlated_features = columns.columns[
        data.corr()['pathology'].abs() < threshold]
    data_selected = data.drop(low_correlated_features, axis=1)
    print("Persion picked " + str(data_selected.shape[1]) +
          " variables and eliminated the other " +
          str(data.shape[1] - data_selected.shape[1]))
    return data_selected


# %%
low_original = reorder(radiomics_low_original)
low_wavelet = reorder(radiomics_low_wavelet)
low_squareroot = reorder(radiomics_low_squareroot)
low_square = reorder(radiomics_low_square)
low_exponential = reorder(radiomics_low_exponential)
low_logarithm = reorder(radiomics_low_logarithm)
# low_lbp_2d = reorder(radiomics_low_lbp_2d)
low_gradient = reorder(radiomics_low_gradient)

re_original = reorder(radiomics_re_original)
re_wavelet = reorder(radiomics_re_wavelet)
re_squareroot = reorder(radiomics_re_squareroot)
re_square = reorder(radiomics_re_square)
re_exponential = reorder(radiomics_re_exponential)
re_logarithm = reorder(radiomics_re_logarithm)
# re_lbp_2d = reorder(radiomics_re_lbp_2d)
re_gradient = reorder(radiomics_re_gradient)


low_original = corr(low_original, .25)
low_wavelet = corr(low_wavelet, .25)
low_squareroot = corr(low_squareroot, .3)
low_square = corr(low_square, .25)
low_exponential = corr(low_exponential, .2)
low_logarithm = corr(low_logarithm, .3)
# low_lbp_2d = corr(low_lbp_2d, .1)
low_gradient = corr(low_gradient, .15)

re_original = corr(re_original, .2)
re_wavelet = corr(re_wavelet, .25)
re_squareroot = corr(re_squareroot, .15)
re_square = corr(re_square, .2)
re_exponential = corr(re_exponential, .15)
re_logarithm = corr(re_logarithm, .15)
# re_lbp_2d = corr(re_lbp_2d, .1)
re_gradient = corr(re_gradient, .15)
# %%
def get_best_model_and_accuracy(model, params, X, y):
    grid = GridSearchCV(model,
                        params,
                        scoring='accuracy',
                        cv=10,
                        error_score=0.,
                        n_jobs=-1)
    grid.fit(X, y)
    # print('Model: {}'.format(str(model)))
    print('\nBest Accuracy: {}'.format(grid.best_score_))
    print('Best Parameters: {}'.format(grid.best_params_))
    # print('Best estimator: {}'.format(grid.best_estimator_))
    print('Average Time to Fit (s): {}'.format(
        round(grid.cv_results_['mean_fit_time'].mean(), 3)))
    print('Average Time to Score (s): {}'.format(
        round(grid.cv_results_['mean_score_time'].mean(), 3)))

def get_matrix_info(test, pred, model, X_test):
    print(classification_report(test, pred))
    print('混淆矩阵：\n', metrics.confusion_matrix(test, pred))
    print('准确率：%.3f' % metrics.accuracy_score(test, pred))
    print('精准率：%.3f' % metrics.precision_score(test, pred))
    print('查全率：%.3f' % metrics.recall_score(test, pred))
    print('AUC值：%.3f' % roc_auc_score(test, model.predict_proba(X_test)[:, 1]))
    tn, fp, fn, tp = confusion_matrix(test, pred).ravel()
    sensitivity = tp / float(tp + fn)
    specificity = tn / float(tn + fp)
    print('sensitivity: {:.3f}'.format(sensitivity))
    print('specificity: {:.3f}'.format(specificity))



def select_feature(svc_pipe, X_train, X_test, y_train):
    svc_pipe.steps[0][1].fit(X_train, y_train)
    columns_selected = X_train.columns[svc_pipe.steps[0][1].get_support()]
    X_train = X_train[columns_selected]
    X_test = X_test[columns_selected]
    return X_train, X_test


def tpr_fpr(y_test, X_test, model):
    fpr, tpr, _ = metrics.roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = metrics.auc(fpr, tpr)

    return fpr, tpr, roc_auc

def performance(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    get_matrix_info(y_test, y_pred, model, X_test)

    fpr, tpr, roc_auc = tpr_fpr(y_test, X_test, model)

    _, axs = plt.subplots(1, 1, figsize=(5, 5))

    axs.plot([0, 1], [0, 1], 'k--', lw=0.5)
    axs.plot(fpr,
                tpr,
                label='(AUC = {0:0.3f})' ''.format(roc_auc))
    # axs[0, 0].set_title('ROC Curve', fontsize=15)
    axs.set_xlabel('False positive rate', fontsize=14)
    axs.set_ylabel('True positive rate', fontsize=14)
    axs.legend(loc='lower right', fontsize=15)
# %%
svc_selector = SelectFromModel(LinearSVC())

lr = LogisticRegression()
knn = KNeighborsClassifier()
d_tree = DecisionTreeClassifier()
forest = RandomForestClassifier()
svm = SVC()
xgb_1 = XGBClassifier()


lr_pipe_params = {'classifier__C': [1e-2, 1e-1, 1e0, 1e1, 1e2], 'classifier__penalty': ['l1', 'l2']}
knn_pipe_params = {'classifier__n_neighbors': range(0, 50, 1)}
tree_pipe_params = {'classifier__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
forest_pipe_params = {
    'classifier__n_estimators': range(0, 300, 20),
    'classifier__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
svm_pipe_params = {
    'classifier__C': [1e-2, 1e-1, 1, 10, 100],
    'classifier__gamma': [100, 10, 1, 0.1, 0.01, 0.001]
}
xgb_pipe_params = {
    'classifier__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'classifier__n_estimators': range(0, 300, 20)}


svc_pipe_lr = Pipeline([('select', svc_selector), ('classifier', lr)])
svc_pipe_knn = Pipeline([('select', svc_selector), ('classifier', knn)])
svc_pipe_tree = Pipeline([('select', svc_selector), ('classifier', d_tree)])
svc_pipe_forest = Pipeline([('select', svc_selector), ('classifier', forest)])
svc_pipe_svm = Pipeline([('select', svc_selector), ('classifier', svm)])
svc_pipe_xgb = Pipeline([('select', svc_selector), ('classifier', xgb_1)])


svc_pipe_lr_params = copy.deepcopy(lr_pipe_params)
svc_pipe_lr_params.update({
    'select__threshold': [.01, .02, .03, .04, .05, .06, .07, .08,.09, .1, .2, .3, .4, .5, .6, 'mean', 'median', '2.*mean'],
    'select__estimator__penalty': ['l1', 'l2'],
    'select__estimator__loss': ['squared_hinge', 'hinge'],
    'select__estimator__dual': [True, False]
})

svc_pipe_knn_params = copy.deepcopy(knn_pipe_params)
svc_pipe_knn_params.update({
    'select__threshold': [.01, .02, .03, .04, .05, .06, .07, .08,.09, .1, .2, .3, .4, .5, .6, 'mean', 'median', '2.*mean'],
    'select__estimator__penalty': ['l1', 'l2'],
    'select__estimator__loss': ['squared_hinge', 'hinge'],
    'select__estimator__dual': [True, False]
})

svc_pipe_tree_params = copy.deepcopy(tree_pipe_params)
svc_pipe_tree_params.update({
    'select__threshold': [.01, .02, .03, .04, .05, .06, .07, .08,.09, .1, .2, .3, .4, .5, .6, 'mean', 'median', '2.*mean'],
    'select__estimator__penalty': ['l1', 'l2'],
    'select__estimator__loss': ['squared_hinge', 'hinge'],
    'select__estimator__dual': [True, False]
})

svc_pipe_forest_params = copy.deepcopy(forest_pipe_params)
svc_pipe_forest_params.update({
    'select__threshold': [.01, .02, .03, .04, .05, .06, .07, .08,.09, .1, .2, .3, .4, .5, .6, 'mean', 'median', '2.*mean'],
    'select__estimator__penalty': ['l1', 'l2'],
    'select__estimator__loss': ['squared_hinge', 'hinge'],
    'select__estimator__dual': [True, False]
})

svc_pipe_svm_params = copy.deepcopy(svm_pipe_params)
svc_pipe_svm_params.update({
    'select__threshold': [.01, .02, .03, .04, .05, .06, .07, .08,.09, .1, .2, .3, .4, .5, .6, 'mean', 'median', '2.*mean'],
    'select__estimator__penalty': ['l1', 'l2'],
    'select__estimator__loss': ['squared_hinge', 'hinge'],
    'select__estimator__dual': [True, False]
})

svc_pipe_xgb_params = copy.deepcopy(xgb_pipe_params)
svc_pipe_xgb_params.update({
    'select__threshold': [.01, .02, .03, .04, .05, .06, .07, .08,.09, .1, .2, .3, .4, .5, .6, 'mean', 'median', '2.*mean'],
    'select__estimator__penalty': ['l1', 'l2'],
    'select__estimator__loss': ['squared_hinge', 'hinge'],
    'select__estimator__dual': [True, False]
})



# %%
# # CESM
#
data_quantatitivecesm.iloc[:, :15] = data_quantatitivecesm.iloc[:, :15].astype(
    'object')
data_quantatitivecesm['pattern_12'] = data_quantatitivecesm[
    'pattern_12'].astype('object')
data_quantatitivecesm['age'] = data_quantatitivecesm['age'].astype('int64')
data_quantatitivecesm['LE_value'] = data_quantatitivecesm['LE_value'].astype(
    'int64')
data_quantatitivecesm['pathology'] = data_quantatitivecesm['pathology'].astype(
    'int64')

dummy_cols = data_quantatitivecesm.drop(
    ['pathology', 'ImageLaterality', 'patientID', 'LBPE_1', 'LBPE_2','LE_value', 'enhanced_value_all','LE_EN_value', 'image_background_1', 'image_background_2'],
    1).select_dtypes(include=['object']).columns
scale_cols = data_quantatitivecesm.drop(
    ['pathology', 'ImageLaterality', 'patientID', 'LBPE_1', 'LBPE_2','LE_value', 'enhanced_value_all','LE_EN_value', 'image_background_1', 'image_background_2'],
    1).select_dtypes(include=['int64', 'float']).columns

# %%
#
#
X_train_cesm, X_test_cesm, y_train_cesm, y_test_cesm = train_test_split(
    data_quantatitivecesm.drop(
        ['pathology', 'ImageLaterality', 'patientID', 'LBPE_1', 'LBPE_2','LE_value', 'enhanced_value_all','LE_EN_value', 'image_background_1', 'image_background_2'], 1),
    data_quantatitivecesm['pathology'],
    test_size=0.3,
    random_state=42)
# %%
enc = OneHotEncoder(handle_unknown="ignore")
col_dummies = enc.fit_transform(X_train_cesm[dummy_cols]).toarray()
col_names = enc.get_feature_names()

cols = pd.DataFrame(col_dummies, columns=col_names, index=X_train_cesm.index)
X_train_cesm = pd.concat([X_train_cesm.drop(dummy_cols, 1), cols], axis=1)

X_test_col_dummies = enc.transform(X_test_cesm[dummy_cols]).toarray()
X_test_cols = pd.DataFrame(X_test_col_dummies,
                           columns=col_names,
                           index=X_test_cesm.index)
X_test_cesm = pd.concat([X_test_cesm.drop(dummy_cols, 1), X_test_cols], axis=1)

scaler = MinMaxScaler()
X_train_cesm[scale_cols] = scaler.fit_transform(X_train_cesm[scale_cols])
X_test_cesm[scale_cols] = scaler.transform(X_test_cesm[scale_cols])

# %%
%%time
get_best_model_and_accuracy(svc_pipe_lr, svc_pipe_lr_params, X_train_cesm, y_train_cesm)

get_best_model_and_accuracy(svc_pipe_knn, svc_pipe_knn_params, X_train_cesm, y_train_cesm)

get_best_model_and_accuracy(svc_pipe_tree, svc_pipe_tree_params, X_train_cesm, y_train_cesm)

get_best_model_and_accuracy(svc_pipe_forest, svc_pipe_forest_params, X_train_cesm, y_train_cesm)

get_best_model_and_accuracy(svc_pipe_svm, svc_pipe_svm_params, X_train_cesm, y_train_cesm)

get_best_model_and_accuracy(svc_pipe_xgb, svc_pipe_xgb_params, X_train_cesm, y_train_cesm)

# %%
svc_pipe_lr.set_params(
    **{
        'classifier__C': 1.0, 'classifier__penalty': 'l2', 'select__estimator__dual': False, 'select__estimator__loss': 'squared_hinge', 'select__estimator__penalty': 'l1', 'select__threshold': 0.09
    })

X_train_cesm_lr, X_test_cesm_lr = select_feature(svc_pipe_lr, X_train_cesm, X_test_cesm, y_train_cesm)
print(len(X_train_cesm_lr.columns),X_train_cesm_lr.columns)

best_model_cesm_lr = LogisticRegression(C=1, random_state=42)


best_model_cesm_lr.fit(X_train_cesm_lr, y_train_cesm)

performance(best_model_cesm_lr, X_train_cesm_lr, X_test_cesm_lr, y_train_cesm, y_test_cesm)

# %%
svc_pipe_knn.set_params(
    **{
        'classifier__n_neighbors': 9, 'select__estimator__dual': False, 'select__estimator__loss': 'squared_hinge', 'select__estimator__penalty': 'l1', 'select__threshold': 'median'
    })

X_train_cesm_knn, X_test_cesm_knn = select_feature(svc_pipe_knn, X_train_cesm, X_test_cesm, y_train_cesm)
print(len(X_train_cesm_knn.columns),X_train_cesm_knn.columns)

best_model_cesm_knn = KNeighborsClassifier(n_neighbors=9)


best_model_cesm_knn.fit(X_train_cesm_knn, y_train_cesm)

performance(best_model_cesm_knn, X_train_cesm_knn, X_test_cesm_knn, y_train_cesm, y_test_cesm)

# %%
svc_pipe_tree.set_params(
    **{
       'classifier__max_depth': 3, 'select__estimator__dual': True, 'select__estimator__loss': 'hinge', 'select__estimator__penalty': 'l2', 'select__threshold': 0.6
    })

X_train_cesm_tree, X_test_cesm_tree = select_feature(svc_pipe_tree, X_train_cesm, X_test_cesm, y_train_cesm)
print(len(X_train_cesm_tree.columns),X_train_cesm_tree.columns)

best_model_cesm_tree =  DecisionTreeClassifier(max_depth=3, random_state=80)

best_model_cesm_tree.fit(X_train_cesm_tree, y_train_cesm)

performance(best_model_cesm_tree, X_train_cesm_tree, X_test_cesm_tree, y_train_cesm, y_test_cesm)

# %%
svc_pipe_forest.set_params(
    **{
       'classifier__max_depth': 8, 'classifier__n_estimators': 20, 'select__estimator__dual': False, 'select__estimator__loss': 'squared_hinge', 'select__estimator__penalty': 'l2', 'select__threshold': 0.2
    })

X_train_cesm_forest, X_test_cesm_forest = select_feature(svc_pipe_forest, X_train_cesm, X_test_cesm, y_train_cesm)
print(len(X_train_cesm_forest.columns),X_train_cesm_forest.columns)

best_model_cesm_forest =  RandomForestClassifier(max_depth=8,
                                           n_estimators=20,
                                           random_state=42)

best_model_cesm_forest.fit(X_train_cesm_forest, y_train_cesm)

performance(best_model_cesm_forest, X_train_cesm_forest, X_test_cesm_forest, y_train_cesm, y_test_cesm)

# %%
svc_pipe_svm.set_params(
    **{
      'classifier__C': 1, 'classifier__gamma': 0.1, 'select__estimator__dual': False, 'select__estimator__loss': 'squared_hinge', 'select__estimator__penalty': 'l1', 'select__threshold': 'mean'
    })

X_train_cesm_svm, X_test_cesm_svm = select_feature(svc_pipe_svm, X_train_cesm, X_test_cesm, y_train_cesm)
print(len(X_train_cesm_svm.columns),X_train_cesm_svm.columns)

best_model_cesm_svm =  SVC(C=1, gamma=0.1, random_state=42, probability=True)

best_model_cesm_svm.fit(X_train_cesm_svm, y_train_cesm)

performance(best_model_cesm_svm, X_train_cesm_svm, X_test_cesm_svm, y_train_cesm, y_test_cesm)

# %%
svc_pipe_xgb.set_params(
    **{
     'classifier__max_depth': None, 'classifier__n_estimators': 180, 'select__estimator__dual': False, 'select__estimator__loss': 'squared_hinge', 'select__estimator__penalty': 'l1', 'select__threshold': 'median'
    })

X_train_cesm_xgb, X_test_cesm_xgb = select_feature(svc_pipe_xgb, X_train_cesm, X_test_cesm, y_train_cesm)
print(len(X_train_cesm_xgb.columns),X_train_cesm_xgb.columns)

best_model_cesm_xgb =  XGBClassifier(max_depth=None, n_estimators=100)

best_model_cesm_xgb.fit(X_train_cesm_xgb, y_train_cesm)

performance(best_model_cesm_xgb, X_train_cesm_xgb, X_test_cesm_xgb, y_train_cesm, y_test_cesm)

# %%
# original
#
#
#
#
#
#
X_train_low_original, X_test_low_original, y_train_low_original, y_test_low_original = train_test_split(
    low_original.drop(['pathology', 'ImageLaterality', 'patientID'], 1),
    low_original['pathology'],
    test_size=0.3,
    random_state=42)
scaler = MinMaxScaler()
X_train_low_original.iloc[:, :] = scaler.fit_transform(
    X_train_low_original.iloc[:, :])

X_test_low_original.iloc[:, :] = scaler.transform(
    X_test_low_original.iloc[:, :])
# %%
%%time
param_low_org_lr = get_best_model_and_accuracy(svc_pipe_lr, svc_pipe_lr_params, X_train_low_original, y_train_low_original)

param_low_org_knn = get_best_model_and_accuracy(svc_pipe_knn, svc_pipe_knn_params, X_train_low_original, y_train_low_original)

param_low_org_tree = get_best_model_and_accuracy(svc_pipe_tree, svc_pipe_tree_params, X_train_low_original, y_train_low_original)

param_low_org_forest = get_best_model_and_accuracy(svc_pipe_forest, svc_pipe_forest_params, X_train_low_original, y_train_low_original)

param_low_org_svm = get_best_model_and_accuracy(svc_pipe_svm, svc_pipe_svm_params, X_train_low_original, y_train_low_original)

# %%
param_low_org_xgb = get_best_model_and_accuracy(svc_pipe_xgb, svc_pipe_xgb_params, X_train_low_original, y_train_low_original)

# %%
svc_pipe_lr.set_params(
    **{
     'classifier__C': 10.0, 'classifier__penalty': 'l2', 'select__estimator__dual': False, 'select__estimator__loss': 'squared_hinge', 'select__estimator__penalty': 'l1', 'select__threshold': 0.3
    })

X_train_low_original_lr, X_test_low_original_lr = select_feature(svc_pipe_lr, X_train_low_original, X_test_low_original, y_train_low_original)
print(len(X_train_low_original_lr.columns),X_train_low_original_lr.columns)

best_model_low_original_lr = LogisticRegression(C=10, random_state=42)


best_model_low_original_lr.fit(X_train_low_original_lr, y_train_low_original)

performance(best_model_low_original_lr, X_train_low_original_lr, X_test_low_original_lr, y_train_low_original, y_test_low_original)
# %%
svc_pipe_knn.set_params(
    **{
       'classifier__n_neighbors': 25, 'select__estimator__dual': True, 'select__estimator__loss': 'hinge', 'select__estimator__penalty': 'l2', 'select__threshold': 0.09
    })

X_train_low_original_knn, X_test_low_original_knn = select_feature(svc_pipe_knn, X_train_low_original, X_test_low_original, y_train_low_original)
print(len(X_train_low_original_knn.columns),X_train_low_original_knn.columns)

best_model_low_original_knn = KNeighborsClassifier(n_neighbors=25)


best_model_low_original_knn.fit(X_train_low_original_knn, y_train_low_original)

performance(best_model_low_original_knn, X_train_low_original_knn, X_test_low_original_knn, y_train_low_original, y_test_low_original)

# %%
svc_pipe_tree.set_params(
    **{
       'classifier__max_depth': 2, 'select__estimator__dual': False, 'select__estimator__loss': 'squared_hinge', 'select__estimator__penalty': 'l1', 'select__threshold': 0.01
    })

X_train_low_original_tree, X_test_low_original_tree = select_feature(svc_pipe_tree, X_train_low_original, X_test_low_original, y_train_low_original)
print(len(X_train_low_original_tree.columns)
    # ,X_train_low_original_tree.columns
    )

best_model_low_original_tree =  DecisionTreeClassifier(max_depth=2, random_state=80)

best_model_low_original_tree.fit(X_train_low_original_tree, y_train_low_original)

performance(best_model_low_original_tree, X_train_low_original_tree, X_test_low_original_tree, y_train_low_original, y_test_low_original)

# %%
svc_pipe_forest.set_params(
    **{
       'classifier__max_depth': 1, 'classifier__n_estimators': 180, 'select__estimator__dual': False, 'select__estimator__loss': 'squared_hinge', 'select__estimator__penalty': 'l2', 'select__threshold': 'median'
    })

X_train_low_original_forest, X_test_low_original_forest = select_feature(svc_pipe_forest, X_train_low_original, X_test_low_original, y_train_low_original)
print(len(X_train_low_original_forest.columns)
    # ,X_train_low_original_forest.columns
    )

best_model_low_original_forest =  RandomForestClassifier(max_depth=1,
                                           n_estimators=180,
                                           random_state=42)

best_model_low_original_forest.fit(X_train_low_original_forest, y_train_low_original)

performance(best_model_low_original_forest, X_train_low_original_forest, X_test_low_original_forest, y_train_low_original, y_test_low_original)

# %%
svc_pipe_svm.set_params(
    **{
      'classifier__C': 10, 'classifier__gamma': 0.1, 'select__estimator__dual': True, 'select__estimator__loss': 'squared_hinge', 'select__estimator__penalty': 'l2', 'select__threshold': 0.03
    })

X_train_low_original_svm, X_test_low_original_svm = select_feature(svc_pipe_svm, X_train_low_original, X_test_low_original, y_train_low_original)
print(len(X_train_low_original_svm.columns)
# ,X_train_low_original_svm.columns
)

best_model_low_original_svm =  SVC(C=10, gamma=0.1, random_state=42, probability=True)

best_model_low_original_svm.fit(X_train_low_original_svm, y_train_low_original)

performance(best_model_low_original_svm, X_train_low_original_svm, X_test_low_original_svm, y_train_low_original, y_test_low_original)

# %%
svc_pipe_xgb.set_params(
    **{
     'classifier__max_depth': 1, 'classifier__n_estimators': 40, 'select__estimator__dual': True, 'select__estimator__loss': 'squared_hinge', 'select__estimator__penalty': 'l2', 'select__threshold': 0.5
    })

X_train_low_original_xgb, X_test_low_original_xgb = select_feature(svc_pipe_xgb, X_train_low_original, X_test_low_original, y_train_low_original)
print(len(X_train_low_original_xgb.columns),X_train_low_original_xgb.columns)

best_model_low_original_xgb =  XGBClassifier(max_depth=1, n_estimators=40)

best_model_low_original_xgb.fit(X_train_low_original_xgb, y_train_low_original)

performance(best_model_low_original_xgb, X_train_low_original_xgb, X_test_low_original_xgb, y_train_low_original, y_test_low_original)


# %%
# logarithm
#
#
#
#
#
#
X_train_re_logarithm, X_test_re_logarithm, y_train_re_logarithm, y_test_re_logarithm = train_test_split(
    re_logarithm.drop(['pathology', 'ImageLaterality', 'patientID'], 1),
    re_logarithm['pathology'],
    test_size=0.3,
    random_state=42)
scaler = MinMaxScaler()
X_train_re_logarithm.iloc[:, :] = scaler.fit_transform(
    X_train_re_logarithm.iloc[:, :])

X_test_re_logarithm.iloc[:, :] = scaler.transform(
    X_test_re_logarithm.iloc[:, :])

# %%
%%time
param_re_logarithm_lr = get_best_model_and_accuracy(svc_pipe_lr, svc_pipe_lr_params, X_train_re_logarithm, y_train_re_logarithm)

param_re_logarithm_knn = get_best_model_and_accuracy(svc_pipe_knn, svc_pipe_knn_params, X_train_re_logarithm, y_train_re_logarithm)

param_re_logarithm_tree = get_best_model_and_accuracy(svc_pipe_tree, svc_pipe_tree_params, X_train_re_logarithm, y_train_re_logarithm)

param_re_logarithm_forest = get_best_model_and_accuracy(svc_pipe_forest, svc_pipe_forest_params, X_train_re_logarithm, y_train_re_logarithm)

param_re_logarithm_svm = get_best_model_and_accuracy(svc_pipe_svm, svc_pipe_svm_params, X_train_re_logarithm, y_train_re_logarithm)

# %%
param_re_logarithm_xgb = get_best_model_and_accuracy(svc_pipe_xgb, svc_pipe_xgb_params, X_train_re_logarithm, y_train_re_logarithm)

# %%
svc_pipe_lr.set_params(
    **{
        'classifier__C': 0.01, 'classifier__penalty': 'l2', 'select__estimator__dual': True, 'select__estimator__loss': 'squared_hinge', 'select__estimator__penalty': 'l2', 'select__threshold': 0.01
    })

X_train_re_logarithm_lr, X_test_re_logarithm_lr = select_feature(svc_pipe_lr, X_train_re_logarithm, X_test_re_logarithm, y_train_re_logarithm)
print(len(X_train_re_logarithm_lr.columns),X_train_re_logarithm_lr.columns)

best_model_re_logarithm_lr = LogisticRegression(C=0.01, random_state=42)


best_model_re_logarithm_lr.fit(X_train_re_logarithm_lr, y_train_re_logarithm)

performance(best_model_re_logarithm_lr, X_train_re_logarithm_lr, X_test_re_logarithm_lr, y_train_re_logarithm, y_test_re_logarithm)
# %%
svc_pipe_knn.set_params(
    **{
       'classifier__n_neighbors': 8, 'select__estimator__dual': True, 'select__estimator__loss': 'squared_hinge', 'select__estimator__penalty': 'l2', 'select__threshold': 0.08
    })

X_train_re_logarithm_knn, X_test_re_logarithm_knn = select_feature(svc_pipe_knn, X_train_re_logarithm, X_test_re_logarithm, y_train_re_logarithm)
print(len(X_train_re_logarithm_knn.columns),X_train_re_logarithm_knn.columns)

best_model_re_logarithm_knn = KNeighborsClassifier(n_neighbors=9)


best_model_re_logarithm_knn.fit(X_train_re_logarithm_knn, y_train_re_logarithm)

performance(best_model_re_logarithm_knn, X_train_re_logarithm_knn, X_test_re_logarithm_knn, y_train_re_logarithm, y_test_re_logarithm)

# %%
svc_pipe_tree.set_params(
    **{
       'classifier__max_depth': 5, 'select__estimator__dual': True, 'select__estimator__loss': 'hinge', 'select__estimator__penalty': 'l2', 'select__threshold': 0.2
    })

X_train_re_logarithm_tree, X_test_re_logarithm_tree = select_feature(svc_pipe_tree, X_train_re_logarithm, X_test_re_logarithm, y_train_re_logarithm)
print(len(X_train_re_logarithm_tree.columns)
    # ,X_train_re_logarithm_tree.columns
    )

best_model_re_logarithm_tree =  DecisionTreeClassifier(max_depth=5, random_state=80)

best_model_re_logarithm_tree.fit(X_train_re_logarithm_tree, y_train_re_logarithm)

performance(best_model_re_logarithm_tree, X_train_re_logarithm_tree, X_test_re_logarithm_tree, y_train_re_logarithm, y_test_re_logarithm)

# %%
svc_pipe_forest.set_params(
    **{
      'classifier__max_depth': 9, 'classifier__n_estimators': 40, 'select__estimator__dual': False, 'select__estimator__loss': 'squared_hinge', 'select__estimator__penalty': 'l1', 'select__threshold': 0.05
    })

X_train_re_logarithm_forest, X_test_re_logarithm_forest = select_feature(svc_pipe_forest, X_train_re_logarithm, X_test_re_logarithm, y_train_re_logarithm)
print(len(X_train_re_logarithm_forest.columns)
    # ,X_train_re_logarithm_forest.columns
    )

best_model_re_logarithm_forest =  RandomForestClassifier(max_depth=9,
                                           n_estimators=40,
                                           random_state=42)

best_model_re_logarithm_forest.fit(X_train_re_logarithm_forest, y_train_re_logarithm)

performance(best_model_re_logarithm_forest, X_train_re_logarithm_forest, X_test_re_logarithm_forest, y_train_re_logarithm, y_test_re_logarithm)

# %%
svc_pipe_svm.set_params(
    **{
      'classifier__C': 10, 'classifier__gamma': 1, 'select__estimator__dual': True, 'select__estimator__loss': 'hinge', 'select__estimator__penalty': 'l2', 'select__threshold': 0.04
    })

X_train_re_logarithm_svm, X_test_re_logarithm_svm = select_feature(svc_pipe_svm, X_train_re_logarithm, X_test_re_logarithm, y_train_re_logarithm)
print(len(X_train_re_logarithm_svm.columns)
# ,X_train_re_logarithm_svm.columns
)

best_model_re_logarithm_svm =  SVC(C=10, gamma=1, random_state=42, probability=True)

best_model_re_logarithm_svm.fit(X_train_re_logarithm_svm, y_train_re_logarithm)

performance(best_model_re_logarithm_svm, X_train_re_logarithm_svm, X_test_re_logarithm_svm, y_train_re_logarithm, y_test_re_logarithm)

# %%
svc_pipe_xgb.set_params(
    **{
     'classifier__max_depth': 1, 'classifier__n_estimators': 80, 'select__estimator__dual': False, 'select__estimator__loss': 'squared_hinge', 'select__estimator__penalty': 'l1', 'select__threshold': 0.06
    })

X_train_re_logarithm_xgb, X_test_re_logarithm_xgb = select_feature(svc_pipe_xgb, X_train_re_logarithm, X_test_re_logarithm, y_train_re_logarithm)
print(len(X_train_re_logarithm_xgb.columns),X_train_re_logarithm_xgb.columns)

best_model_re_logarithm_xgb =  XGBClassifier(max_depth=1, n_estimators=80)

best_model_re_logarithm_xgb.fit(X_train_re_logarithm_xgb, y_train_re_logarithm)

performance(best_model_re_logarithm_xgb, X_train_re_logarithm_xgb, X_test_re_logarithm_xgb, y_train_re_logarithm, y_test_re_logarithm)



# %%
X = pd.merge(re_logarithm[X_train_re_logarithm_svm.columns |[ 'patientID', 'ImageLaterality', 'pathology']],
             low_original[X_train_low_original_xgb.columns |['patientID', 'ImageLaterality', 'pathology']],
             on=['patientID', 'ImageLaterality', 'pathology'])

X = pd.merge(X,
             data_quantatitivecesm,
             on=['patientID', 'ImageLaterality', 'pathology'])

y = low_original['pathology']
X = X.drop(['ImageLaterality', 'patientID', 'pathology', 'LBPE_1', 'LBPE_2','LE_value', 'enhanced_value_all','LE_EN_value', 'image_background_1', 'image_background_2'], 1)


# %%
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)

dummy_cols = X.select_dtypes(include=['object']).columns
scale_cols = X.select_dtypes(include=['int64', 'float64']).columns

enc = OneHotEncoder(handle_unknown="ignore")
col_dummies = enc.fit_transform(X_train[dummy_cols]).toarray()
col_names = enc.get_feature_names()

cols = pd.DataFrame(col_dummies, columns=col_names, index=X_train.index)
X_train = pd.concat([X_train.drop(dummy_cols, 1), cols], axis=1)

X_test_col_dummies = enc.transform(X_test[dummy_cols]).toarray()
X_test_cols = pd.DataFrame(X_test_col_dummies,
                           columns=col_names,
                           index=X_test.index)
X_test = pd.concat([X_test.drop(dummy_cols, 1), X_test_cols], axis=1)

scaler = MinMaxScaler()

X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])

# %%
X_train = X_train[X_train_low_original_xgb.columns | X_train_re_logarithm_svm.columns | X_train_cesm_lr.columns]

X_test = X_test[X_train_low_original_xgb.columns | X_train_re_logarithm_svm.columns | X_train_cesm_lr.columns]
# %%
# %%
####################固化模型
# 保存该对象-joblib

# %%
# 载入模型
best_model_cesm_lr = joblib.load('./data/model/model_cesm_lr_05226.pkl')
best_model_re_logarithm_svm = joblib.load('./data/model/model_re_log_svm_0523.pkl')
best_model_low_original_xgb = joblib.load('./data/model/model_low_original_xgb_0527.pkl')
# sclf_xgb = joblib.load('./data/model/model_sclf_xgb_0523.pkl')
sclf_rf = joblib.load('./data/model/model_sclf_rf_0527.pkl')

# %%
# STACKING
from mlxtend.classifier import StackingClassifier, StackingCVClassifier
from mlxtend.feature_selection import ColumnSelector
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pipe1 = make_pipeline(ColumnSelector(cols=list(range(3,33))), best_model_re_logarithm_svm)
pipe2 = make_pipeline(ColumnSelector(cols=(33,34,35,36,37,38)), best_model_low_original_xgb)
pipe3 = make_pipeline(ColumnSelector(cols= (0,1,2,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,)), best_model_cesm_lr)

# %%
sclf_lr = StackingClassifier(
    classifiers=[pipe1, pipe2, pipe3],
    use_probas=True,
    #  average_probas=True,
    meta_classifier=LogisticRegression())

sclf_lr_params = {
    'meta_classifier__C': [1e-2, 1e-1, 1e0, 1e1, 1e2],
    'meta_classifier__penalty': ['l1', 'l2']
}

sclf_knn = StackingClassifier(
    classifiers=[pipe1, pipe2, pipe3],
    use_probas=True,
    #   average_probas=True,
    meta_classifier=KNeighborsClassifier())

sclf_knn_params = {'meta_classifier__n_neighbors': range(0, 50, 1)}

sclf_dt = StackingClassifier(
    classifiers=[pipe1, pipe2, pipe3],
    use_probas=True,
    #  average_probas=True,
    meta_classifier=DecisionTreeClassifier())
sclf_dt_params = {
    'meta_classifier__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'meta_classifier__min_samples_leaf': np.linspace(0.1,
                                                     0.8,
                                                     5,
                                                     endpoint=True)
}

sclf_rf = StackingClassifier(
    classifiers=[pipe1, pipe2, pipe3],
    use_probas=True,
    #  average_probas=True,
    meta_classifier=RandomForestClassifier())

sclf_rf_params = {
    'meta_classifier__n_estimators': range(0, 500, 20),
    'meta_classifier__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

sclf_svm = StackingClassifier(
    classifiers=[pipe1, pipe2, pipe3],
    use_probas=True,
    #   average_probas=True,
    meta_classifier=SVC())
sclf_svm_params = {
    'meta_classifier__C': [1e-2, 1e-1, 1, 10, 100],
    'meta_classifier__gamma': [100, 10, 1, 0.1, 0.01, 0.001]
}

sclf_xgb = StackingClassifier(
    classifiers=[pipe1, pipe2, pipe3],
    use_probas=True,
    #   average_probas=True,
    meta_classifier=XGBClassifier())

sclf_xgb_params = {
    'meta_classifier__max_depth': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'meta_classifier__n_estimators': range(0, 500, 10)
}

get_best_model_and_accuracy(sclf_lr, sclf_lr_params, X_train, y_train)
get_best_model_and_accuracy(sclf_knn, sclf_knn_params, X_train, y_train)
get_best_model_and_accuracy(sclf_dt, sclf_dt_params, X_train, y_train)
get_best_model_and_accuracy(sclf_rf, sclf_rf_params, X_train, y_train)
get_best_model_and_accuracy(sclf_svm, sclf_svm_params, X_train, y_train)
get_best_model_and_accuracy(sclf_xgb, sclf_xgb_params, X_train, y_train)
# %%
sclf_lr = StackingClassifier(
    classifiers=[pipe1, pipe2, pipe3],
    use_probas=True,
    #  average_probas=True,
    meta_classifier=LogisticRegression(
        C=1, 
        penalty='l2',
        random_state=42
        ))

sclf_knn = StackingClassifier(
    classifiers=[pipe1, pipe2, pipe3],
    use_probas=True,
    # average_probas=True,
    meta_classifier=KNeighborsClassifier(
        # n_neighbors=12
        n_neighbors=9,
        ))

sclf_dt = StackingClassifier(
    classifiers=[pipe1, pipe2, pipe3],
    use_probas=True,
    #  average_probas=True,
    meta_classifier=DecisionTreeClassifier(
        # max_depth=8,
        # min_samples_split=0.45,
        max_depth=8,
        min_samples_split=0.05,
        random_state=42
        ))

sclf_rf = StackingClassifier(
    classifiers=[pipe1, pipe2, pipe3],
    use_probas=True,
    #  average_probas=True,
    meta_classifier=RandomForestClassifier(
        max_depth=2, 
        # n_estimators=20,
        # n_estimators=40,
        random_state=140
        ))

sclf_svm = StackingClassifier(
    classifiers=[pipe1, pipe2, pipe3],
    use_probas=True,
    #   average_probas=True,
    meta_classifier=SVC(
        # C=0.1, 
        # gamma=10, 
        C=0.1, 
        gamma=1, 
        probability=True,
        random_state=10086
        ))

sclf_xgb = StackingClassifier(
    classifiers=[pipe1, pipe2, pipe3],
    use_probas=True,
    #   average_probas=True,
    meta_classifier=XGBClassifier(
        # max_depth=1, 
        # n_esimators=50
        max_depth= 1,
        n_esimators=10
        ))


sclf_lr.fit(X_train, y_train)
sclf_knn.fit(X_train, y_train)
sclf_dt.fit(X_train, y_train)
sclf_rf.fit(X_train, y_train)
sclf_svm.fit(X_train, y_train)
sclf_xgb.fit(X_train, y_train)

y_pred_lr = sclf_lr.predict(X_test)
y_pred_knn = sclf_knn.predict(X_test)
y_pred_dt = sclf_dt.predict(X_test)
y_pred_forest = sclf_rf.predict(X_test)
y_pred_svm = sclf_svm.predict(X_test)
y_pred_xgb = sclf_xgb.predict(X_test)

get_matrix_info(y_test, y_pred_lr, sclf_lr, X_test)
get_matrix_info(y_test, y_pred_knn, sclf_knn, X_test)
get_matrix_info(y_test, y_pred_dt, sclf_dt, X_test)
get_matrix_info(y_test, y_pred_forest, sclf_rf, X_test)
get_matrix_info(y_test, y_pred_svm, sclf_svm, X_test)
get_matrix_info(y_test, y_pred_xgb, sclf_xgb, X_test)

def sclf_auc_comparison(X_train, X_test, y_train, y_test):
    def tpr_fpr(y_test, X_test, model):
        fpr, tpr, _ = metrics.roc_curve(y_test,
                                        model.predict_proba(X_test)[:, 1])
        roc_auc = metrics.auc(fpr, tpr)

        return fpr, tpr, roc_auc

    fpr_lr, tpr_lr, roc_auc_lr = tpr_fpr(y_test, X_test, sclf_lr)
    fpr_knn, tpr_knn, roc_auc_knn = tpr_fpr(y_test, X_test, sclf_knn)
    fpr_dt, tpr_dt, roc_auc_dt = tpr_fpr(y_test, X_test, sclf_dt)
    fpr_rf, tpr_rf, roc_auc_rf = tpr_fpr(y_test, X_test, sclf_rf)
    fpr_svm, tpr_svm, roc_auc_svm = tpr_fpr(y_test, X_test, sclf_svm)
    fpr_xgb, tpr_xgb, roc_auc_xgb = tpr_fpr(y_test, X_test, sclf_xgb)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].plot([0, 1], [0, 1], 'k--', lw=0.5)
    axs[0, 0].plot(fpr_lr,
                   tpr_lr,
                   label='LR (AUC = {0:0.3f})'
                   ''.format(roc_auc_lr))
    # axs[0, 0].set_title('ROC Curve', fontsize=15)
    axs[0, 0].set_xlabel('False positive rate', fontsize=14)
    axs[0, 0].set_ylabel('True positive rate', fontsize=14)
    axs[0, 0].legend(loc='lower right', fontsize=15)

    axs[0, 1].plot([0, 1], [0, 1], 'k--', lw=0.5)
    axs[0, 1].plot(fpr_knn,
                   tpr_knn,
                   label='KNN (AUC = {0:0.3f})'
                   ''.format(roc_auc_knn))
    # axs[0, 1].set_title('ROC Curve', fontsize=15)
    axs[0, 1].set_xlabel('False positive rate', fontsize=14)
    axs[0, 1].set_ylabel('True positive rate', fontsize=14)
    axs[0, 1].legend(loc='lower right', fontsize=15)

    axs[0, 2].plot([0, 1], [0, 1], 'k--', lw=0.5)
    axs[0, 2].plot(fpr_dt,
                   tpr_dt,
                   label='DT (AUC = {0:0.3f})'
                   ''.format(roc_auc_dt))
    # axs[0, 2].set_title('ROC Curve', fontsize=15)
    axs[0, 2].set_xlabel('False positive rate', fontsize=14)
    axs[0, 2].set_ylabel('True positive rate', fontsize=14)
    axs[0, 2].legend(loc='lower right', fontsize=15)

    axs[1, 0].plot([0, 1], [0, 1], 'k--', lw=0.5)
    axs[1, 0].plot(fpr_rf,
                   tpr_rf,
                   label='RF (AUC = {0:0.3f})'
                   ''.format(roc_auc_rf))
    # axs[1, 0].set_title('ROC Curve', fontsize=15)
    axs[1, 0].set_xlabel('False positive rate', fontsize=14)
    axs[1, 0].set_ylabel('True positive rate', fontsize=14)
    axs[1, 0].legend(loc='lower right', fontsize=15)

    axs[1, 1].plot([0, 1], [0, 1], 'k--', lw=0.5)
    axs[1, 1].plot(fpr_svm,
                   tpr_svm,
                   label='SVM (AUC = {0:0.3f})'
                   ''.format(roc_auc_svm))
    # axs[1, 1].set_title('ROC Curve', fontsize=15)
    axs[1, 1].set_xlabel('False positive rate', fontsize=14)
    axs[1, 1].set_ylabel('True positive rate', fontsize=14)
    axs[1, 1].legend(loc='lower right', fontsize=15)

    axs[1, 2].plot([0, 1], [0, 1], 'k--', lw=0.5)
    axs[1, 2].plot(fpr_xgb,
                   tpr_xgb,
                   label='XGBoost (AUC = {0:0.3f})'
                   ''.format(roc_auc_xgb))
    # axs[1, 2].set_title('ROC Curve', fontsize=15)
    axs[1, 2].set_xlabel('False positive rate', fontsize=14)
    axs[1, 2].set_ylabel('True positive rate', fontsize=14)
    axs[1, 2].legend(loc='lower right', fontsize=15)

    plt.style.use('fivethirtyeight')
    fig.suptitle('ROC Curve Comparison')  #整个图的标题


sclf_auc_comparison(X_train, X_test, y_train, y_test)

# %%
proba_sclf = sclf_rf.predict_proba(X_test)[:, 1]
proba_cem = best_model_cesm_lr.predict_proba(X_test[X_train_cesm_lr.columns])[:, 1]
proba_low = best_model_low_original_xgb.predict_proba(X_test[X_train_low_original_xgb.columns])[:, 1]
proba_re = best_model_re_logarithm_svm.predict_proba(X_test[X_train_re_logarithm_svm.columns])[:, 1]
proba_sclf_train = sclf_rf.predict_proba(X_train)[:, 1]

result = {
    'y_test': y_test,
    'cem': proba_cem,
    'low_energy': proba_low,
    'recombined': proba_re,
    'stacking_model': proba_sclf
}
result = pd.DataFrame(result)
result.to_csv('./data/result/result-0610.csv')

result_train = {'y_train': y_train, 'stacking_model': proba_sclf_train}
result_train = pd.DataFrame(result_train)
# result_train.to_csv('./data/result/result-train-0530.csv')

# %%
mrmc_predict = pd.concat(
    [result_train[['stacking_model']], result[['stacking_model']]],
    axis=0,
    sort=True).sort_index()
path = pd.read_csv(
    '/home/bowen/git/coding/python/CEM-lymphnode/data/path_case0420.csv')

mrmc = pd.concat([patient, mrmc_predict], axis=1)
mrmc = pd.merge(path, mrmc, on=['patientID', 'ImageLaterality',
                                'pathology']).drop(['Unnamed: 0'], 1)
mrmc.to_csv('./data/result/mrmc_0603.csv')

# %%
####################固化模型
# 保存该对象-joblib

# 压缩存储
joblib.dump(best_model_cesm_lr, './data/model/model_cesm_lr_05226.pkl', compress=True)
# joblib.dump(best_model_re_logarithm_svm, './data/model/model_re_log_svm_0523.pkl', compress=True)
joblib.dump(best_model_low_original_xgb, './data/model/model_low_original_xgb_0527.pkl', compress=True)
# joblib.dump(sclf_xgb, './data/model/model_sclf_xgb_0523.pkl', compress=True)
# joblib.dump(sclf_dt, './data/model/model_sclf_dt_0523.pkl', compress=True)
joblib.dump(sclf_rf, './data/model/model_sclf_rf_0603.pkl', compress=True)
# %%
import shap

shap.initjs()

# %%
cesm_explainer = shap.KernelExplainer(best_model_cesm_lr.predict_proba, X_train_cesm_lr)
cesm_value = cesm_explainer.shap_values(X_test_cesm_lr)

# %%
shap.summary_plot(cesm_value[1], X_test_cesm_lr, feature_names=X_test_cesm_lr.columns, max_display=15,show=False)
# shap.summary_plot(cesm_value[1], X_test_cesm_lr, feature_names=X_test_cesm_lr.columns, plot_type="bar")

# %%
low_explainer = shap.KernelExplainer(best_model_low_original_xgb.predict_proba, X_train_low_original_xgb)
low_value = low_explainer.shap_values(X_test_low_original_xgb)

# %%

shap.summary_plot(low_value[1], X_test_low_original_xgb, feature_names=X_test_low_original_xgb.columns)
# shap.summary_plot(low_value[1], X_test_low_original_xgb, feature_names=X_test_low_original_xgb.columns, plot_type="bar")

# %%
re_explainer = shap.KernelExplainer(best_model_re_logarithm_svm.predict_proba, X_train_re_logarithm_svm)
re_value = re_explainer.shap_values(X_test_re_logarithm_svm)

# %%
shap.summary_plot(re_value[1], X_test_re_logarithm_svm, feature_names=X_test_re_logarithm_svm.columns)

# shap.summary_plot(re_value[1], X_test_re_logarithm_svm, feature_names=X_test_re_logarithm_svm.columns, plot_type="bar")

# %%
##############针对融合模型，输出top6的特征
sclf_explainer = shap.KernelExplainer(sclf_rf.predict_proba, X_train)

sclf_value = sclf_explainer.shap_values(X_test)
# sclf_value_train = sclf_explainer.shap_values(X_train)
# %%
shap.summary_plot(sclf_value[1], X_test, feature_names=X_test.columns)
shap.summary_plot(sclf_value[1], X_test, feature_names=X_test.columns, plot_type="bar")

# %%
############单独输出训练集
shap.summary_plot(sclf_value_train[1], X_train, feature_names=X_train.columns)
shap.summary_plot(sclf_value_train[1], X_train, feature_names=X_train.columns, plot_type="bar")

# %%
###### 将训练集和测试机一块输出
shap.summary_plot(
    pd.concat([pd.DataFrame(sclf_value[1]), pd.DataFrame(sclf_value_train[1])],axis=0).values, 
    pd.concat([X_test, X_train], axis=0),
    feature_names=X_test.columns)
# %%

feature_top6_sclf = pd.DataFrame()
feature_top6_sclf_test = pd.DataFrame()
feature_top6_sclf_train = pd.DataFrame()

for i in range(0,len(y_train)):
    ind = i

    sclf_features_train = pd.DataFrame()
    sclf_features_train['feature'] = X_train.columns
    sclf_features_train['feature_value'] = X_train[X_train.columns].iloc[ind].values
    sclf_features_train['shap_value'] = sclf_value_train[1][ind,:]
    sclf_features_train['shap_abs'] = abs(sclf_value_train[1][ind,:])
    sclf_features_train = sclf_features_train.sort_values(by=['shap_abs'], ascending=False,ignore_index=True)

    dd =pd.concat(
        [sclf_features_train.iloc[0:1,:].reset_index(drop=True),
        sclf_features_train.iloc[1:2,:].reset_index(drop=True), 
        sclf_features_train.iloc[2:3,:].reset_index(drop=True),
        sclf_features_train.iloc[3:4,:].reset_index(drop=True),
        sclf_features_train.iloc[4:5,:].reset_index(drop=True),
        sclf_features_train.iloc[5:6,:].reset_index(drop=True),
        ], axis=1)
        
    feature_top6_sclf_train = pd.concat([feature_top6_sclf_train, dd],axis=0, ignore_index=True)

for i in range(0,len(y_test)):
    ind = i

    sclf_features_test = pd.DataFrame()
    sclf_features_test['feature'] = X_test.columns
    sclf_features_test['feature_value'] = X_test[X_test.columns].iloc[ind].values
    sclf_features_test['shap_value'] = sclf_value[1][ind,:]
    sclf_features_test['shap_abs'] = abs(sclf_value[1][ind,:])
    sclf_features_test = sclf_features_test.sort_values(by=['shap_abs'], ascending=False,ignore_index=True)

    ee =pd.concat(
        [sclf_features_test.iloc[0:1,:].reset_index(drop=True),
        sclf_features_test.iloc[1:2,:].reset_index(drop=True), 
        sclf_features_test.iloc[2:3,:].reset_index(drop=True),
        sclf_features_test.iloc[3:4,:].reset_index(drop=True),
        sclf_features_test.iloc[4:5,:].reset_index(drop=True),
        sclf_features_test.iloc[5:6,:].reset_index(drop=True),
        ], axis=1)
        
    feature_top6_sclf_test = pd.concat([feature_top6_sclf_test, ee],axis=0, ignore_index=True)

#%%
feature_top6_sclf_test.index = y_test.index
feature_top6_sclf_train.index = y_train.index
feature_top6_sclf = pd.concat([feature_top6_sclf_train, feature_top6_sclf_test], axis=0)
# %%
# %%
mrmc_new = pd.concat([mrmc, feature_top6_sclf],axis=1)
mrmc_new.to_csv('./data/result/mrmc_sclftop6_0531.csv')
# %%


# %%
####  单独case的可视化
num = 0
shap.force_plot(sclf_explainer.expected_value[1], sclf_value[1][num,:], X_test.iloc[num,:])
# %%
explainer = shap.Explainer(best_model_cesm_lr, X_train_cesm_lr)
shap_values = explainer(X_train_cesm_lr)

# %%
shap.plots.bar(shap_values[10])
shap.plots.beeswarm(shap_values)