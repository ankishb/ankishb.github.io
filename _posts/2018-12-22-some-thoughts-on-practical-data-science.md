---
layout: post
title: "some-thoughts-on-practical-data-science"
categories: data-science
author: "Ankish Bansal"
---

## Control Overfitting

There are in general two ways that you can control overfitting in XGBoost:

- The first way is to directly control model complexity.
    This includes `max_depth`, `min_child_weight` and `gamma`.
- The second way is to add randomness to make training robust to noise.
    This includes `subsample` and `colsample_bytree`.
    You can also reduce stepsize `eta`. Remember to increase `num_round` when you do so.

## Handle Imbalanced Dataset

- Feature selection with weighted samples approach

- If you care only about the overall performance metric (AUC) of your prediction
    Balance the positive and negative weights via `scale_pos_weight`
    Use AUC for evaluation
- If you care about predicting the right probability
    In such a case, you cannot re-balance the dataset
    Set parameter `max_delta_step` to a finite number (say 1) to help convergence




## Booster Parameters

1. min_child_weight
  - **Min samples to further divide from that node of tree**
  
  - "minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning.

  - Intuitively, this is the minimum number of samples that a node can represent in order to be split further. If there are fewer than min_child_weight samples at that node, the node becomes a leaf and is no longer split. This can help reduce the model complexity and prevent overfitting.

2. max_depth 
  - 3 - 10


3. max_leaf_nodes

    -The maximum number of terminal nodes or leaves in a tree

4. gamma [default=0]

    A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.
    
5. max_delta_step [default=0]

    In maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0, it means there is no constraint   
    Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced.


6. subsample [default=1]

    Same as the subsample of GBM. Denotes the fraction of observations to be randomly samples for each tree.
    Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.
    Typical values: 0.5-1

   
   
7. colsample_bytree [default=1]

    Similar to max_features in GBM. Denotes the fraction of columns to be randomly samples for each tree.
    Typical values: 0.5-1

8. colsample_bylevel [default=1] ==>> Doesn't need

    Denotes the subsample ratio of columns for each split, in each level.

9. alpha [default=0]

    - L1 regularization term on weight (analogous to Lasso regression)
    - useful for high-dim data

10. scale_pos_weight [default=1]

    A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence


11. lambda [default=1]

    L2 regularization term on weights (analogous to Ridge regression)



objective [default=reg:linear]

    reg:linear: linear regression
    reg:logistic: logistic regression
    binary:logistic: logistic regression for binary classification, output probability
    binary:hinge: hinge loss for binary classification. This makes predictions of 0 or 1, rather than producing probabilities.
    count:poisson –poisson regression for count data, output mean of poisson distribution
        max_delta_step is set to 0.7 by default in poisson regression (used to safeguard optimization)
    multi:softmax: set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
    rank:pairwise: Use LambdaMART to perform pairwise ranking where the pairwise loss is minimized



eval_metric [ default according to objective ]

    The metric to be used for validation data.
    The default values are rmse for regression and error for classification.
    Typical values are:
        rmse – root mean square error
        mae – mean absolute error
        logloss – negative log-likelihood
        error – Binary classification error rate (0.5 threshold)
        merror – Multiclass classification error rate
        mlogloss – Multiclass logloss
        auc: Area under the curve


## import imp lib

```python
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
```



## XGBModel(XGBModelBase):
```python
# pylint: disable=too-many-arguments, too-many-instance-attributes, invalid-name
    """Implementation of the Scikit-Learn API for XGBoost.
    Parameters
    ----------
    max_depth : int
        Maximum tree depth for base learners.
    learning_rate : float
        Boosting learning rate (xgb's "eta")
    n_estimators : int
        Number of boosted trees to fit.
    silent : boolean
        Whether to print messages while running boosting.
    objective : string or callable
        Specify the learning task and the corresponding learning objective or
        a custom objective function to be used (see note below).
    booster: string
        Specify which booster to use: gbtree, gblinear or dart.
    nthread : int
        Number of parallel threads used to run xgboost.  (Deprecated, please use ``n_jobs``)
    n_jobs : int
        Number of parallel threads used to run xgboost.  (replaces ``nthread``)
    gamma : float
        Minimum loss reduction required to make a further partition on a leaf node of the tree.
    min_child_weight : int
        Minimum sum of instance weight(hessian) needed in a child.
    max_delta_step : int
        Maximum delta step we allow each tree's weight estimation to be.
    subsample : float
        Subsample ratio of the training instance.
    colsample_bytree : float
        Subsample ratio of columns when constructing each tree.
    colsample_bylevel : float
        Subsample ratio of columns for each split, in each level.
    reg_alpha : float (xgb's alpha)
        L1 regularization term on weights
    reg_lambda : float (xgb's lambda)
        L2 regularization term on weights
    scale_pos_weight : float
        Balancing of positive and negative weights.
    base_score:
        The initial prediction score of all instances, global bias.
    seed : int
        Random number seed.  (Deprecated, please use random_state)
    random_state : int
        Random number seed.  (replaces seed)
    missing : float, optional
        Value in the data which needs to be present as a missing value. If
        None, defaults to np.nan.
    importance_type: string, default "gain"
        The feature importance type for the feature_importances_ property: either "gain",
        "weight", "cover", "total_gain" or "total_cover".
    \*\*kwargs : dict, optional
        Keyword arguments for XGBoost Booster object.  Full documentation of parameters can
        be found here: https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst.
        Attempting to set a parameter via the constructor args and \*\*kwargs dict simultaneously
        will result in a TypeError.
        .. note:: \*\*kwargs unsupported by scikit-learn
            \*\*kwargs is unsupported by scikit-learn.  We do not guarantee that parameters
            passed via this argument will interact properly with scikit-learn.
    Note
    ----
    A custom objective function can be provided for the ``objective``
    parameter. In this case, it should have the signature
    ``objective(y_true, y_pred) -> grad, hess``:
    y_true: array_like of shape [n_samples]
        The target values
    y_pred: array_like of shape [n_samples]
        The predicted values
    grad: array_like of shape [n_samples]
        The value of the gradient for each sample point.
    hess: array_like of shape [n_samples]
        The value of the second derivative for each sample point
    """

    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100,
                 silent=True, objective="reg:linear", booster='gbtree',
                 n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0,
                 subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, random_state=0, seed=None, missing=None,
importance_type="gain", **kwargs)
```





# Custom objective function

```python
#!/usr/bin/python
import numpy as np
import xgboost as xgb
###
# advanced: customized loss function
#
print('start running example to used customized objective function')

dtrain = xgb.DMatrix('../data/agaricus.txt.train')
dtest = xgb.DMatrix('../data/agaricus.txt.test')

# note: for customized objective function, we leave objective as default
# note: what we are getting is margin value in prediction
# you must know what you are doing
param = {'max_depth': 2, 'eta': 1, 'silent': 1}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 2

# user define objective function, given prediction, return gradient and second order gradient
# this is log likelihood loss
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess

# user defined evaluation function, return a pair metric_name, result
# NOTE: when you do customized loss function, the default prediction value is margin
# this may make builtin evaluation metric not function properly
# for example, we are doing logistic loss, the prediction is score before logistic transformation
# the builtin evaluation error assumes input is after logistic transformation
# Take this in mind when you use the customization, and maybe you need write customized evaluation function
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result. The metric name must not contain a colon (:) or a space
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'my-error', float(sum(labels != (preds > 0.0))) / len(labels)

# training with customized objective, we can also do step by step training
# simply look at xgboost.py's implementation of train
bst = xgb.train(param, dtrain, num_round, watchlist, obj=logregobj, feval=evalerror)

```













General Approach for Parameter Tuning

We will use an approach similar to that of GBM here. The various steps to be performed are:

    Choose a relatively high learning rate. Generally a learning rate of 0.1 works but somewhere between 0.05 to 0.3 should work for different problems. Determine the optimum number of trees for this learning rate. XGBoost has a very useful function called as “cv” which performs cross-validation at each boosting iteration and thus returns the optimum number of trees required.
    Tune tree-specific parameters ( max_depth, min_child_weight, gamma, subsample, colsample_bytree) for decided learning rate and number of trees. Note that we can choose different parameters to define a tree and I’ll take up an example here.
    Tune regularization parameters (lambda, alpha) for xgboost which can help reduce model complexity and enhance performance.
    Lower the learning rate and decide the optimal parameters .





#  Use XGBClassifier ==> That has CV method inbuilt

```python

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
```



# Step 1: Fix learning rate and number of estimators for tuning tree-based parameters

In order to decide on boosting parameters, we need to set some initial values of other parameters. Lets take the following values:

    max_depth = 5 : This should be between 3-10. I’ve started with 5 but you can choose a different number as well. 4-6 can be good starting points.
    min_child_weight = 1 : A smaller value is chosen because it is a highly imbalanced class problem and leaf nodes can have smaller size groups.
    gamma = 0 : A smaller value like 0.1-0.2 can also be chosen for starting. This will anyways be tuned later.
    subsample, colsample_bytree = 0.8 : This is a commonly used used start value. Typical values range between 0.5-0.9.
    scale_pos_weight = 1: Because of high class imbalance.

Please note that all the above are just initial estimates and will be tuned later. Lets take the default learning rate of 0.1 here and check the optimum number of trees using cv function of xgboost. The function defined above will do it for us.

```python
#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target, IDcol]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)
```


# Step 2: Tune max_depth and min_child_weight

```python
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])


modelfit(gsearch1.best_estimator_, train, predictors)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
```

# Step 3: Tune gamma


```python

param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

```

# Step 4: Tune subsample and colsample_bytree

The next step would be try different subsample and colsample_bytree values. Lets do this in 2 stages as well and take values 0.6,0.7,0.8,0.9 for both to start with.

```python
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

`To check more conjustedly, use 
param_test5 = {
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}
`

gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
```


# Step 5: Tuning Regularization Parameters

Next step is to apply regularization to reduce overfitting. Though many people don’t use this parameters much as gamma provides a substantial way of controlling complexity. But we should always try it. I’ll tune ‘reg_alpha’ value here and leave it upto you to try different values of ‘reg_lambda’.

```python
param_test6 = {
 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
 }
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch6.fit(train[predictors],train[target])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_
```
# Step 6: Reducing Learning Rate and repeat jon again

'''python
xgb4 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb4, train, predictors)
'''

##  As I mentioned in the end, techniques like feature engineering and blending have a much greater impact than parameter tuning. For instance, I generally do some parameter tuning and then run 10 different models on same parameters but different seeds. Averaging their results generally gives a good boost to the performance of the model.



# Startified sampling

```python
from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)

print(skf)  

for train_index, test_index in skf.split(X, y):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]

```

## eval_result

```python
Example

param_dist = {'objective':'binary:logistic', 'n_estimators':2}

clf = xgb.XGBModel(**param_dist)

clf.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric='logloss',
        verbose=True)

evals_result = clf.evals_result()

The variable evals_result will contain:

{'validation_0': {'logloss': ['0.604835', '0.531479']},
'validation_1': {'logloss': ['0.41965', '0.17686']}}


```







