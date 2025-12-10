"""
Utilities Functions for Model Training and Evaluation
======================================================

Module chứa các hàm tiện ích để:
- Chia dữ liệu train/test với stratification
- Đánh giá và tối ưu hóa các mô hình ML (Decision Tree, Random Forest, XGBoost)
- Tìm kiếm siêu tham số tối ưu bằng Bayesian Optimization
- Tính toán các metrics đánh giá (precision, recall, f1, accuracy, confusion matrix)

Requirements:
    - pandas
    - numpy
    - scikit-learn
    - xgboost
    - bayesian-optimization (pip install bayesian-optimization)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from bayes_opt import BayesianOptimization


def split_training_testing(database, test_size, random):
    """
    Chia dữ liệu thành tập train và test với stratification.
    
    Args:
        database: pandas DataFrame chứa dữ liệu đầy đủ
        test_size: tỷ lệ dữ liệu test (0.0 - 1.0)
        random: random seed để đảm bảo reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
            - X_train, X_test: features của train/test set
            - y_train, y_test: labels (Malicious, Package Repository, Package Name)
            - feature_names: danh sách tên các features
    """
    # Lọc các cột features (loại bỏ cột không phải feature)
    f = [c for c in database.columns if c not in [
        'Malicious', 'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 
        'Package Repository', 'Package Name'
    ]]
    
    # Features matrix
    X = database[f].iloc[:, :].values
   
    # Target info (Malicious, Package Repository, Package Name)
    y = database.loc[:, ['Malicious', 'Package Repository', 'Package Name']].values
 
    # Stratification based on benign/malicious và repository origin ratio
    # Stratify theo y[:,0:2] = [Malicious, Package Repository]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        shuffle=True, 
        stratify=y[:, 0:2],  # stratify by malicious status và repo origin
        random_state=random
    )

    return (X_train, X_test, y_train, y_test, f)


def get_best_hyperparams(hyperparams_list):
    """
    Tìm bộ hyperparameters tốt nhất dựa trên precision cao nhất.
    
    Args:
        hyperparams_list: list of dict, mỗi dict có format:
            {'precision': float, 'hyperparams': dict}
    
    Returns:
        dict: bộ hyperparameters có precision cao nhất
    """
    max_prec = hyperparams_list[0]['precision']
    final_hyperparam_set = hyperparams_list[0]['hyperparams']
    
    for e in hyperparams_list:
        if e['precision'] > max_prec:
            max_prec = e['precision']
            final_hyperparam_set = e['hyperparams']
    
    return final_hyperparam_set


# ============================================================================
# XGBOOST CLASSIFIER
# ============================================================================

def evaluation_NPM_Pypi_xgb(database): 
    """
    Đánh giá hiệu suất của XGBoost classifier trên dữ liệu NPM/PyPI.
    
    Pipeline:
        1. Loại bỏ các cột Unnamed
        2. Chuyển đổi Package Repository sang mã số (NPM=1, PyPI=2)
        3. Thực hiện 10 lần cross-validation với random seeds khác nhau
        4. Mỗi lần: tìm hyperparams tối ưu, train, test, tính metrics
        5. Tổng hợp kết quả: mean và std của các metrics
    
    Args:
        database: pandas DataFrame với cột 'Package Repository', 'Malicious', features
    
    Returns:
        tuple: (result_df, best_hyperparams)
            - result_df: DataFrame chứa mean và std của các metrics
            - best_hyperparams: dict chứa bộ hyperparameters tốt nhất
    """
    # Làm sạch dữ liệu
    database = database.loc[:, ~database.columns.str.contains('^Unnamed')]
    database['Package Repository'] = np.where(database['Package Repository'] == "NPM", 1, 2)

    # Dict để lưu các bộ hyperparams và precision tương ứng
    hyperpar_list = []
    
    # Danh sách các metrics cần đánh giá
    evaluation = [
        'precision', 'recall', 'f1', 'accuracy',
        'false positive', 'false negative', 'true negative', 'true positive',
        'precision_npm', 'recall_npm', 'f1_npm', 'acc_npm',
        'precision_pypi', 'recall_pypi', 'f1_pypi', 'acc_pypi'
    ]
    
    # Lấy danh sách features
    f = [c for c in database.columns if c not in [
        'Malicious', 'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 
        'Package Repository', 'Package Name'
    ]]
    
    # Khởi tạo DataFrame để lưu kết quả đánh giá
    eval = pd.DataFrame(data=None, index=[y for x in [f, evaluation] for y in x])
    
    # 10 random seeds khác nhau để cross-validation
    random_split = [123, 333, 567, 999, 876, 371, 459, 111, 902, 724]
    
    for i in range(0, len(random_split)):
        print(f"\n[*] Cross-validation iteration {i+1}/{len(random_split)} (seed={random_split[i]})")
        
        # Chia dữ liệu train/test (loại bỏ repository=3 nếu có)
        split_ = split_training_testing(
            database[database['Package Repository'] != 3], 
            test_size=0.2, 
            random=random_split[i]
        )
        
        # Tối ưu hóa hyperparameters bằng Bayesian Optimization trên tập train
        train_rf_ = grid_xgb_py(split_[0], split_[2])
        
        # Fit model với hyperparameters tốt nhất
        classifier = xgb.XGBClassifier(
            random_state=123,
            n_estimators=train_rf_['n_estimators'],
            max_depth=train_rf_['max_depth'],
            gamma=train_rf_['gamma'],
            eta=train_rf_['eta'],
            colsample_bytree=train_rf_['colsample_bytree'],
            min_child_weight=train_rf_['min_child_weight']
        )
        classifier.fit(split_[0], split_[2][:, 0].astype('int'))
        
        # Dự đoán trên tập test
        y_pred_test_ = classifier.predict(split_[1])
        
        # Lưu precision và hyperparams
        hyperpar_list.append({
            'precision': round(precision_score(split_[3][:, 0].astype('int'), y_pred_test_) * 100, 2),
            'hyperparams': train_rf_
        })
        
        # Tính các metrics
        precision = np.append(
            classifier.feature_importances_,
            round(precision_score(split_[3][:, 0].astype('int'), y_pred_test_) * 100, 2)
        )
        recall = np.append(
            precision,
            round(recall_score(split_[3][:, 0].astype('int'), y_pred_test_) * 100, 2)
        )
        f1 = np.append(
            recall,
            round(f1_score(split_[3][:, 0].astype('int'), y_pred_test_) * 100, 2)
        )
        acc = np.append(
            f1,
            round(accuracy_score(split_[3][:, 0].astype('int'), y_pred_test_) * 100, 2)
        )
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(split_[3][:, 0].astype('int'), y_pred_test_).ravel()
        false_positive = np.append(acc, fp)
        false_negative = np.append(false_positive, fn)
        true_negative = np.append(false_negative, tn)
        true_positive = np.append(true_negative, tp)
        
        # Đánh giá theo từng repository (NPM vs PyPI)
        repository = np.concatenate((
            split_[3][:, 0].astype('int').reshape(len(split_[3][:, 0].astype('int')), 1), 
            y_pred_test_.reshape(len(y_pred_test_), 1),
            split_[3][:, 1].astype('int').reshape(len(split_[3][:, 1].astype('int')), 1)
        ), axis=1, out=None)
        
        npm = repository[repository[:, 2] == 1]
        pypi = repository[repository[:, 2] == 2]
        
        # Metrics cho NPM
        precision_npm = np.append(true_positive, round(precision_score(npm[:, 0], npm[:, 1]) * 100, 2))
        recall_npm = np.append(precision_npm, round(recall_score(npm[:, 0], npm[:, 1]) * 100, 2))
        f1_npm = np.append(recall_npm, round(f1_score(npm[:, 0], npm[:, 1]) * 100, 2))
        acc_npm = np.append(f1_npm, round(accuracy_score(npm[:, 0], npm[:, 1]) * 100, 2))
        
        # Metrics cho PyPI
        precision_pypi = np.append(acc_npm, round(precision_score(pypi[:, 0], pypi[:, 1]) * 100, 2))
        recall_pypi = np.append(precision_pypi, round(recall_score(pypi[:, 0], pypi[:, 1]) * 100, 2))
        f1_pypi = np.append(recall_pypi, round(f1_score(pypi[:, 0], pypi[:, 1]) * 100, 2))
        metrics = np.append(f1_pypi, round(accuracy_score(pypi[:, 0], pypi[:, 1]) * 100, 2))
        
        # Lưu metrics vào DataFrame
        eval[i] = metrics.tolist()
    
    # Thay thế 0 bằng NaN để tính mean/std chính xác hơn
    eval = eval.replace(0, np.nan)
    mean = eval.mean(axis=1)
    std = eval.std(axis=1)
    result = pd.concat([mean, std], axis=1)
    
    return (result, get_best_hyperparams(hyperpar_list))


def grid_xgb_py(regressors, labels):
    """
    Tìm kiếm hyperparameters tối ưu cho XGBoost bằng Bayesian Optimization.
    
    Args:
        regressors: features matrix (X_train)
        labels: target array có shape (n, 3) với [Malicious, Repository, Package Name]
    
    Returns:
        dict: bộ hyperparameters tối ưu cho XGBoost
    """
    # Hàm mục tiêu để maximize (test precision)
    def xgb_cl_bo(max_depth, n_estimators, colsample_bytree, eta, gamma, min_child_weight):
        params_xgb = {
            'max_depth': int(max_depth),
            'n_estimators': int(n_estimators),
            'colsample_bytree': colsample_bytree,
            'min_child_weight': int(min_child_weight),
            'eta': eta,
            'gamma': gamma
        }
        
        classifier = xgb.XGBClassifier(
            random_state=123,
            n_estimators=params_xgb['n_estimators'],
            max_depth=params_xgb['max_depth'],
            gamma=params_xgb['gamma'],
            eta=params_xgb['eta'],
            colsample_bytree=params_xgb['colsample_bytree'],
            min_child_weight=params_xgb['min_child_weight']
        )
        
        # Cross-validation với 5 folds
        scoring = {'rec': 'recall', 'prec': 'precision'}
        scores = cross_validate(
            classifier, 
            regressors, 
            labels[:, 0].astype('int'), 
            scoring=scoring,
            cv=5, 
            return_train_score=True, 
            n_jobs=-1
        )
        
        print(f'  recall: {round(scores["test_rec"].mean(), 2)}, precision_train: {round(scores["train_prec"].mean(), 2)}')
        
        target = scores['test_prec'].mean()
        return target
    
    # Không gian tìm kiếm hyperparameters
    params_xgb = {
        'max_depth': (2, 4),
        'n_estimators': (64, 256), 
        'min_child_weight': (8, 16), 
        'gamma': (0.6, 1.2),
        'eta': (0.08, 0.16),
        'colsample_bytree': (0.1, 0.3)
    }
    
    # Bayesian Optimization
    xgb_bo = BayesianOptimization(xgb_cl_bo, params_xgb, random_state=111, verbose=1)
    xgb_bo.maximize(init_points=25, n_iter=5)
    
    print(f"\n[*] Best parameters found: {xgb_bo.max}")
    
    # Trích xuất hyperparameters tốt nhất
    params_xgb_best = {
        'n_estimators': int(xgb_bo.max["params"]["n_estimators"]),
        'max_depth': int(xgb_bo.max["params"]["max_depth"]),
        'min_child_weight': int(xgb_bo.max["params"]["min_child_weight"]),
        'eta': xgb_bo.max['params']['eta'],
        'gamma': xgb_bo.max['params']['gamma'],
        'colsample_bytree': xgb_bo.max['params']['colsample_bytree']
    }
    
    return params_xgb_best


# ============================================================================
# DECISION TREE CLASSIFIER
# ============================================================================

def evaluation_decision_tree(database):
    """
    Đánh giá hiệu suất của Decision Tree classifier.
    
    Args:
        database: pandas DataFrame với dữ liệu đầy đủ
    
    Returns:
        tuple: (result_df, best_hyperparams)
    """
    # Làm sạch dữ liệu
    database = database.loc[:, ~database.columns.str.contains('^Unnamed')]
    database['Package Repository'] = np.where(database['Package Repository'] == "NPM", 1, 2)
    
    hyperpar_list = []
    
    evaluation = [
        'precision', 'recall', 'f1', 'accuracy',
        'false positive', 'false negative', 'true negative', 'true positive',
        'precision_npm', 'recall_npm', 'f1_npm', 'acc_npm',
        'precision_pypi', 'recall_pypi', 'f1_pypi', 'acc_pypi'
    ]
    
    f = [c for c in database.columns if c not in [
        'Malicious', 'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1',
        'Package Repository', 'Package Name'
    ]]
    
    eval = pd.DataFrame(data=None, index=[y for x in [f, evaluation] for y in x])
    random_split = [123, 333, 567, 999, 876, 371, 459, 111, 902, 724]
    
    for i in range(0, 10):
        print(f"\n[*] Cross-validation iteration {i+1}/10 (seed={random_split[i]})")
        
        split_ = split_training_testing(
            database[database['Package Repository'] != 3], 
            test_size=0.2, 
            random=random_split[i]
        )
        
        # Tối ưu hóa hyperparameters
        train_rf_ = grid_tree(split_[0], split_[2])
        
        # Fit model
        classifier = DecisionTreeClassifier(
            random_state=123,
            criterion=train_rf_['criterion'],
            max_depth=train_rf_['max_depth'],
            max_features=train_rf_['max_features'],
            min_samples_leaf=train_rf_['min_sample_leaf'],
            min_samples_split=train_rf_['min_sample_split']
        )
        classifier.fit(split_[0], split_[2][:, 0].astype('int'))
        
        # Predict
        y_pred_test_ = classifier.predict(split_[1])
        
        hyperpar_list.append({
            'precision': round(precision_score(split_[3][:, 0].astype('int'), y_pred_test_) * 100, 2),
            'hyperparams': train_rf_
        })
        
        # Tính metrics (tương tự XGBoost)
        precision = np.append(
            classifier.feature_importances_,
            round(precision_score(split_[3][:, 0].astype('int'), y_pred_test_) * 100, 2)
        )
        recall = np.append(precision, round(recall_score(split_[3][:, 0].astype('int'), y_pred_test_) * 100, 2))
        f1 = np.append(recall, round(f1_score(split_[3][:, 0].astype('int'), y_pred_test_) * 100, 2))
        acc = np.append(f1, round(accuracy_score(split_[3][:, 0].astype('int'), y_pred_test_) * 100, 2))
        
        tn, fp, fn, tp = confusion_matrix(split_[3][:, 0].astype('int'), y_pred_test_).ravel()
        false_positive = np.append(acc, fp)
        false_negative = np.append(false_positive, fn)
        true_negative = np.append(false_negative, tn)
        true_positive = np.append(true_negative, tp)
        
        repository = np.concatenate((
            split_[3][:, 0].astype('int').reshape(len(split_[3][:, 0].astype('int')), 1), 
            y_pred_test_.reshape(len(y_pred_test_), 1),
            split_[3][:, 1].astype('int').reshape(len(split_[3][:, 1].astype('int')), 1)
        ), axis=1, out=None)
        
        npm = repository[repository[:, 2] == 1]
        pypi = repository[repository[:, 2] == 2]
        
        precision_npm = np.append(true_positive, round(precision_score(npm[:, 0], npm[:, 1]) * 100, 2))
        recall_npm = np.append(precision_npm, round(recall_score(npm[:, 0], npm[:, 1]) * 100, 2))
        f1_npm = np.append(recall_npm, round(f1_score(npm[:, 0], npm[:, 1]) * 100, 2))
        acc_npm = np.append(f1_npm, round(accuracy_score(npm[:, 0], npm[:, 1]) * 100, 2))
        
        precision_pypi = np.append(acc_npm, round(precision_score(pypi[:, 0], pypi[:, 1]) * 100, 2))
        recall_pypi = np.append(precision_pypi, round(recall_score(pypi[:, 0], pypi[:, 1]) * 100, 2))
        f1_pypi = np.append(recall_pypi, round(f1_score(pypi[:, 0], pypi[:, 1]) * 100, 2))
        metrics = np.append(f1_pypi, round(accuracy_score(pypi[:, 0], pypi[:, 1]) * 100, 2))
        
        eval[i] = metrics.tolist()
    
    eval = eval.replace(0, np.nan)
    mean = eval.mean(axis=1)
    std = eval.std(axis=1)
    result = pd.concat([mean, std], axis=1)
    
    return (result, get_best_hyperparams(hyperpar_list))


def grid_tree(regressors, labels):
    """
    Tìm kiếm hyperparameters tối ưu cho Decision Tree.
    
    Args:
        regressors: features matrix
        labels: target array
    
    Returns:
        dict: bộ hyperparameters tối ưu
    """
    criteria = ['gini', 'entropy', 'log_loss']
    number_features = ['sqrt', 'log2', None]
    
    def tree_cl_bo(max_depth, max_features, criterion, min_sample_leaf, min_sample_split):
        params_tree = {
            'max_depth': int(max_depth),
            'max_features': number_features[int(max_features)],
            'criterion': criteria[int(criterion)],
            'min_sample_leaf': int(min_sample_leaf),
            'min_sample_split': int(min_sample_split)
        }
        
        classifier = DecisionTreeClassifier(
            random_state=123,
            criterion=params_tree['criterion'],
            max_depth=params_tree['max_depth'],
            min_samples_leaf=params_tree['min_sample_leaf'],
            max_features=params_tree['max_features'],
            min_samples_split=params_tree['min_sample_split']
        )
        
        scoring = {'rec': 'recall', 'prec': 'precision'}
        scores = cross_validate(
            classifier, 
            regressors, 
            labels[:, 0].astype('int'), 
            scoring=scoring,
            cv=5, 
            return_train_score=True, 
            n_jobs=-1
        )
        
        print(f'  recall: {round(scores["test_rec"].mean(), 2)}, precision_train: {round(scores["train_prec"].mean(), 2)}')
        
        target = scores['test_prec'].mean()
        return target
    
    params_tree = {
        'max_depth': (2, 4),
        'max_features': (0, 2.99), 
        'criterion': (0, 2.99),
        'min_sample_leaf': (4, 8),
        'min_sample_split': (6, 16)
    }
    
    tree_bo = BayesianOptimization(tree_cl_bo, params_tree, random_state=111)
    tree_bo.maximize(init_points=25, n_iter=5)
    
    print(f"\n[*] Best parameters found: {tree_bo.max}")
    
    params_tree_best = {
        'max_features': number_features[int(tree_bo.max["params"]["max_features"])],
        'max_depth': int(tree_bo.max["params"]["max_depth"]),
        'criterion': criteria[int(tree_bo.max["params"]["criterion"])],
        'min_sample_leaf': int(tree_bo.max['params']['min_sample_leaf']),
        'min_sample_split': int(tree_bo.max['params']['min_sample_split'])
    }
    
    return params_tree_best


# ============================================================================
# RANDOM FOREST CLASSIFIER
# ============================================================================

def evaluation_random_forest(database):
    """
    Đánh giá hiệu suất của Random Forest classifier.
    
    Args:
        database: pandas DataFrame với dữ liệu đầy đủ
    
    Returns:
        tuple: (result_df, best_hyperparams)
    """
    database = database.loc[:, ~database.columns.str.contains('^Unnamed')]
    database['Package Repository'] = np.where(database['Package Repository'] == "NPM", 1, 2)

    hyperpar_list = [] 
    
    evaluation = [
        'precision', 'recall', 'f1', 'accuracy',
        'false positive', 'false negative', 'true negative', 'true positive',
        'precision_npm', 'recall_npm', 'f1_npm', 'acc_npm',
        'precision_pypi', 'recall_pypi', 'f1_pypi', 'acc_pypi'
    ]
    
    f = [c for c in database.columns if c not in [
        'Malicious', 'Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 
        'Package Repository', 'Package Name'
    ]]
    
    eval = pd.DataFrame(data=None, index=[y for x in [f, evaluation] for y in x])
    random_split = [123, 333, 567, 999, 876, 371, 459, 111, 902, 724]
    
    for i in range(0, 10):
        print(f"\n[*] Cross-validation iteration {i+1}/10 (seed={random_split[i]})")
        
        split_ = split_training_testing(
            database[database['Package Repository'] != 3], 
            test_size=0.2, 
            random=random_split[i]
        )
        
        train_rf_ = grid_rf(split_[0], split_[2])
        
        classifier = RandomForestClassifier(
            random_state=123,
            criterion=train_rf_['criterion'],
            n_estimators=train_rf_['n_estimators'],
            max_depth=train_rf_['max_depth'],
            max_features=train_rf_['max_features'],
            min_samples_leaf=train_rf_['min_sample_leaf'],
            min_samples_split=train_rf_['min_sample_split'],
            max_samples=train_rf_['max_samples']
        )
        classifier.fit(split_[0], split_[2][:, 0].astype('int'))
        
        y_pred_test_ = classifier.predict(split_[1])

        hyperpar_list.append({
            'precision': round(precision_score(split_[3][:, 0].astype('int'), y_pred_test_) * 100, 2),
            'hyperparams': train_rf_
        })
        
        precision = np.append(
            classifier.feature_importances_,
            round(precision_score(split_[3][:, 0].astype('int'), y_pred_test_) * 100, 2)
        )
        recall = np.append(precision, round(recall_score(split_[3][:, 0].astype('int'), y_pred_test_) * 100, 2))
        f1 = np.append(recall, round(f1_score(split_[3][:, 0].astype('int'), y_pred_test_) * 100, 2))
        acc = np.append(f1, round(accuracy_score(split_[3][:, 0].astype('int'), y_pred_test_) * 100, 2))
        
        tn, fp, fn, tp = confusion_matrix(split_[3][:, 0].astype('int'), y_pred_test_).ravel()
        false_positive = np.append(acc, fp)
        false_negative = np.append(false_positive, fn)
        true_negative = np.append(false_negative, tn)
        true_positive = np.append(true_negative, tp)
        
        repository = np.concatenate((
            split_[3][:, 0].astype('int').reshape(len(split_[3][:, 0].astype('int')), 1), 
            y_pred_test_.reshape(len(y_pred_test_), 1),
            split_[3][:, 1].astype('int').reshape(len(split_[3][:, 1].astype('int')), 1)
        ), axis=1, out=None)
        
        npm = repository[repository[:, 2] == 1]
        pypi = repository[repository[:, 2] == 2]
        
        precision_npm = np.append(true_positive, round(precision_score(npm[:, 0], npm[:, 1]) * 100, 2))
        recall_npm = np.append(precision_npm, round(recall_score(npm[:, 0], npm[:, 1]) * 100, 2))
        f1_npm = np.append(recall_npm, round(f1_score(npm[:, 0], npm[:, 1]) * 100, 2))
        acc_npm = np.append(f1_npm, round(accuracy_score(npm[:, 0], npm[:, 1]) * 100, 2))
        
        precision_pypi = np.append(acc_npm, round(precision_score(pypi[:, 0], pypi[:, 1]) * 100, 2))
        recall_pypi = np.append(precision_pypi, round(recall_score(pypi[:, 0], pypi[:, 1]) * 100, 2))
        f1_pypi = np.append(recall_pypi, round(f1_score(pypi[:, 0], pypi[:, 1]) * 100, 2))
        metrics = np.append(f1_pypi, round(accuracy_score(pypi[:, 0], pypi[:, 1]) * 100, 2))
        
        eval[i] = metrics.tolist()
    
    eval = eval.replace(0, np.nan)
    mean = eval.mean(axis=1)
    std = eval.std(axis=1)
    result = pd.concat([mean, std], axis=1)
    
    return (result, get_best_hyperparams(hyperpar_list))


def grid_rf(regressors, labels):
    """
    Tìm kiếm hyperparameters tối ưu cho Random Forest.
    
    Args:
        regressors: features matrix
        labels: target array
    
    Returns:
        dict: bộ hyperparameters tối ưu
    """
    criteria = ['gini', 'entropy', 'log_loss']
    number_features = ['sqrt', 'log2', None]
    
    def rf_cl_bo(max_depth, max_features, n_estimators, criterion, min_sample_leaf, min_sample_split, max_samples):
        params_rf = {
            'max_depth': int(max_depth),
            'max_features': number_features[int(max_features)],
            'criterion': criteria[int(criterion)],
            'n_estimators': int(n_estimators),
            'min_sample_leaf': int(min_sample_leaf),
            'min_sample_split': int(min_sample_split),
            'max_samples': max_samples
        }
        
        classifier = RandomForestClassifier(
            random_state=123,
            criterion=params_rf['criterion'],
            n_estimators=params_rf['n_estimators'],
            max_depth=params_rf['max_depth'],
            min_samples_leaf=params_rf['min_sample_leaf'],
            max_features=params_rf['max_features'],
            max_samples=params_rf['max_samples'],
            min_samples_split=params_rf['min_sample_split']
        )
        
        scoring = {'rec': 'recall', 'prec': 'precision'}
        scores = cross_validate(
            classifier, 
            regressors, 
            labels[:, 0].astype('int'), 
            scoring=scoring,
            cv=5, 
            return_train_score=True, 
            n_jobs=-1
        )
        
        print(f'  recall: {round(scores["test_rec"].mean(), 2)}, precision_train: {round(scores["train_prec"].mean(), 2)}')
        
        target = scores['test_prec'].mean()
        return target
    
    params_rf = {
        'max_depth': (2, 4),
        'max_features': (0, 2.99),
        'n_estimators': (64, 256), 
        'criterion': (0, 2.99),
        'min_sample_leaf': (4, 8),
        'min_sample_split': (6, 16),
        'max_samples': (0.1, 1)
    }
    
    rf_bo = BayesianOptimization(rf_cl_bo, params_rf, random_state=111)
    rf_bo.maximize(init_points=25, n_iter=5)
    
    print(f"\n[*] Best parameters found: {rf_bo.max}")
    
    params_rf_best = {
        'n_estimators': int(rf_bo.max["params"]["n_estimators"]),
        'max_features': number_features[int(rf_bo.max["params"]["max_features"])],
        'max_depth': int(rf_bo.max["params"]["max_depth"]),
        'criterion': criteria[int(rf_bo.max["params"]["criterion"])],
        'min_sample_leaf': int(rf_bo.max['params']['min_sample_leaf']),
        'min_sample_split': int(rf_bo.max['params']['min_sample_split']),
        'max_samples': rf_bo.max['params']['max_samples']
    }
    
    return params_rf_best
