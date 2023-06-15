import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

from data_prepare_func import convert_to_array


def pipe_setup(model_name):
    if model_name == 'Logistic Regression':
        pipe_lr = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression())])

        pipe_lr.set_params(
            classifier__C=0.01, classifier__multi_class='multinomial', classifier__solver='saga', classifier__random_state=42)

        return pipe_lr

    elif model_name == 'Neural Network':
        pipe_nn = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier())])

        pipe_nn.set_params(
            classifier__alpha=0.1, classifier__hidden_layer_sizes=(50, 50), classifier__random_state=42)

        return pipe_nn

    elif model_name == 'Random Forest':
        pipe_rf = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())])

        pipe_rf.set_params(
            classifier__bootstrap=False, classifier__class_weight='balanced', classifier__criterion='gini',
            classifier__max_depth=13, classifier__max_features='sqrt', classifier__max_leaf_nodes=200,
            classifier__n_estimators=75, classifier__random_state=42, classifier__warm_start=False)
        return pipe_rf

    elif model_name == 'Extra Trees Classifier':
        pipe_et = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', ExtraTreesClassifier())])

        pipe_et.set_params(
            classifier__bootstrap=False, classifier__class_weight='balanced_subsample', classifier__criterion='gini',
            classifier__max_depth=14, classifier__max_features='sqrt', classifier__max_leaf_nodes=200, classifier__n_estimators=75,
            classifier__random_state=42, classifier__warm_start=True)
        return pipe_et

    elif model_name == 'XGB Classifier':
        pipe_xg = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier())])

        pipe_xg.set_params(
            classifier__max_depth=4, classifier__learning_rate=0.001, classifier__gamma=0, classifier__reg_alpha=1,
            classifier__reg_lambda=1, classifier__colsample_bytree=0.5, classifier__subsample=0.7,
            classifier__tree_method='hist', classifier__n_estimators=140, classifier__eval_metric='merror',
            classifier__colsample_bylevel=0.5, classifier__colsample_bynode=0.5)

        return pipe_xg


def fit_model(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y)

    model.fit(x_train, y_train)

    train_yhat = model.predict(x_train)
    train_accuracy = accuracy_score(train_yhat, y_train)

    test_yhat = model.predict(x_test)
    test_accuracy = accuracy_score(test_yhat, y_test)

    cv_accuracy = cross_val_score(
        model, x_train, y_train, scoring='accuracy', cv=5)

    cm = confusion_matrix(y_test, test_yhat)

    return train_accuracy, test_accuracy, cv_accuracy, cm


if __name__ == '__main__':

    x_kit, y_kit = convert_to_array('data_fr_kittinan/', 28)
    x_diy, y_diy = convert_to_array('data_writing_diy/', 28)
    X = np.append(x_kit, x_diy, axis=0)
    y = np.append(y_kit, y_diy, axis=0)

    model_lst = ['XGB Classifier', 'Logistic Regression',
                 'Random Forest', 'Neural Network', 'Extra Trees Classifier']
    best_model = None
    best_accuracy = 0.0
    best_model_name = ""
    best_cm = ""
    model_accuracies = []

    for m in model_lst:
        clf = pipe_setup(m)
        train_ac, test_ac, cv_ac, cm = fit_model(clf, X, y)

        if train_ac > best_accuracy:
            best_model = clf
            best_accuracy = train_ac
            best_model_name = m
            best_cm = cm

        model_accuracies.append((m, train_ac, cv_ac))

    print(model_accuracies)
    print(model_accuracies[0])
    print(model_accuracies[0][0])
    print(model_accuracies[0][1])
    print(model_accuracies[0][2])

    print(model_lst)
