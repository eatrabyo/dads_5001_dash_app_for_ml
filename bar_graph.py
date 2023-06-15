import dash
# import dash_core_components as dcc
# import dash_html_components as html
from dash import html
import dash_bootstrap_components as dbc
from dash import dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score
import dash_daq as daq
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics


def bar_graph(model_accuracies):
    # Split data
    model_names = [data[0] for data in model_accuracies]
    train_scores = [data[1] for data in model_accuracies]
    validate_scores = [np.mean(data[2]) for data in model_accuracies]
    test_scores = [data[3] for data in model_accuracies]

    # Create bar graph
    performance_graph = go.Figure()
    performance_graph.add_trace(go.Bar(x=model_names, y=train_scores, name='Train Score',marker_color='rgb(8,29,88)'))
    performance_graph.add_trace(go.Bar(x=model_names, y=validate_scores, name='Validation Score',marker_color='rgb(57,174,195)'))
    performance_graph.add_trace(go.Bar(x=model_names, y=test_scores, name='Test Score',marker_color='rgb(233,247,177)'))
    performance_graph.update_layout(
        xaxis=dict(title='Models'),
        yaxis=dict(title='Accuracy Score'),
        barmode='group'
    )
    performance_graph.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),plot_bgcolor='rgb(255,255,255)')
    return performance_graph

if __name__ == '__main__':

    model_accuracies = [('XGB Classifier', 0.7982786444324906, np.array([0.69354839, 0.64247312, 0.66129032, 0.69354839, 0.62803235]), 0.6553884711779449),
            ('Logistic Regression', 0.9435180204410973, np.array([0.79032258, 0.80913978, 0.75537634, 0.80107527, 0.75202156]), 0.8020050125313283),
            ('Random Forest', 0.9575040344271114, np.array([0.79032258, 0.78225806, 0.76612903, 0.80107527, 0.73854447]), 0.7769423558897243),
            ('Neural Network', 0.9795589026358257, np.array([0.81182796, 0.79032258, 0.78494624, 0.80645161, 0.75471698]), 0.8082706766917294),
            ('Extra Trees Classifier', 0.9381387842926304, np.array([0.78494624, 0.78225806, 0.72849462, 0.78494624, 0.75202156]), 0.7756892230576441)]
    
    a = bar_graph(model_accuracies)
    a.show()

