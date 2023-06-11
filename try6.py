import dash
from dash import dcc
from dash import html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import dash_daq as daq

# dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# start dash
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Classification Model Simulator"),
    html.Div([
        html.Label("Select Models:"),
        dcc.Checklist(
            id='model-selector',
            options=[
                {'label': 'Logistic Regression', 'value': 'Logistic Regression'},
                {'label': 'Random Forest', 'value': 'Random Forest'},
                {'label': 'K-Nearest Neighbors', 'value': 'K-Nearest Neighbors'},
                {'label': 'Support Vector Machines', 'value': 'Support Vector Machines'},
                {'label': 'Decision Tree', 'value': 'Decision Tree'},
                {'label': 'Neural Network', 'value': 'Neural Network'},
            ],
            value=['Logistic Regression', 'Random Forest', 'K-Nearest Neighbors', 'Support Vector Machines', 'Decision Tree', 'Neural Network'],
            labelStyle={'display': 'inline-block'}
        ),
        html.Label("Test Set Size:"),
        dcc.Slider(
            id='test-size-slider',
            min=0.1,
            max=0.5,
            step=0.1,
            value=0.2,
            marks={i/10: str(i/10) for i in range(1, 6)}
        ),
        html.Label("Number of Splits:"),
        dcc.Dropdown(
            id='num-splits-dropdown',
            options=[
                {'label': '2', 'value': 2},
                {'label': '3', 'value': 3},
                {'label': '4', 'value': 4},
                {'label': '5', 'value': 5}
            ],
            value=2
        ),
        html.Div(id='dataset-status', style={'margin-top': '20px'}),
        html.Div(id='model-status', style={'margin-top': '20px'})
    ], style={'width': '50%', 'margin': 'auto'}),
    html.Div(id='roc-container', style={'width': '80%', 'margin': 'auto'}),
    html.Div(id='accuracy-output', style={'text-align': 'center', 'margin-top': '20px'})
])


# Part callback
@app.callback(
    [Output('roc-container', 'children'),
     Output('accuracy-output', 'children'),
     Output('dataset-status', 'children'),
     Output('model-status', 'children')],
    [Input('model-selector', 'value'),
     Input('test-size-slider', 'value'),
     Input('num-splits-dropdown', 'value')]
)
def update_graphs(selected_models, test_size, num_splits):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Prepare variable to store value
    best_model = None
    best_accuracy = 0.0
    best_model_name = ""

    # Dict of models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machines': SVC(probability=True),
        'Decision Tree': DecisionTreeClassifier(),
        'Neural Network': MLPClassifier()
    }

    # Prepare variable for dataset status
    dataset_status = [
        daq.LEDDisplay(
            label="Number of Records",
            value=str(X.shape[0])
        ),
        daq.LEDDisplay(
            label="Number of Training Dataset",
            value=str(X_train.shape[0])
        ),
        daq.LEDDisplay(
            label="Number of Testing Dataset",
            value=str(X_test.shape[0])
        ),
        daq.LEDDisplay(
            label="Number of Categories",
            value=str(np.unique(y).shape[0])
        )
    ]

    # Iterate through selected models
    for model_name in selected_models:
        clf = models[model_name]  # Get the classifier for the selected model

        # Initialize variables for average accuracy across splits
        train_accuracy = 0.0
        validate_accuracy = 0.0

        # Perform cross-validation
        for _ in range(num_splits):
            # Split the training set into train and validation sets
            X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            # Fit the model
            clf.fit(X_train_split, y_train_split)

            # Make predictions on the train and validation sets
            y_train_pred = clf.predict(X_train_split)
            y_val_pred = clf.predict(X_val)

            # Calculate accuracy on the train and validation sets
            train_accuracy += accuracy_score(y_train_split, y_train_pred)
            validate_accuracy += accuracy_score(y_val, y_val_pred)

        # Calculate average accuracy across splits
        train_accuracy /= num_splits
        validate_accuracy /= num_splits

        # Check if current model has higher accuracy than previous best model
        if train_accuracy > best_accuracy:
            best_model = clf
            best_accuracy = train_accuracy
            best_model_name = model_name

    # Make predictions on the test set using the best model
    y_pred = best_model.predict(X_test)

    # Calculate accuracy, precision, recall on the test set
    test_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Create ROC curve data for the best model
    fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Create ROC curve figure for the best model
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Best Model (AUC = {roc_auc:.2f})'))
    roc_fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
        legend=dict(x=0.7, y=0.9)
    )

    # Create performance bar graph
    model_names = [m for m in selected_models]
    train_scores = [train_accuracy] * len(selected_models)
    validate_scores = [validate_accuracy] * len(selected_models)
    test_scores = [test_accuracy] * len(selected_models)

    performance_graph = go.Figure()
    performance_graph.add_trace(go.Bar(x=model_names, y=train_scores, name='Train Score'))
    performance_graph.add_trace(go.Bar(x=model_names, y=validate_scores, name='Validation Score'))
    performance_graph.add_trace(go.Bar(x=model_names, y=test_scores, name='Test Score'))
    performance_graph.update_layout(
        title='Model Performance',
        xaxis=dict(title='Models'),
        yaxis=dict(title='Accuracy Score'),
        barmode='group'
    )

    # Create model status
    model_status = [
        daq.LEDDisplay(
            label="Precision",
            value=f"{precision:.2f}"
        ),
        daq.LEDDisplay(
            label="Recall",
            value=f"{recall:.2f}"
        ),
        daq.LEDDisplay(
            label="Accuracy Score",
            value=f"{test_accuracy:.2f}"
        ),
        daq.LEDDisplay(
            label="AUC",
            value=f"{roc_auc:.2f}"
        )
    ]

    # Create accuracy output
    accuracy_output = [
        html.H3(f'Best Model: {best_model_name}'),
        html.P(f'Train Score: {best_accuracy:.2f}'),
        html.P(f'Validation Score: {validate_accuracy:.2f}'),
        html.P(f'Test Score: {test_accuracy:.2f}')
    ]

    # Return the ROC curve figure, accuracy output, dataset status, and model status
    return [dcc.Graph(figure=roc_fig), dcc.Graph(figure=performance_graph)], accuracy_output, dataset_status, model_status


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
