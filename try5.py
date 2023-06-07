import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns

# Generate a toy classification dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Classification Model Simulator"),
    html.Div([
        #ไม่ให้เลือก model ละ ใช้ทุกอันแล้วเลือกอันที่ดีที่สุดแทน
        # html.Label("Select Model:"),
        # dcc.Checklist(
        #     id='model-selector',
        #     options=[
        #         {'label': 'Logistic Regression', 'value': 'lr'},
        #         {'label': 'Random Forest', 'value': 'rf'},
        #         {'label': 'K-Nearest Neighbors', 'value': 'knn'},
        #         {'label': 'Support Vector Machines', 'value': 'svm'},
        #         {'label': 'Decision Tree', 'value': 'dt'},
        #         {'label': 'Neural Network', 'value': 'nn'},
        #     ],
        #     value=['lr'],
        #     labelStyle={'display': 'inline-block'}
        # ),
        html.Label("Test Set Size:"),
        dcc.Slider(
            id='test-size-slider',
            min=0.1,
            max=0.5,
            step=0.1,
            value=0.2,
            marks={i/10: str(i/10) for i in range(1, 6)}
        ),
    ], style={'width': '50%', 'margin': 'auto'}),
    html.Div(id='graphs-container', style={'width': '80%', 'margin': 'auto'}),
])

@app.callback(
    Output('graphs-container', 'children'),
    [Input('test-size-slider', 'value')]
)
def update_graphs(test_size):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize variables for the best model
    best_model = None
    best_accuracy = 0.0
    best_model_name = ""

    # Dictionary of models to iterate through
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True),
        'Decision Tree': DecisionTreeClassifier(),
        'Neural Network': MLPClassifier()
    }

    # Iterate through models
    for model_name, clf in models.items():
        # Fit the model
        clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Check if current model has higher accuracy than previous best model
        if accuracy > best_accuracy:
            best_model = clf
            best_accuracy = accuracy
            best_model_name = model_name

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

    # Create accuracy output
    accuracy_output = html.Div([
        html.H3('Best Model Accuracy:', style={'text-align': 'center', 'display': 'inline-block'}),
        html.H4(f'{best_model_name} ({best_accuracy:.2f})', style={'text-align': 'center', 'display': 'inline-block', 'margin-left': '10px'})
    ])

    # Return the ROC curve figure and accuracy output
    return [dcc.Graph(figure=roc_fig), accuracy_output]


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
