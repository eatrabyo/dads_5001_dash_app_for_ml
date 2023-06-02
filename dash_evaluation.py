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
from sklearn.feature_selection import SelectFromModel
import seaborn as sns

# Generate a toy classification dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Classification Model Simulator"),
    html.Div([
        html.Label("Select Model:"),
        dcc.Checklist(
            id='model-selector',
            options=[
                {'label': 'Logistic Regression', 'value': 'lr'},
                {'label': 'Random Forest', 'value': 'rf'}
            ],
            value=['lr']
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
    ], style={'width': '50%', 'margin': 'auto'}),
    html.Div([
        dcc.Graph(id='roc-graph')
    ], style={'width': '50%', 'margin': 'auto'}),
    html.Div([
        dcc.Graph(id='tpr-fpr-graph')
    ], style={'width': '50%', 'margin': 'auto'}),
    html.Div([
        dcc.Graph(id='correlation-matrix')
    ], style={'width': '50%', 'margin': 'auto'}),
    html.Div([
        dcc.Graph(id='variable-importance')
    ], style={'width': '50%', 'margin': 'auto'}),
    html.Div(id='accuracy-output', style={'text-align': 'center', 'margin-top': '20px'}),
])

# Define the callback function for model selection and test set size
@app.callback(
    [Output('roc-graph', 'figure'), Output('tpr-fpr-graph', 'figure'), Output('correlation-matrix', 'figure'), Output('variable-importance', 'figure'), Output('accuracy-output', 'children')],
    [Input('model-selector', 'value'), Input('test-size-slider', 'value')]
)
def update_graphs(selected_models, test_size):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize empty figures
    roc_fig = go.Figure()
    tpr_fpr_fig = go.Figure()
    corr_matrix_fig = go.Figure()
    var_imp_fig = go.Figure()

    # Iterate through selected models
    for model in selected_models:
        if model == 'lr':
            # Train a logistic regression model
            clf = LogisticRegression()
            clf.fit(X_train, y_train)
        elif model == 'rf':
            # Train a random forest model
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Create ROC curve data
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Add ROC curve to figure
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model} (AUC = {roc_auc:.2f})'))

        # Add TPR/FPR graph data
        tpr_fpr_fig.add_trace(go.Bar(x=['TPR'], y=[recall], name=f'{model} (Recall)'))
        tpr_fpr_fig.add_trace(go.Bar(x=['FPR'], y=[1 - precision], name=f'{model} (False Positive Rate)'))

        # Calculate correlation matrix
        df = pd.DataFrame(X)
        corr_matrix = df.corr()

        # Create correlation matrix heatmap
        corr_matrix_fig.add_trace(go.Heatmap(
            x=df.columns,
            y=df.columns,
            z=corr_matrix.values,
            colorscale='Viridis',
            reversescale=True
        ))

        # Calculate variable importance (feature importance) for tree-based models
        if isinstance(clf, RandomForestClassifier):
            feature_importance = clf.feature_importances_
            feature_names = df.columns

            # Sort feature importance in descending order
            sorted_indices = np.argsort(feature_importance)[::-1]
            sorted_importance = feature_importance[sorted_indices]
            sorted_names = feature_names[sorted_indices]

            # Add variable importance bar chart
            var_imp_fig.add_trace(go.Bar(x=sorted_names, y=sorted_importance, name=model))

    # Set layout for ROC curve figure
    roc_fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis=dict(title='False Positive Rate'),
        yaxis=dict(title='True Positive Rate'),
        legend=dict(x=0.7, y=0.9)
    )

    # Set layout for TPR/FPR figure
    tpr_fpr_fig.update_layout(
        title='True Positive Rate (Recall) vs False Positive Rate',
        yaxis=dict(title='Score'),
        legend=dict(x=0.7, y=0.9)
    )

    # Set layout for correlation matrix figure
    corr_matrix_fig.update_layout(
        title='Correlation Matrix',
        xaxis=dict(title='Features'),
        yaxis=dict(title='Features'),
        legend=dict(x=0.7, y=0.9)
    )

    # Set layout for variable importance figure
    var_imp_fig.update_layout(
        title='Variable Importance',
        xaxis=dict(title='Features'),
        yaxis=dict(title='Importance'),
        legend=dict(x=0.7, y=0.9)
    )

    # Return the figures and accuracy value
    return roc_fig, tpr_fpr_fig, corr_matrix_fig, var_imp_fig, f"Accuracy: {accuracy:.2f}"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
