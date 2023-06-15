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

# dataset
# X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

X = np.loadtxt('X.csv', delimiter=',', dtype = 'uint8' )
y = np.loadtxt('y.csv', delimiter=',', dtype = 'uint8' )

# start dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Sidebar layout

sidebar = html.Div(
    [
        html.Hr(),
        html.H5("Classification Model Simulator", style={'margin-top': '20px'}),
        html.Hr(),
        html.P("Select Models:", style={'background-color': 'lightgray'}),
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
            value=['Logistic Regression', 'Random Forest', 'K-Nearest Neighbors', 'Support Vector Machines',
                   'Decision Tree', 'Neural Network'],
            labelStyle={'display': 'block'}
        ),
        html.P("Test Set Size:", style={'background-color': 'lightgray'}),
        dcc.Slider(
            id='test-size-slider',
            min=0.1,
            max=0.5,
            step=0.1,
            value=0.2,
            marks={i / 10: str(i / 10) for i in range(1, 6)}
        ),
        html.P("Number of Splits:", style={'background-color': 'lightgray'}),
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

    ],
    className="sidebar",
    style={'background-color': 'lightgray', 'padding': '10px'}
)

content1 = html.Div(
    [
        html.Hr(),
        html.H5("Dataset:"),
        html.Hr(),
        html.Div(id='dataset-status', style={'margin-top': '20px'}),
        html.Hr(),
        html.H5("Evaluation:"),
        html.Hr(),
        html.Div(id='model-status', style={'margin-top': '20px'})
    ],
    style={'background-color': 'white', 'padding': '10px'}
)


content2 = html.Div(
    [
        html.Hr(),
        html.H5("Confusion metrics"),
        html.Hr(),
        # html.Div(id='classification-report', style={'margin-top': '20px'}),
        # html.Div(id='roc-container', style={'margin-top': '20px'}),
        html.Div(id='cm-container', style={'margin-top': '20px'}),
        html.Hr(),
        html.H5("Model Performance (Accuracy score):"),
        html.Hr(),
        html.Div(id='accuracy-output', style={'margin-top': '20px'})
    ],
    style={'background-color': 'white', 'padding': '10px'}
)


# Layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col([
                    html.H1('Thai Number Handwritting recognition', style={'text-align': 'center'})
                ], width=12)
            ]
        ),

        dbc.Row(
            [
                dbc.Col(sidebar, width=3),
                dbc.Col(content1, width=3),
                dbc.Col(content2, width=6),
            ]
        )
    ]
)


@app.callback(
    [
        Output('dataset-status', 'children'),
        Output('model-status', 'children'),
        Output('cm-container', 'children'),
        # Output('roc-container', 'children'),
        Output('accuracy-output', 'children'),
        # Output('classification-report', 'children'),
    ],
    [
        Input('model-selector', 'value'),
        Input('test-size-slider', 'value'),
        Input('num-splits-dropdown', 'value')
    ]
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

    # Prepare variable to model accuracy -> later for graph use
    model_accuracies = []

    # Iterate through selected models
    for model_name in selected_models:
        clf = models[model_name]  # Get the classifier for the selected model

        # Initialize variables for average accuracy across splits
        train_accuracy = 0.0
        validate_accuracy = 0.0

        # Perform cross-validation
        for _ in range(num_splits):
            # Split the training set into train and validation sets
            X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=1/num_splits,
                                                                          random_state=42)

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



        # Append model accuracies to the list
        model_accuracies.append((model_name, train_accuracy, validate_accuracy))

    # Make predictions on the test set using the best model
    y_pred = best_model.predict(X_test)

    # Calculate accuracy on the test set
    # test_accuracy = multilabel_confusion_matrix(y_test, y_pred)
    test_accuracy = accuracy_score(y_test, y_pred)

    # label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # report = classification_report(y_test, y_pred)

    # # Calculate precision and recall
    # total_precision = 0
    # for i in range(len(label_names)):
    #     p = metrics.precision_score(np.reshape(y_test, (-1, 1))[:, i], np.reshape(y_pred, (-1, 1))[:, i])
    #     total_precision += p
    #     precision = total_precision
    #     # print("For {} precision: {:.2f}".format(label_names[i], p))
    #     # print("Macro Precision: {:.2f}".format(totalPrecision/len(label_names)))

    # # precision = precision_score(y_test, y_pred)
    # # recall = recall_score(y_test, y_pred)

    # r = metrics.recall_score(np.reshape(y_test, (-1, 1)), np.reshape(y_pred, (-1, 1)), average=None)
    # recall = r

    # # Create ROC curve data for the best model
    # fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
    # roc_auc = auc(fpr, tpr)

    # # Create ROC curve figure for the best model
    # roc_fig = go.Figure()
    # roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Best Model (AUC = {roc_auc:.2f})'))
    # roc_fig.update_layout(
    #     xaxis=dict(title='False Positive Rate'),
    #     yaxis=dict(title='True Positive Rate'),
    #     showlegend=True
    # )

    # roc_fig.update_layout(legend=dict(
    #     orientation="h",
    #     yanchor="bottom",
    #     y=1.02,
    #     xanchor="right",
    #     x=1
    # ))



    # Create confusion matrix for the best model
    cm = confusion_matrix(y_test, y_pred)
    graph_classes = np.arange(10)
    # heatmap_data = go.Heatmap(z=cm, x=graph_classes, y=graph_classes, colorscale='Blues')

    # Create confusion matrix figure for the best model
    cm_fig = go.Figure(data=go.Heatmap(
        z = cm,
        x = graph_classes,
        y = graph_classes,
        colorscale='YlGnBu',
        reversescale=True,
        hovertemplate='True label: %{y}<br>Predicted label: %{x} <br>Frequency: %{z}',

    ))

    # add annotated in diagonal
    for i in range(len(cm)):
        cm_fig.add_annotation(
            x=i, y=i,
            text=str(cm[i][i]),
            showarrow=False,
            font=dict(color='white' if cm[i][i] < 0.5 else 'black')
    )

    cm_fig.update_layout(
    # title='Confusion Matrix',
    xaxis=dict(
        title='Predicted label',
        tickmode='array',
        tickvals=list(range(len(graph_classes))),
        ticktext=graph_classes
    ),
    yaxis=dict(
        title='True label',
        tickmode='array',
        tickvals=list(range(len(graph_classes))),
        ticktext=graph_classes
    )
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
    ))


    # Create a list of model accuracy components
    accuracy_components = [
        html.P(f"Test Accuracy: {test_accuracy:.2f}"),
        # html.P(f"Precision: {precision:.2f}"),
        # html.P(f"Recall: {recall:.2f}")
    ]

    # Create a list of model accuracy components for all models
    model_accuracy_components = [
        html.H4('Model Accuracies'),
        dbc.Table(
            [
                html.Thead(
                    html.Tr([html.Th('Model'), html.Th('Train Accuracy'), html.Th('Validation Accuracy')])
                ),
                html.Tbody(
                    [
                        html.Tr([html.Td(model_name), html.Td(train_accuracy), html.Td(validate_accuracy)])
                        for model_name, train_accuracy, validate_accuracy in model_accuracies
                    ]
                )
            ],
            bordered=True,
            hover=True,
            responsive=True,
            striped=True
        )
    ]

    # Create the graph components
    # roc_graph = dcc.Graph(figure=roc_fig)
    # cm_graph = dcc.Graph(figure=cm_fig)


    # Prepare variable for dataset status
    dataset_status = [
    dbc.Row(
        [
            dbc.Col(
                daq.LEDDisplay(
                    label="No. of Record",
                    value=str(X.shape[0]),
                    style={'font-size': '15px'}
                ),
                width=6
            ),
            dbc.Col(
                daq.LEDDisplay(
                    label="No. of Categories",
                    value=str(np.unique(y).shape[0]),
                    style={'font-size': '15px'}
                ),
                width=6
            )
        ],
        className="mb-3"
    ),
    dbc.Row(
        [
            dbc.Col(
                daq.LEDDisplay(
                    label="No. of Train set",
                    value=str(X_train.shape[0]),
                    style={'font-size': '15px'}
                ),
                width=6
            ),
            dbc.Col(
                daq.LEDDisplay(
                    label="No. of Test set",
                    value=str(X_test.shape[0]),
                    style={'font-size': '15px'}
                ),
                width=6
            )
        ],
        className="mb-3"
    )
]

# Create model status
    model_status = [
    dbc.Row(
        [
    html.H6(f'Best Model: {best_model_name}'),
    html.P(f'Train Score: {best_accuracy:.2f}'),
    html.P(f'Validation Score: {validate_accuracy:.2f}'),
    html.P(f'Test Score: {test_accuracy:.2f}')
        ]),
    dbc.Row(
        # [
        #     dbc.Col(
        #         daq.LEDDisplay(
        #             label="Precision",
        #             # value=f"{precision:.2f}",
        #             style={'font-size': '15px'}
        #         ),
        #         width=6
        #     ),
        #     dbc.Col(
        #         daq.LEDDisplay(
        #             label="Recall",
        #             # value=f"{recall:.2f}",
        #             style={'font-size': '15px'}
        #         ),
        #         width=6
        #     )
        # ],
        # className="mb-3"
    ),
    dbc.Row(
        [
            dbc.Col(
                daq.LEDDisplay(
                    label="Accuracy Score",
                    value=f"{test_accuracy:.2f}",
                    style={'font-size': '15px'}
                ),
                width=6
            )
            # dbc.Col(
            #     daq.LEDDisplay(
            #         label="AUC",
            #         value=f"{roc_auc:.2f}",
            #         style={'font-size': '15px'}
            #     ),
            #     width=6
            # )
        ],
        className="mb-3"
    )
]

    # Return the graph components as the outputs of the callback

    return [
        html.Div(dataset_status),
        html.Div(model_status),
        # html.Div(dcc.Markdown(report)),
        html.Div(dcc.Graph(figure=cm_fig)),
        html.Div(dcc.Graph(figure=performance_graph))
        
    ]

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)

