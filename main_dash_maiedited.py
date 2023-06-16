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
from sklearn.metrics import confusion_matrix
import dash_daq as daq
from sklearn import metrics
from model_maiedited import pipe_setup, fit_model
from data_prepare_func import convert_to_array
from bar_graph import bar_graph
from confusion_matrix_fig import confusion_matrix_fig
from classification_report_heat import get_classification_report
import seaborn as sns

# dataset
# if __name__ == '__main__':
#     x_kit, y_kit = convert_to_array('data_fr_kittinan/', 28)
#     x_diy, y_diy = convert_to_array('data_writing_diy/', 28)
#     X = np.append(x_kit, x_diy, axis=0)
#     y = np.append(y_kit, y_diy, axis=0)

X = np.loadtxt('X.csv', delimiter=',', dtype = 'uint8' )
y = np.loadtxt('y.csv', delimiter=',', dtype = 'uint8' )

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
                {'label': 'Neural Network', 'value': 'Neural Network'},
                {'label': 'Random Forest', 'value': 'Random Forest'},
                {'label': 'Logistic Regression', 'value': 'Logistic Regression'},
                {'label': 'Extra Trees Classifier', 'value': 'Extra Trees Classifier'},
                {'label': 'XGB Classifier', 'value': 'XGB Classifier'},
            ],
            value=['Neural Network','Logistic Regression','XGB Classifier' ],
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
        html.Div(id='cm-container', style={'margin-top': '20px'}),
        html.Hr(),
        # html.H5("Classification report"),
        # html.Hr(),
        # html.Div(id='class-container', style={'margin-top': '20px'}),
        # html.Hr(),
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
        # Output('class-container', 'children'),
        Output('accuracy-output', 'children'),
    ],
    [
        Input('model-selector', 'value'),
        Input('test-size-slider', 'value'),
        Input('num-splits-dropdown', 'value')
    ]
)
def update_graphs(selected_model, test_size, num_splits):
    # Split the dataset into training and testing sets

    # Prepare variable to store value
    best_model = None
    best_accuracy = 0.0
    best_model_name = ""
    best_cm = ""
    model_accuracies = []

    # Dict of models
    # model_lst = ['XGB Classifier', 'Logistic Regression',
    #              'Random Forest', 'Neural Network', 'Extra Trees Classifier']
    
    # Iterate through selected models
    for m in selected_model:
        clf = pipe_setup(m)
        fitted_model, train_ac, test_ac, cv_ac, cm, X_train, X_test, y_train, y_test,train_yhat, test_yhat = fit_model(
            clf, X, y, num_splits, test_size)

        # if train_ac > best_accuracy:
        #     best_model = fitted_model
        #     best_accuracy = train_ac
        #     best_model_name = m
        #     best_cm = cm

        if train_ac > best_accuracy:
            best_model = fitted_model
            best_accuracy_train = train_ac
            best_accuracy_cv = cv_ac
            mean_best_accuracy_cv = sum(best_accuracy_cv)/len(best_accuracy_cv)
            best_accuracy_test = test_ac
            best_model_name = m
            best_cm = cm

   

    # append model accuracy (further use as an input for cm, classification report, model performance)
        model_accuracies.append((m, train_ac, cv_ac, test_ac))

    performance_graph = bar_graph(model_accuracies)
    cm_fig = confusion_matrix_fig(best_cm)
    
    # report_df = get_classification_report(y_test, test_yhat)
    # report_df = report_df.iloc[0:10, 0:3]
    # class_fig = sns.heatmap(report_df, annot=True, fmt=".2f")

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
    html.H6(f'Accuracy score of the best model:'),
    # html.P(f'Validation Score: {validate_accuracy:.2f}'),
    # html.P(f'Test Score: {test_accuracy:.2f}')
        ]),
    dbc.Row(
        [
            dbc.Col(
                daq.LEDDisplay(
                    label="Train Score",
                    value=f"{best_accuracy_train:.2f}",
                    style={'font-size': '15px'}
                ),
                width=6
            ),
            dbc.Col(
                daq.LEDDisplay(
                    label="Validation Score",
                    value=f"{mean_best_accuracy_cv:.2f}",
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
                    label=" Test Score",
                    value=f"{best_accuracy_test:.2f}",
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
    )]
    
    # Return the graph components as the outputs of the callback

    return [
        html.Div(dataset_status),
        html.Div(model_status),
        html.Div(dcc.Graph(figure=cm_fig)),
        #html.Div(dcc.Graph(figure=class_fig)),
        html.Div(dcc.Graph(figure=performance_graph))
        
    ]


if __name__ == '__main__':
    app.run_server(debug=True)
