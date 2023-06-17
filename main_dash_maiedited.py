import dash
from dash import html
import dash_bootstrap_components as dbc
from dash import dcc
import numpy as np
from dash.dependencies import Input, Output
import dash_daq as daq
from model_maiedited import pipe_setup, fit_model
from bar_graph import bar_graph
from confusion_matrix_fig import confusion_matrix_fig
import base64
from dash.dependencies import Input, Output, State
import cv2
from data_prepare_func import detect_and_crop_handwriting

# Load dataset
X = np.loadtxt('X.csv', delimiter=',', dtype='uint8')
y = np.loadtxt('y.csv', delimiter=',', dtype='uint8')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Sidebar layout
sidebar = html.Div(
    [
        html.Hr(),
        html.H5("Predict your digit handwritting",
                style={'margin-top': '20px'}),
        html.Hr(),
        html.Div(children=[
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '95%',
                    'height': '100px',
                    'lineHeight': '100px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=False
            ),
            html.Div(id='output-image')
        ]),

        html.Hr(),
        html.Div(id='Predict-from-pic', style={'margin-top': '20px'}),
        html.Hr(),
        html.H5("Classification Model Simulator",
                style={'margin-top': '20px'}),
        html.Hr(),
        html.P("Select Models:", style={'background-color': 'lightgray'}),
        dcc.Checklist(
            id='model-selector',
            options=[
                {'label': 'Neural Network', 'value': 'Neural Network'},
                {'label': 'Random Forest', 'value': 'Random Forest'},
                {'label': 'Logistic Regression', 'value': 'Logistic Regression'},
                {'label': 'Extra Trees Classifier',
                    'value': 'Extra Trees Classifier'},
                {'label': 'XGB Classifier', 'value': 'XGB Classifier'},
            ],
            value=['Neural Network', 'Logistic Regression', 'XGB Classifier'],
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
        html.H5("Confusion matrix"),
        html.Hr(),
        html.Div(id='cm-container', style={'margin-top': '5px'}),
        html.Hr(),
        html.H5("Model Performance (Accuracy score):"),
        html.Hr(),
        html.Div(id='accuracy-output', style={'margin-top': '5px'})
    ],
    style={'background-color': 'white', 'padding': '10px'}
)

# Thainum picture
thainum_png = 'Thainum.png'
thainum_base64 = base64.b64encode(
    open(thainum_png, 'rb').read()).decode('ascii')

# Layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col([
                    html.H1('Thai Number Handwritting recognition',
                            style={'text-align': 'center'})
                ], width=12)
            ]
        ),

        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        style={
                            'display': 'flex',
                            'justify-content': 'center',
                            'align-items': 'center',
                            'height': '100%',
                        },
                        children=[
                            html.Img(
                                src='data:image/png;base64,{}'.format(
                                    thainum_base64),
                                alt='Image',
                                style={
                                    'max-width': '50%',
                                    'max-height': 'auto',
                                    'object-fit': 'contain',
                                }
                            )
                        ]
                    ),
                    width=12
                )
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


@app.callback(Output('output-image', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'))
def update_output(contents, filename):
    if contents is not None:

        # แสดงภาพที่อัพโหลด
        return html.Div([
            html.H5(f'Filename: {filename}'),
            html.Img(src=contents)
        ])
    else:
        return None


@app.callback(
    [
        Output('dataset-status', 'children'),
        Output('model-status', 'children'),
        Output('cm-container', 'children'),
        Output('accuracy-output', 'children'),
        Output('Predict-from-pic', 'children')
    ],
    [
        Input('model-selector', 'value'),
        Input('test-size-slider', 'value'),
        Input('num-splits-dropdown', 'value'),
        Input('upload-image', 'contents')
    ]
)
def update_graphs(selected_model, test_size, num_splits, new_img):
    # Split the dataset into training and testing sets

    # Prepare variable to store value
    best_model = None
    best_accuracy = 0.0
    best_model_name = ""
    best_cm = ""
    model_accuracies = []

    # check selected model must not be blank
    if len(selected_model) == 0:
        selected_model = ['Logistic Regression']

    # Iterate through selected models
    for m in selected_model:
        clf = pipe_setup(m)
        fitted_model, train_ac, test_ac, cv_ac, cm, X_train, X_test, y_train, y_test, train_yhat, test_yhat = fit_model(
            clf, X, y, num_splits, test_size)

        if train_ac > best_accuracy:
            best_model = fitted_model
            best_accuracy = train_ac
            best_model_name = m
            best_cm = cm
            best_accuracy_cv = cv_ac
            mean_best_accuracy_cv = sum(best_accuracy_cv)/len(best_accuracy_cv)
            best_accuracy_test = test_ac

    # append model accuracy (further use as an input for cm, classification report, model performance)
        model_accuracies.append((m, train_ac, cv_ac, test_ac))

    predict_result = []
    if new_img is not None:
        # new pic from drag and drop
        image_data = new_img.split(',')[1]
        decoded_image = base64.b64decode(image_data)

        # convert to array
        new_x = []
        nparr = np.frombuffer(decoded_image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = detect_and_crop_handwriting(img)
        img = cv2.resize(img, (28, 28))
        new_x.append(img.flatten())
        new_x = np.array(new_x)
        predict_result = best_model.predict(new_x)
    else:
        pass

    performance_graph = bar_graph(model_accuracies)
    cm_fig = confusion_matrix_fig(best_cm)
    # class_fig = classification_fig(y_test, test_yhat)

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

            ]),
        dbc.Row(
            [
                dbc.Col(
                    daq.LEDDisplay(
                        label="Train Score",
                        value=f"{best_accuracy:.2f}",
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
            ],
            className="mb-3"
        )]

    # predict result
    predict_from_pic = [
        dbc.Row([
            html.H6(f'predicted value: {predict_result}')]
        )]

    # Return the graph components as the outputs of the callback

    return [
        html.Div(dataset_status),
        html.Div(model_status),
        html.Div(dcc.Graph(figure=cm_fig)),
        # html.Div(dcc.Graph(figure=class_fig)),
        html.Div(dcc.Graph(figure=performance_graph)),
        html.Div(predict_from_pic)

    ]


if __name__ == '__main__':
    app.run_server(debug=True)
