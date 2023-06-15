import base64
import cv2
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
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


app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'ลากและวางไฟล์รูปภาพ PNG ที่นี่'
            ]),
            style={
                'width': '100%',
                'height': '200px',
                'lineHeight': '200px',
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
    ]
)


@app.callback(Output('output-image', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'))
def update_output(contents, filename):
    if contents is not None:
        # ดึงข้อมูลรูปภาพจาก base64
        image_data = contents.split(',')[1]
        decoded_image = base64.b64decode(image_data)

        # บันทึกรูปภาพเป็นไฟล์ PNG
        with open(filename, 'wb') as f:
            f.write(decoded_image)

        # convert to array
        nparr = np.frombuffer(decoded_image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        print(img)
        print(img.shape)

        # แสดงภาพที่อัพโหลด
        return html.Div([
            html.H5(f'ไฟล์ที่อัพโหลด: {filename}'),
            html.Img(src=contents)
        ])
    else:
        return None


if __name__ == '__main__':
    app.run_server(debug=True)
