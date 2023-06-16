import base64
import cv2
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash
from dash import html
from dash import dcc
import numpy as np
from dash.dependencies import Input, Output

def detect_and_crop_handwriting(image):
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = image.shape[:2]
    center_x = width // 2
    center_y = height // 2
    max_offset = -1
    max_offset_contour = None

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] == 0:
            cX = 0
            cY = 0
        else:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        offset = np.sqrt((center_x - cX) ** 2 + (center_y - cY) ** 2)

        if offset > max_offset:
            max_offset = offset
            max_offset_contour = contour

    if max_offset_contour is not None:
        x, y, w, h = cv2.boundingRect(max_offset_contour)

        aspect_ratio = float(w) / h

        if aspect_ratio > 1:
            y_padding = int((w - h) / 2)
            x_padding = 0
        else:
            x_padding = int((h - w) / 2)
            y_padding = 0

        x -= x_padding
        w += 2 * x_padding
        y -= y_padding
        h += 2 * y_padding

        x = max(x, 0)
        w = min(w, width)
        y = max(y, 0)
        h = min(h, height)

        cropped_image = image[y:y + h, x:x + w]

        return cropped_image

    else:
        print('No handwriting detected in the image.')
        return None

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
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = detect_and_crop_handwriting(img)
        img = cv2.resize(img, (28, 28))
        img = img.flatten()
        # cv2.imshow('window',img)
        # cv2.waitKey(0)
        print(img.shape)
        print(img)

        # แสดงภาพที่อัพโหลด
        return html.Div([
            html.H5(f'ไฟล์ที่อัพโหลด: {filename}'),
            html.Img(src=contents)
        ])
    else:
        return None


if __name__ == '__main__':
    app.run_server(debug=True)