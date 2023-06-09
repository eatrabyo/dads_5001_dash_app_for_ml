import plotly.graph_objects as go
import numpy as np


def confusion_matrix_fig(cm):
    # Create confusion matrix for the best model
    # cm = confusion_matrix(y_test, y_pred)
    graph_classes = np.arange(10)

    # Create confusion matrix figure for the best model
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=graph_classes,
        y=graph_classes,
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
        ))
    return cm_fig


if __name__ == '__main__':
    cm = np.array([[61,  5,  0,  5,  0,  2,  0,  1,  3,  2],
                   [9, 62,  0,  6,  0,  1,  0,  1,  0,  1],
                   [0,  0, 50,  5,  1,  2,  9,  6,  2,  5],
                   [1,  0,  1, 70,  0,  1,  0,  1,  3,  4],
                   [0,  0,  0,  0, 32, 19,  1,  6, 13,  8],
                   [0,  0,  0,  0,  5, 57,  1,  2,  5,  9],
                   [0,  0, 11,  0,  3,  2, 47,  5, 11,  0],
                   [0,  2,  0,  0,  1,  3,  1, 49,  5, 19],
                   [0,  0,  0,  3,  5,  8,  4, 10, 34, 15],
                   [0,  0,  0,  0,  3,  8,  2,  4,  4, 61]])

    a = confusion_matrix(cm)
    a.show()
