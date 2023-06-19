import seaborn as sns
import pandas as pd
import plotly.express as px


def get_classification_report(y_test, y_pred):
    from sklearn import metrics
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    return df_classification_report

def classification_fig(y_test, test_yhat):
    report_df = get_classification_report(y_test, test_yhat)
    report_df = report_df.iloc[0:10, 0:3]
    report_df = report_df.round(2)
    class_fig = px.imshow(report_df, x= report_df.columns, y = report_df.index, 
                          width=50,
                          color_continuous_scale='YlGnBu_r',text_auto=True)
    
    class_fig.update_traces(hovertemplate='digit: %{x}</> Value:%{z:.2f}</>Score:%{y}')
    class_fig.update_xaxes(side="top")
    class_fig.update_layout(showlegend = False)
    
    return class_fig 

