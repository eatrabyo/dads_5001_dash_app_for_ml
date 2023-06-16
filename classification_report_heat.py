import seaborn as sns
import pandas as pd


def get_classification_report(y_test, y_pred):
    from sklearn import metrics
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    return df_classification_report

def classification_fig(y_test, test_yhat):
    report_df = get_classification_report(y_test, test_yhat)
    report_df = report_df.iloc[0:10, 0:3]
    class_fig = sns.heatmap(report_df, annot=True, fmt=".2f")
    return class_fig , report_df

