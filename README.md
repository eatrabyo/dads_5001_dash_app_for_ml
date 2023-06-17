# DADS 5001: Machine Learning Model Simulation
# Mini-Project : Thai Number Handwritting Recognition Application

# Source data
Data source:
  1. kittinan/thai-handwriting-number
  2. Written Thai Number by ourself

# Sample Web Page
![image](https://github.com/eatrabyo/dads_5001_dash_app_for_ml/assets/114765725/0ebf15d0-8c32-41d4-8064-76e72b6ac308)

# Component Part
  1. Predict your digit handwritting
  2. Classification Model Simulator
  3. Dataset
  4. Evaluation
  5. Confusion matrix
  6. Model Performance (Accuracy score)

# 1. Predict your digit handwritting
The fisrt of all component, this one will predict your digit handwritting by upload picture(.png) into Drag and Drop or Select Files.
Then app will show your image and result on predicted value.

# 2. Classification Model Simulator
There are 3 options:
  1. Select Models    : Use for analyzes data patterns to determine future outcomes which provide 5 model are Neural Network, Random Forest, Logistic Regression,                         Extra Trees Classifier and XGB Classifier
  2. Test Set Size    : This will be split dataset to train and test size. Able to split test size minimun 0.1 and maximun 0.5
  3. Number of Splits : Setting number of K-fold (minimun 2 and maximun 5) for cross validation that is resampling procedure used to evaluate machine learning                             models on a limited data sample.

# 3. Dataset
In this part show dataset information
  1. No. of Record      : Show total number of images in the dataset
  2. No. of Categories  : Show total number of categories. Maximun is 10 categories (๐ ๑ ๒ ๓ ๔ ๕ ๖ ๗ ๘ ๙)
  3. No. of Train set   : Show total number of images in train dataset
  4. No. of Test set    : Show total number of images in test dataset

# 4. Evaluation
In this part show evaluate results
  1. Best Model                       : Show the best model which is the highest train score. Model refer from user selecting model
  2. Accuracy score of the best model : Show score of best model that are train score, validation score and test score

# 5. Confusion matrix
pls fill information

# 6. Model Performance (Accuracy score)
pls fill information

## Credits

- [Data Set](https://github.com/kittinan/thai-handwriting-number) from [kittinan](https://github.com/kittinan)
