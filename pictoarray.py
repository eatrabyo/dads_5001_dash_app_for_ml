from data_prepare_func import detect_and_crop_handwriting,convert_to_array
import numpy as np

X_train,y_train = convert_to_array('data_train',28)
X_test,y_test = convert_to_array('data_test',28)

np.savetxt("X_train.csv", X_train, delimiter=",", fmt='%d')
np.savetxt("y_train.csv", y_train, fmt='%d')
np.savetxt("X_test.csv", X_test, delimiter=",", fmt='%d')
np.savetxt("y_test.csv", y_test, fmt='%d')

