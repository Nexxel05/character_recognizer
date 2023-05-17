import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def load_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path).astype("float32")
    data = dataset.drop("10", axis=1)
    labels = dataset["10"]
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.2)

    train_x = np.reshape(train_x.values, (train_x.shape[0], 28, 28))
    test_x = np.reshape(test_x.values, (test_x.shape[0], 28, 28))

    train_x_final = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
    test_x_final = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)

    train_y_final = to_categorical(train_y, num_classes=36, dtype="int")
    test_y_final = to_categorical(test_y, num_classes=36, dtype="int")

    return train_x_final, test_x_final, train_y_final, test_y_final
