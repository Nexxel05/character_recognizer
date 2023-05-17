import os
import sys
import cv2
import numpy as np
import tensorflow as tf

MAPPING = {
    0: "048",
    1: "049",
    2: "050",
    3: "051",
    4: "052",
    5: "053",
    6: "054",
    7: "055",
    8: "056",
    9: "057",
    10: "065",
    11: "066",
    12: "067",
    13: "068",
    14: "069",
    15: "070",
    16: "071",
    17: "072",
    18: "073",
    19: "074",
    20: "075",
    21: "076",
    22: "077",
    23: "078",
    24: "079",
    25: "080",
    26: "081",
    27: "082",
    28: "083",
    29: "084",
    30: "085",
    31: "086",
    32: "087",
    33: "088",
    34: "089",
    35: "090"

}

directory = sys.argv[-1]

model = tf.keras.models.load_model("model.h5")

for file in os.listdir(directory):
    file_name = os.fsdecode(file)
    path = os.path.join(directory, file_name)
    try:
        img = cv2.imread(path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_final = cv2.resize(img_gray, (28, 28))
        img_final = np.reshape(img_final, (1, 28, 28, 1))

        prediction = MAPPING[np.argmax(model.predict(img_final))]
        print(f"{prediction}, {path}")
    except:
        print("Error occurred")
