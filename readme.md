# Handwritten characters recognizer

Neural network project for recognizing and classifying 
handwritten digits and letters with Convolutional algorith.

Datasets used:
* Deep Learning A-Z dataset from Kaggle
* MNIST Handwritten Digit Dataset from Kaggle

datasets were combined into one behind the scene.

Sequential model was initiated and layer by layer were added. 
We create Conv2D layers doubling filters number every pair.
Next, to these layers, the pooling layer is used which reduces the
dimensionality of the image and computation in the network.
Following it, Flatten Layer is used which generates a column matrix
from the 2-dimensional matrix.
This column matrix will be fed into Dense layer with 64 neurons and,
then, to Dense layer with 128 neurons.
Finally, output layer with 36 neurons representing 10 digits and 25 letters.

Model accuracy = 0.9718,
Model loss = 0.1004

## How to run
```
cd character_recognizer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
dir - directory with images of characters your want to check
```
python character_recognizer/inference.py dir
```

Printed results:
* character ASCII index in decimal
* format and path to image sample

## Train model

If you want to work on model itself, consider downloading 
https://www.kaggle.com/datasets/kostyantynchernyk/handwritten-digits-and-letters
dataset.
Not much changed here, 0-9 labels represent respective digits, 
letter labels, however, were incremented by 10 to avoid conflict with digits, 
e.g. 10 is A, 11 is B ... 35 is Z
