# Project Road Segmentation

For this project, we were given a set of satellite images acquired 
from GoogleMaps and ground-truth images where each pixel is labeled 
as road (1) or background (0). 

Our task is to train a classifier to segment roads in these images, i.e. 
assigns a label `road=1, background=0` to each pixel.

## Submission system environment setup:

1. The dataset is available from the 
[AICrowd page](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation).

2. Given the python notebook `segment_aerial_images.ipynb`, an example code on how to extract the images as well as 
corresponding labels of each pixel, we must develop new solutions. The notebook shows how to use `scikit learn` to generate features from each pixel, and finally train a linear classifier to predict whether each pixel is road or background. The example code here also provides helper functions to visualize the images, labels and predictions. In particular, the two functions `mask_to_submission.py` and `submission_to_mask.py` helped us to convert from the submission format to a visualization, and vice versa.

3. As a more advanced approach, we were given the file `tf_aerial_images.py`, which demonstrates the use of a basic convolutional neural network in TensorFlow for the same prediction task.

4. The first method we tried to implement is a CNN, of the architecture of AlexNet. (Pre-trained or not TBD) Then UNet, kNN, SVM, ...

5. In order to evaluate our solutions 
Evaluation Metric:
 [F1 score](https://en.wikipedia.org/wiki/F1_score)
