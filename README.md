# Project Road Segmentation

For this project, we were given a set of satellite images acquired from GoogleMaps and ground-truth images where each pixel is labeled as road (1) or background (0). 
Our task is to train a classifier to segment roads in these images, i.e. assigns a label `road=1, background=0` to each pixel.

## Repo setup:

1. The dataset is available from the 
[AICrowd page](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation).

2. The first method we tried to implement is a classic UNet, then we tried to implement the MARESUNET from the paper [Multi-stage Attention ResU-Net for Semantic Segmentation of Fine-Resolution Remote Sensing Images](https://github.com/lironui/MAResU-Net) with modifications to suit our problem, and finally the ResNet34 from [Pytorch](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html). All the different models are available in script/models.py

3. In order to evaluate our solutions we used our own implementation available in script/metrics.py

4. Finally, the file run.py lets a chosen model (between the 3 introduced above) train and plot some results. Use model_1 as model_factory for UNet, model_2 for MAResUNet and model_3 for ResNet. Using 300 epochs will give the best performances for the MAResUNet, and the best performances we had for this project. 
