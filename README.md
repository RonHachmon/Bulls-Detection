# Transfer Learning for Bull Detection using SSD

This project demonstrates how to use transfer learning to train an SSD (Single Shot Detector) model to detect bulls in images.
The project uses the pytorch detection API and the pre-trained SSD-MobileNet V2 model as a base for transfer learning.
Dataset

The dataset used for this project is a collection of images of bulls from various sources. The images are labeled with bounding boxes around the bulls using the CSV format.
The dataset is split into a training set and a validation set.
Requirements

To run this project, you need to have the following software installed:

    Python 3.6 or higher
    pytorch 
    pandas


## Getting Started

To get started with this project, follow these steps:

    1. Clone this repository to your local machine.
    2. Download the a dataset from open images  and extract it to a directory.
    3. Install the required software listed in the requirements section.
    4. adjust the path and labels
    5. run the program

    Use the video_detector.py script to test the model on new images.

## Results

The trained model achieved an average precision of 0.95 on the validation set. The model can detect bulls in various poses and orientations with high accuracy.


<img src="/media/test_video.gif" height="250"/> <img src="/media/result_5.jpg" height="250" />








