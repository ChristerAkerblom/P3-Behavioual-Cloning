
# Project goals

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Model Visualization"
[image3]: ./examples/leftmiddle1.jpg "Recovery Image"
[image4]: ./examples/leftmiddle2.jpg "Recovery Image"
[image6]: ./examples/normal.jpg "Normal Image"
[image7]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the result
* video.mp4 of vehicle completing autonomuous driving of the track

#### 2. Code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model architecture overview

The data were is normalized in the model using a Keras lambda layer and cropped in order to remove non-relevant infomration in the top and bottom of the images.
My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 64. The model includes RELU layers to introduce nonlinearity.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving in both directions.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use well known architectures from the Udacity course.

My first step was to use a convolution neural network model similar to the AlexNet I thought this model might be appropriate because it is a CNN suitable for images and had worked well for Traffic signs. It was not that succesful and I soon changed to the model proposed by Nvidia.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added dropouts between the layers but also additional data to the training of the model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track epecially in the sharp curves and at the bridge. To improve the driving behavior in these cases, I collected more data for theese scenarios.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes.

|  **Layer**          |  **Specification**|
|---------------------|-------------------|
|   Normalization     |   Normalize images to values between (-0.5,0.5)                              |
|   Image Cropping    |   70 rows at the top and 20 rows in the bottom of the picture was removed    |
|   Convolution       |   5x5 filter with 24 channels out, strides (2,2)                             |
|   Dropout           |   dropout rate 0.2                                                           |
|   Convolution       |   5x5 filter with 36 channels out, strides (2,2)                             |
|   Dropout           |   dropout rate 0.2                                                           |
|   Convolution       |   5x5 filter with 48 channels out, strides (2,2)                             |
|   Dropout           |   dropout rate 0.2                                                           |
|   Convolution       |   3x3 filter with 64 channels out                                            |
|   Convolution       |   3x3 filter with 64 channels out                                            |
|   Fully connected   |   Output size 100                                                            |
|   Fully connected   |   Output size 50                                                             |
|   Fully connected   |   Output size 10                                                             |
|   Fully connected   |   Output size 1                                                              |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the middle of the lane when beeing close to the side of the road. These images show what a recovery looks like starting from the left side of the road:

![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would improve generalisation. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 29k number of data points. I then preprocessed this data by noramlizing the images to range (-0.5,0.5).


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by a number of test with more epochs where validation error increased after 4-5 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
