**Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./visualization.png "Visualization"
[image2]: ./normalization.png "Normalization"
[image9]: ./original.png "Original"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./traffic-signs-data/0.jpg "Traffic Sign 1"
[image5]: ./traffic-signs-data/1.jpg "Traffic Sign 2"
[image6]: ./traffic-signs-data/2.jpg "Traffic Sign 3"
[image7]: ./traffic-signs-data/3.jpg "Traffic Sign 4"
[image8]: ./traffic-signs-data/4.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/bwuzhang/CarND-Traffic-Sign-Classifier-Project)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32x32x3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among different traffic sign classes.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

My image preprocessing only includes normalization the images. This removes the difference of brightness of the scene at where the image was captures. It made the training process more robust.

I didn't convert the images to gray scale because that would reduce the amount of information contained in the image. Another reason is that color is a strong indicator of what the traffic sign is.

Here is an example of a traffic sign image before and after normalization.

![alt text][image9]
![alt text][image2]

As a last step, I resized the image to 32x32 because my network requires a fix size input image.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3  + Relu   	| 1x1 stride, valid padding, outputs 30x30x6 	|
| Convolution 3x3  + Relu   	| 1x1 stride, valid padding, outputs 28x28x16 	|
| Convolution 3x3  + Relu   	| 1x1 stride, valid padding, outputs 26x26x26 	|
| Convolution 3x3  + Relu   	| 1x1 stride, valid padding, outputs 24x24x36 	|
| Max pooling	      	| 2x2 stride,  outputs 12x12x36 				|
| Convolution 3x3  + Relu   	| 1x1 stride, valid padding, outputs 10x10x56 	|
| Convolution 3x3  + Relu   	| 1x1 stride, valid padding, outputs 8x8x66 	|
| Convolution 3x3  + Relu   	| 1x1 stride, valid padding, outputs 6x6x90 	|
| Convolution 3x3  + Relu   	| 1x1 stride, valid padding, outputs 4x4x120 	|
| flatten	      	| outputs 1920 				|
| Fully connected	+ Relu	| outputs 240      									|
| Fully connected	+ Relu	| outputs 120      									|
| Fully connected	| outputs 43      									|
| Softmax				| outputs 43         									|



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer, with batch size of 128 and learning rate of 0.001. I trained it for 5 epochs.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I trained the network and record training accuracy and validation accuracy at the end of every epoch. I stopped the training when the validation accuracy started to saturate.

The network was designed by myself. I used 8 convolution layers with only 1 pooling layer. The size of the feature map is gradually decreased by 2 after each convolution due to the 'valid' convolution operation. Polling is only used once in the middle to reduce the number of parameters of the network. Since the input is already small (32x32), pooling should be used as little as possible. 3 fully connected layers are used at the end to regress to the final probabilities.

My final model results were:
* training set accuracy of 99.1%
* validation set accuracy of 95.3%
* test set accuracy of 92.5%

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The last image might be difficult to classify because it contains large watermarks.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Priority road      		| Priority road   									|
| Speed limit (120km/h)      			| Speed limit (30km/h)  										|
| Speed limit (60km/h) 					| Speed limit (50km/h) 											|
| Stop 	      		| Stop 					 				|
| Bicycles crossing			| Bicycles crossing      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. The two signs it misclassified are all speed limit signs. The reason might be that these signs share a huge amount of similarities on the shape and color, so it is naturally hard to distinguish them among each other comparing to other signs.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first, fourth and fifth images, the model is very sure about its prediction and it turns out they are correct.

For the second image, it is only 36.9 percents sure it's a 30km/h speed limit sign, and it is wrong.

For the third image, it has over 99.9% sure about its prediction and it turns out its wrong.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .999         			| Priority road   									|
| .369     				| Speed limit (30km/h) 										|
| .999					| Speed limit (50km/h)											|
| .999	      			| Stop					 				|
| .995				    | Bicycles crossing      							|
