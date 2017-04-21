#**Traffic Sign Recognition**

##Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1-1]: ./visualization1.jpg "Visualization1"
[image1-2]: ./visualization2.jpg "Visualization2"
[image1-3]: ./visualization3.jpg "Visualization3"
[image2]: ./rgb.jpg "RGB"
[image3-1]: ./grayscale.jpg "Grayscaling"
[image3-2]: ./equalizer.jpg "Equalizing"
[image4]: ./sign1.jpg "Traffic Sign 1"
[image5]: ./sign2.jpg "Traffic Sign 2"
[image6]: ./sign3.jpg "Traffic Sign 3"
[image7]: ./sign4.jpg "Traffic Sign 4"
[image8]: ./sign5.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/smashkoala/Traffic-sign-recognition-project)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data, the validation
data and the test data are consisted.

![alt text][image1-1]
![alt text][image1-2]
![alt text][image1-3]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the validation accuracy did not go up with RGB.

Here is an example of a traffic sign image before and after grayscaling.  
Before gray scaling  
![alt text][image2]  
After gray scaling  
![alt text][image3-1]  

As a last step, I used histogram equalizer because I thought that contrast of some images are not sufficient. Some looked quite dark.  
After exposure equalizing  
![alt text][image3-2]

Before feeding the data to the network, I made the ratio of number of images in each class the same between the training data and the test data set.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   				    |
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x24 	|
| RELU					| 											    |
| Max pooling	      	| 2x2 stride,  outputs 14x14x24 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x48    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x48 				    |
| Fully connected		| input = 1200 output = 120       			    |
| RELU					|												|
| Fully connected		| input = 120 output = 84       				|
| RELU					|												|
| Softmax				|          		                                |



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following optimizer and hyperparameters:  
* Optimizer: Adam optimizer  
* Learning rate: 0.005  
* Number of EPOCS: 30  
* Number of batches: 64  

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.958
* test set accuracy of 0.935  

The training set accuracy and the validation accuracy are calculated in the 15th cell of the Ipython notebook. The test set accuracy is calculated in the 17th cell of the notebook.



If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?  
A. The first architecture chosen was the same LeNet-5 as in the course, because I thought it works for any images. In addition, both of them process the similar size of input images.

* What were some problems with the initial architecture?  
A. The accuracy of the training and validation set did not go up as desired. In addition, the network was over fitting. I could not get the desired result after so many trial and errors. In the end, overfitting was not improved so much, but the both training accuracy and the validation accuracy went up together after the depth of the convolution layers are changed.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.  
A. With the initial architecture, the network was overfitting. In order to improve it, I added a dropout to each fully connected layer. However, the overfitting was not improved. Both training and validation accuracy were not improved either.
In the end, I tried to increase the depth of the convolution layers, and that improved the accuracy of both training and validation a lot, although the overfitting was not improved so much.  

* Which parameters were tuned? How were they adjusted and why?  
A. In the end, increasing the depth of the convolution layers helped me improve the validation accuracy. Before changing the depth, I mainly tuned learning rate, number of EPOCS, number of batches but they did not give the desired accuracy.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?  
A. Well, I am not sure if I made design choices deliberately, since I followed the instruction of the step2 in the jupyter notebook.
One design change I made from the LeNet-5 was the depth of the convolution layer. I increased the depth significantly because I thought that the accuracy did not go up because the amount of the data processed from the 1st convolution layer until the last fully connected layers was not enough.

If a well known architecture was chosen:
I am not sure if the following information is require in my case, but I still wrote them as below, since my architecture is based on LeNet-5.
* What architecture was chosen?  
A. LeNet-5 was chosen although the depth of the convolution layers is different.
* Why did you believe it would be relevant to the traffic sign application?
A. Because that worked also for image recognition, and the size of the images are similar between this traffic sign project and the LeNet lab project.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?  
A. The accuracy of the training, the validation and the test set,  all of them reached over 93%.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The fourth image of a slippery road sign might be difficult to classify because the actual image of this is reshaped 32x32 by shrinking width instead of cropping the center of it.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Right-of-way          | Right-of-way   								|
| 20 km/h     			| 20 km/h 										|
| Roundabout			| Roundabout									|
| Slippery Road			| Right-of-way      							|
| Priority road	      	| Priority road					 				|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 21st cell of the Ipython notebook.

For the first image, the model is sure that this is a right-of-way sign (probability of 1.00), and the image does contain a right-of-way sign. The top five soft max probabilities were as follows:  

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Right-of-way   								|
| 2.53e-12     			| Beware of ice/snow 							|
| 1.59e-12				| Pedestrians									|
| 5.59e-13	      		| Road narrows on the right					 	|
| 3.36e-16				| General caution      							|

For the second image, the model is sure that this is a 20 km/h sign (probability of 0.99), and the image does contain a 20 km/h sign. The top five soft max probabilities were as follows:  

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 9.99e-01         		| 20 km/h   								    |
| 5.70e-07     			| 70 km/h 							            |
| 3.22e-10				| 30 km/h									    |
| 4.38e-15	      		| 120 km/h					 	                |
| 1.43e-17				| Go straight or left      						|

For the third image, the model is sure that this is a roundabout sign (probability of 0.99), and the image does contain a roundabout sign. The top five soft max probabilities were as follows:  

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 9.99e-01         		| Roundabout mandatory   						|
| 4.03e-07     			| Priority road 							    |
| 4.00e-08				| 30 km/h									    |
| 4.25e-10	      		| Go straight or left					 	    |
| 7.82e-13				| Right-of-way      						    |

For the fourth image, the model is sure that this is a right-of-way sign (probability of 0.94), however the image contains slippery road sign. This is probably because the model cannot detect the image inside of the diagonal. Compared to other images, the probability of the 2nd and 3rd predictions are relatively high. In addition, this may be because the number of training samples of slippery roads are relatively small. The top five soft max probabilities were as follows:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 9.94e-01         		| Right-of-way   						        |
| 4.54e-03     			| Pedestrians 							        |
| 7.15e-04				| General caution							    |
| 3.62e-04	      		| Double curve					 	            |
| 2.42e-05				| Roundabout mandatory      				    |

For the fifth image, the model is sure that this is a priority road sign (probability of 1.00), and the image does contain a priority road sign. The top five soft max probabilities were as follows:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Priority road   						        |
| 3.27e-12     			| Ahead only							        |
| 2.65e-12				| Roundabout mandatory							|
| 1.44e-13	      		| No entry					 	                |
| 1.36e-13			    | Stop      				                    |
