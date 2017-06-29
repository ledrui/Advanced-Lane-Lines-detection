##Writeup
---

**Advanced Lane Finding Project**
Given a video of a car driving in its lane, the program identifies the lane and draws it onto the image. Output video is output.mp4

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/lanesDrawn.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[undistordistors]: ./output_images/undistorted_test_img.jpg "Undistorted_example"
[treshold]: ./output_images/threshold.jpg "Fit Visual"
[mask]: ./output_images/4mask.jpg "Mask"
[slidingWindow]:./output_images/sliding_indow.jpg "sliding windows"
[slidingWindow2]:./output_images/SlidingWindowStep.jpg "sliding windows 2"
[slidingWindow3]:./output_images/Stepthrough.jpg "sliding windows step through"

---

###Writeup / README

###Methodology

###Camera Calibration

The code for this step is contained in the code cell 4 of the IPython notebook located in "Advanced-Lane-Lines.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Example of a distortion-corrected image.

Given these corners, the distortion matrix and distances are calculated using opencv and used to undistort images taken from this camera. The function `get_undistord()` in the cell 3 of the notebook applies the undistortion.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

After undistortion

![alt text][undistordistors]

####2. Combine Color and gradient thresholds for lane detection
I created a combined binary image using color and gradient thresholding 
I used a combination of color and gradient thresholds to generate a binary image in cell 6.  Here's an example of my output for this step.

![alt text][treshold]

####3. Perspective transform

The code for my perspective transform includes a function called `warp()`, which appears in the 10th code cell of the IPython notebook).  The `warp()` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points in the following manner:
```
src = np.float32(
    [[681, 444],
     [1200, imshape[0]],
     [598, 444],
     [200, imshape[0]]])

dst = np.float32(
    [[975, (imshape[0] - imshape[0])],
     [975, imshape[0]],
     [300, (imshape[0] - imshape[0])],
     [300, imshape[0]]])
```
The code can be seen in cell 9 of the notebook

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4  I applied a polygonal Mask after the binary image
the code for applying the mask can be found in cell 12
here you can see the binary-masked image:

![alt text][mask]

####5. Identifying lane-line pixels and fit their positions with a polynomial

To identify the lane-line I applied the sliding windows technique (the code can found in the cell 19 of the notebook) and I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

![alt text][slidingWindow]

![alt text][slidingWindow2]

![alt text][slidingWindow3]


####6. Curvature of the lane and the position of the vehicle with respect to center.

I did this in cell 15 in the notebook 

####7. Example image of my result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the cell 21 in the function `pipeline()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to my final video output.  

Here's a [link to my video result](https://youtu.be/BSKNAGcw6Sw)

---

###Discussion

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

Most of thechniques I've used here are from OpenCV and lot of class lesson code as well
 
I could improve it by averaging several past lane curvatures. Additionally, instead of sliding windows I could use convolutions to determine which pixels are part of the left and right lanes. Finally, this has not been tested with night-time or rainy condition videos, and those may blur the lines to the point where the current model would not work well.

