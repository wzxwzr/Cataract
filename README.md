# Cataract
JHU image processing of capsulorhexis surgery

Here is an overview of the algorithm used

1. blur image
2. convert to hsv
3. threshold hsv to isolate orange pupil
4. perform canny edge detection on the thresholded image
5. detect the 10 best circles in the canny image
6. predict the expected location and dimensions of the pupil based on previous frames
7. chose the circle which best fits the prediction, this is the pupil
8. mask out everything outside 113% of the pupil
5. perform OTSU threshold on blurred image from 1.
6. find the contours within this image
6. perform canny edge detection on the thresholded image
7.  mask out everything outside 113% of the pupil
8.  perform line detection on the two cannys
9.  filter lines which are too close to the circumference of the circle
10.  filter lines which have the wrong slope, based on prediction as well as overall flow of lines
11.  filter lines which have the wrong `pseudo y intercept`, which is where the line would intersect the verticle axis 50 pixels to the left of the centre of the eye
12.  compute ( lines1 ∪ lines2 ) ∩ contours, this is the tool
13.  find the leftmost part of the tool, this is the tip

Notes/TODO:

+ `predict_math.py` needs to be updated so that if the previous frames had no detection, it returns `None`
+ Currently, it doesn't store frames without detected circles or tools. It should store a null result
+ Should add hsv thresholding to find the tool, can be done via subtracting the pupil from the original frame and masking out everything outside 113% of the pupil. See `todo.py` for sample implementation
+ The program doesn't deal with multiple tools very well, which shows during the video section involving tweezers rather than a scalpel. Could update the tool storage container to allow multiple tools. Alternatively, just detect when the section with one tool is finished and increase `angle_variance` and `y_int_variance` accordingly
