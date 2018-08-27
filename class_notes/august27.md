# Computer Vision Intro
#### Dilation
We use disk kernel that looks like

[[0, 1, 0],
 [1, 1, 1],
 [0, 1, 0]]


#### Hough Transform
* Performed after Edge Detection
* Technique to isolate the curves of a given shape in a given image
* Can locate regular curves like straight lines, circles, parabolas, ellipses, etc
* **Edges vote for object detection**
* Tolerant of gaps
* Relatively unaffected by noise
* _Given examples of iris and roads to demonstrate Hough Transform_
* The algorithm votes for the y=mx+n that fits the given data the best
* OpenCV offers functions HoughLines(), HoughCircle()

`Will this create a box?? - Oh yes`
