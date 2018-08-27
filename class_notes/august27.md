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

#### RANSAC
**Random Sample Concensus**
* Randomly choose _s_ samples
* Fit them with a line
* Give a threshold that provides boundaries that transformed along y axis
* Count how many points fall between those lines
* Find the perfect line
* **Can provably show how many rounds you need to do to get optimal outcome with given confidence**
* However, you do need to know the distribution of your dataset in order for the _confidence_ to work

### Connected-component Labeling
* Connected-component analysis, distingishing between different objects
* OpenCV Connected Component Labeling
#### Two-Pass Algorithm
1. First pass - give all neighbors the same number in your first pass
2. Second pass - if there is a group that is connected by given different number, you make them the same


