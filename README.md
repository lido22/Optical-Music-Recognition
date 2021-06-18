# OMR
NOTE: "The project got the highest grade of 18 other projects."
## Used Algorithms

### Deskewing

First, we detect the edges using canny, and dilate them. Then we get the contours
and sort them by area in descending order. And then we start merging contours
whose area are large enough to satisfy certain conditions. After that we get
minAreaRect() and process the output to get the four corners that represent our
area of interest. Then we do perspective transform to map the four corners to be
our new image corners after determining the new dimensions.

After that we redo the previous step but with some changes to deskew the image.
At this point we expected to have a fully deskewed image, If the image is near 90
degrees the becomes vertical. So, we detected the image inclination angel using
Hough transform and rotated it to git rid of that inclination. At the end we repeat
the first step again and get area of interest again to make sure that the output is
correct.

If the image is inclined with an angle more than 90 degrees the output will be
inverted. So, we use SIFT matcher to see if the clef is in the left or right side. If it is
in the right side, we rotate the image by 180 degrees.

### Binarization

We divided the image into small windows and then we calculate the average of the
window. The pixels whose value less than the average with predefined percentage
is considered as black. Otherwise, it is white.

As an enhancement to the proposed algorithm, we used the method of integral
image proposed in [1].

We also tried the method proposed in [2], it is made specifically for the binarization
of music scores based on extraction of _stafflineHeight_ and _staffspaceHeight_ of the
image. But it did not work out so well in images in which there is a vertical change
in illumination because it chooses on threshold for each column or bunch of
columns.


### Stafflines Removal

In order to detect stafflines and remove them form the image, we first need to
know the _stafflineHeight_ and the _staffspaceHeight._ To extract them we run through
the image column by column and calculate the run length of black and white runs
and then we took the most recurring black run as the _stafflineHeight_ and the most
recurring white run as the _staffspaceHeight_.

Although this method was able to calculate the lengths very well but we noticed
that when applying s&p noise the returned value becomes almost 1 for both
lengths so we switched for the method proposed in [3]. It calculates the most
recurring sum of two consecutive black and white runs. Figure (1) explains how the
lengths are chosen.

There was a more accurate method in which we calculate the lengths form the gray
image, but it is too slow and we found it as an overkill.


![image](https://user-images.githubusercontent.com/42592954/122621370-9d23b680-d095-11eb-9832-aa7cc4bb5173.png)

```
figure 1
```

After calculating the lengths, we tried many approaches. The horizontal projections
approach used in [4] works well on scanned images only. Hough transform was
good with inclinations and discontinuity but not good in presence of curvatures.
The method used in [5] depends on selection of candidate points and then joining
them by using DP was so good with curvatures but it was so slow. Our chosen
method is similar to the previous one but much simpler. It is based on that
mentioned in [6] but with improvement in speed and accuracy.

We run through the image column by column and any black run with length in the
range of [ _stafflineHeight-2, stafflineHeight+2]_ is considered to be a candidate staff
line segment.

Then, we filter the set of candidates to so we only leave the segments that have
vertical neighbors in the range of distance [spaceHeight+staffHeight:
spaceHeight+2*staffHeight].

In this way our algorithm is much faster than the proposed in the paper as we
skipped two major steps the we found unnecessary. And is less prone to detecting
parts of the image as staff line segments as our beforementioned removal
condition is more accurate and stricter than the one mentioned in the paper.

There is a powerful and elegant method [7,8,9] that we tried but did not use since
it is too slow and we didn’t have time to create a C++ plugin. It deals with the image
as a weighted graph in which the edges which has at least black end has lower
weight than the edge that connects two white pixels. The algorithm then finds the
shortest path using Dijkstra algorithm as in [7], or using DP as in [8]. Then it trims
the ends of the found path and the remaining part is the staff line. The method
used in [7] was faster but less accurate than [8], however both of them were pretty
amazing and we intend to make a C++ plugin for that algorithm in the future.


### Segmentation

After removing the lines, we split the image into staves by the following approach:
First we create a vector which calculates the row numbers that has a staff line. Each
staff line is represented using on reference row. Then we consider the mid-points
between each two rows if the distance between them is bigger than 5 * (space
Height).

We then use OpenCV’s findContours() to separate the symbols then we does some
postprocessing in which we discard small contours that wrap deformities resulted
from removing lines and binarization. We join disjointed elements based on the
conditions mentioned in [10]. If the height of the bounding box of one of two close
fragments is greater than its width and its height is smaller than staffspaceHeight,
the two fragments are merged.

### Recognition

All segmented symbols are passed to the classifier _described in README-
TRAIN.md_ and then the output is passed to semantic construction functions.

The NN was trained on HoG features of symbols with testing error equal to 98%.

If the contour is wide and it is not clef, we conclude that it is a beam. The classifier
often misclassified quarter notes as half notes and dots as beams. So, we did a
postprocessing step to handle each of them.

The later one was easy, we only check the contour height. In the he former one we
calculate number of black pixels in each column and there is at least one column
with value in range (0.5*spaceHeight, 2*spaceHeight) it is quarter note. Otherwise,
it is half note.

### Semantic Construction

Each class of symbols is dealt with in specific way. In dealing with beamed notes,
we used an improved method based on the one proposed in [11]. We first do
morphological opening with horizontal kernel to eliminate the stem then, we use a
horizontally running thin vertical window and save the length, start and end of the
tallest black run which represents the note head. Then we use a threshold to filter


the offset resulted from the beam. We then extract each maximum from every
group and take its start and end as the start and end of the note head.

![image](https://user-images.githubusercontent.com/42592954/122621392-b0cf1d00-d095-11eb-842a-43c80fe9b554.png)

```
Figure (2)
```
To determine the number of beams we use a similar approach to the one used in
stafflineHeight and staffspaceHeight estimation.

In dealing with chords, we get the x-projection and if the peak is in the middle, we
conclude that there are note heads on both side if the stem. Then we remove the
stem in the same way we did with beamed notes. We then determine the number
of note heads based on the height of the remaining shape. After that, we determine
the position of each note and return the sorted alphabetically.

Other notes are done using a series of y & x projections.


## Accuracy & Performance

The total runtime of all 32 images is: 110 seconds

Testing: 01.txt
Accuracy 99.24812030075188%
_____________________________________________________________________
Testing: 02.txt
Accuracy 90.19607843137256%
_____________________________________________________________________
Testing: 03.txt
Accuracy 100.0%
_____________________________________________________________________
Testing: 04.txt
Accuracy 99.3006993006993%
_____________________________________________________________________
Testing: 05.txt
Accuracy 98.46153846153847%
_____________________________________________________________________
Testing: 06.txt
Accuracy 97.43589743589743%
_____________________________________________________________________
Testing: 07.txt
Accuracy 98.63013698630137%
_____________________________________________________________________
Testing: 08.txt
Accuracy 91.30434782608695%
_____________________________________________________________________
Testing: 09.txt
Accuracy 98.98989898989899%
_____________________________________________________________________
Testing: 10.txt
Accuracy 97.67441860465115%
_____________________________________________________________________
Testing: 11.txt
Accuracy 72.54901960784314%
_____________________________________________________________________
Testing: 12.txt
Accuracy 100.0%
_____________________________________________________________________
Testing: 13.txt
Accuracy 100.0%
_____________________________________________________________________
Testing: 14.txt
Accuracy 100.0%
_____________________________________________________________________
Testing: 15.txt
Accuracy 100.0%
_____________________________________________________________________



## Conclusion

In this paper we walked through the steps of our OMR system. The system is fast
and the results were fascinating on the scanned image and camera capture
printed images the results were not so well on the handwritten camera captured
images. This is due to the various deformations occur to the image in binarization,
rotation and segmentation steps. Further work could be done to improve the
preprocessing steps and the whole result.


## References

1. Adaptive Thresholding Using the Integral Image by Derek Bradley and Gerhard Roth
2. Music Score Binarization Based on Domain Knowledge _by_ Telmo Pinto, Ana Rebelo,
    Gilson Giraldi, and Jaime S. Cardoso
3. Robust staffline thickness and distance estimation in binary and gray-level music scores
    _by_ Jaime S. Cardoso, Ana Rebelo
4. Sheet Music Reader _by_ Sevy Harris, Prateek Verma
5. Stave Extraction for Printed Music Scores _by_ Hidetoshi Miyao
6. An Efficient Staff Removal Approach from Printed Musical Documents _by_ Anjan Dutta
7. A Shortest Path Approach for Staff Line Detection by authers of 8
8. A CONNECTED PATH APPROACH FOR STAFF DETECTION ON A MUSIC SCORE _by_ Jaime S.
    Cardoso, Artur Capela, Ana Rebelo Carlos Guedes
9. STAFF LINE DETECTION AND REMOVAL WITH STABLE PATHS by authers of 8
10. AN OPTICAL MUSIC RECOGNITION SYSTEM FOR SKEW OR INVERTED MUSICAL SCORES
    _by_ YUNG-SHENG CHEN, FENG-SHENG CHEN, CHIN-HUNG TENG
11. Optical Music Sheet Segmentation _by_ P.Nesi


