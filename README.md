# LaneDetection
### The first project in the course "AI is Math" by Yoni Chechik.

In this project we process three videos, mark the lanes on each frame, check for lane switching, and in the relevant video also mark the cars around and check for dangers.

## Lane Detection and lane switching process:
In general, the process uses image-processing techniques:

_An original frame:_ <br>
<img src="Examples/frame.png" alt="frame" width="276">

_Applying gray scale and gaussian blur:_ <br>
<img src="Examples/gray+blur.png" alt="gray+blur" width="276">

_Applying canny:_ <br>
<img src="Examples/canny.png" alt="canny" width="276">

_Leaving only edges inside a region of interest:_ <br>
<img src="Examples/RegionOfInterest.png" alt="RegionOfInterest" width="276">

The algorithm applies the Hough Transform to detect linear segments from masked edges, representing potential lane markings. It filter these based on slope, length, and position to identify the most probable left and right lane lines.

_Result frame of lane marking:_ <br>
<img src="Examples/final.png" alt="final" width="276">

The algorithm also detects lane switching by assessing if detected lanes are near the middle of the lane.

_Result frames for frames with a lane switch:_ <br>
<img src="Examples/switching1.png" alt="switching1" width="276"> &nbsp;
<img src="Examples/switching2.png" alt="switching2" width="276">

## Night-time Lane Detection:
Our algorithm works well with a night video as well.

_Result frames of lane marking at night:_ <br>
<img src="Examples/night.png" alt="night" width="276"> &nbsp;
<img src="Examples/night2.png" alt="night2" width="276">

_Result frame for a frame with a lane switch at night:_ <br>
<img src="Examples/nightSwitch.png" alt="nightSwitch" width="276">

## Proximity-Based Vehicle Detection for Collision Avoidance:
We used template matching to identify vehicles based on visual similarity and assessed danger risks based on proximity. 

_Templates examples:_ <br>
<img src="Templates/Car2.png" alt="Car2" width="82"> &nbsp;
<img src="Templates/Car3.png" alt="Car3" width="70"> &nbsp;
<img src="Templates/Car5.png" alt="Car5" width="30"> &nbsp;
<img src="Templates/Car6.png" alt="Car6" width="110">

The code uses cv2.matchTemplate() to scan each frame with vehicle templates of varying sizes, adding bounding rectangles around detected vehicles while ensuring no overlap between detections. Rectangles within a predefined danger zone are marked in red and accompanied by a warning message, and others are marked in blue.

_A visualization of the danger zone drawn on a frame from our video:_ <br>
<img src="Examples/danger.png" alt="danger" width="276">

_Result frames for the car detection video:_ <br>
<img src="Examples/crashOutput.png" alt="crashOutput" width="276"> &nbsp;
<img src="Examples/crashOutput2.png" alt="crashOutput2" width="276">

### In the repository, there is the full Python code and the original + output videos.
#### The full code report is inside the pdf named Project1_report.