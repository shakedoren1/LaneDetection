import cv2
import numpy as np
import time

# Lane detection global variables
rightLaneGlobal = [-1, 0, 0, 0, 0]
leftLaneGlobal = [-1, 0, 0, 0, 0]
glabalLanesCounter = 0
polygonParameters = [0, 0, 0, 0, 0, 0, 0, 0]
lanePolygon = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.int32)
laneLength = 0

# Lane switching global variables
switching = None
timer = 0

# Cars + collision detection global variables
dangerPolygon = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.int32)
allTemplates = []
templateStrings = []
for i in range(14):
    templateStrings.append("LaneDetection/Templates/Car" + str(i) + '.png')
 
# Load the templates in different sizes - for better detection
for template in templateStrings:
        template = cv2.imread(template)
        for i in range(0, 51, 5):
           allTemplates.append(cv2.resize(template, (int(template.shape[1]*(1-i/100)), int(template.shape[0]*(1-i/100))), interpolation = cv2.INTER_AREA))

# Function to create the lane detection polygon
def SelectLanePolygon(frame):
    frameHeight, frameWidth = frame.shape[:2]
    global lanePolygon
    # Define the lane polygon vertices
    lanePolygon = np.array([[
        (frameWidth * polygonParameters[0], frameHeight * polygonParameters[1]),
        (frameWidth * polygonParameters[2], frameHeight * polygonParameters[3]),
        (frameWidth * polygonParameters[4], frameHeight * polygonParameters[5]),
        (frameWidth * polygonParameters[6], frameHeight * polygonParameters[7]),
    ]], np.int32)
    
# Get the lines in the region of interest (in the lane polygon)
def RegionOfInterest(edgesImages):
    mask = np.zeros_like(edgesImages)
    # Fill the mask according to the polygon
    cv2.fillPoly(mask, lanePolygon, 255)
    # Apply the mask to the edges image
    maskedImage = cv2.bitwise_and(edgesImages, mask)
    return maskedImage

# Function to create the danger polygon
def SelectDangerZone(frame):
    frameHeight, frameWidth = frame.shape[:2]
    global dangerPolygon
    # Define the danger polygon vertices
    dangerPolygon = np.array([[
        (frameWidth * 0.43, frameHeight * 0.53),
        (frameWidth * 0.49, frameHeight * 0.53),
        (frameWidth * 0.68, frameHeight),
        (frameWidth * 0.13, frameHeight),
    ]], np.int32)

# Function to calculate the lane boundaries from a partial part of the lane (represented by two points)
def CalculateLaneBoundaries(x1, y1, x2, y2, frameHeight):
    # Calculate the slope - m
    m = (x2-x1)/(y2-y1)
    # Calculate the y-intercept - n
    n = x1 - m*y1
    # Return the lane boundaries represented by the two edge points
    return(int(frameHeight*m + n), int(frameHeight), int(frameHeight*laneLength*m + n), int(frameHeight*laneLength))

# Function to calculate the L2 distance between two points
def Distance(coordinates):
    x1, y1, x2, y2 = coordinates
    return np.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))

# Function to find the best left line that represents the left lane from the filtered lines
def BestLeftLine(linesAfterFilter, frameWidth):
    bestLeftLine = [-1, 0, 0, 0]
    # The best left line is the first long line with a negative slope
    linesAfterFilter = sorted(linesAfterFilter, key=lambda x:x[0][0])
    for line in linesAfterFilter:
        x1, y1, x2, y2 = line[0]
        # Look at lines in the left half of the frame
        if (x1 < frameWidth // 2 or x2 < frameWidth // 2) and (x2-x1)/(y2-y1) < 0:
            if Distance(line[0]) > 85:
                return line[0]
            
    return bestLeftLine

# Function to find the best right line that represents the right lane from the filtered lines
def BestRightLine(linesAfterFilter, frameWidth):
    bestRightLine = [-1, 0, 0, 0]
    # The best right line is the one with the longest length
    for line in linesAfterFilter:
        x1, y1, x2, y2 = line[0]
        # Look at lines in the right half of the frame
        if(x1 > frameWidth // 2 or x2 > frameWidth // 2):
            if Distance(line[0]) > Distance(bestRightLine):
                bestRightLine = line[0]

    return bestRightLine

# Function to filter the lines in the frame, and return the best left lane and right lane
def FilterLines(lines, frameWidth):
    linesAfterInitialFilter = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if(y1 == y2):
            continue
        elif(np.abs((x2-x1)/(y2-y1)) > 2):
            continue
        # Lines after initial filter are lines with appropriate slopes
        linesAfterInitialFilter.append(line)

    # Find the best left and right lanes in this frame
    LeftLane = BestLeftLine(linesAfterInitialFilter, frameWidth)
    RightLane = BestRightLine(linesAfterInitialFilter, frameWidth)

    global leftLaneGlobal
    global rightLaneGlobal
    if(LeftLane[0] == -1):
        # If there's no sufficent left lane detected in the frame, check if there's still a global left lane from previous frames
        if(leftLaneGlobal[4] > 0):
            # If there's a global left lane, take it and decrease the global counter
            leftLaneGlobal[4] = leftLaneGlobal[4] - 1
            LeftLane = leftLaneGlobal[0:4]
    else:
        # Set the global left lane to the current left lane and set its counter to glabalLanesCounter
        leftLaneGlobal = np.append(LeftLane, glabalLanesCounter)
      
    if(RightLane[0] == -1):
        # If there's no sufficent right lane detected in the frame, check if there's still a global right lane from previous frames
        if(rightLaneGlobal[4] > 0):
            # If there's a global right lane, take it and decrease the global counter
            rightLaneGlobal[4] = rightLaneGlobal[4] - 1
            RightLane = rightLaneGlobal[0:4]
    else:
        # Set the global right lane to the current right lane and set its counter to glabalLanesCounter
        rightLaneGlobal = np.append(RightLane, glabalLanesCounter)
    
    # Return the best left and right lanes if they exist 
    if(LeftLane[0] != -1 and RightLane[0] != -1):
        return [[LeftLane], [RightLane]]
    elif(LeftLane[0] == -1 and RightLane[0] != -1):
        return [[RightLane]]
    elif(LeftLane[0] != -1 and RightLane[0] == -1):
        return [[LeftLane]]
    else:
        return []

# Function to detect lane switching in the video and display a message accordingly
def CheckForLaneSwitching(imageWidth, x1, y1, x2, y2):
    global switching, timer
    if not switching:
        # If we are not currently in a switching state, check if this a beginning of a lane switch
        middle_x = imageWidth / 2
        if ((x1 - 25 < middle_x) and (x2 + 25 > middle_x)) or (
            (x1 + 25 > middle_x) and (x2 - 25 < middle_x)):
            # The line is close to the middle of the frame - which indicates a lane switch!
            timer = time.time()  # Start the timer for this lane switch
            m = (x2-x1) / (y2-y1) # calculate the slope to indicate the switching direction
            switching = "Right" if m > 0 else "Left"
    else:
        # If we are in a switching state, check the time in order to reset when needed
        if time.time() - timer > 4:  # 4 seconds have passed
            # Reset the current switching state
            switching = None  
            timer = 0                  

# Function to draw the lane image
def DisplayLines(image, lines):
    lineImage = np.zeros_like(image)
    # Ensure that the lines are not None
    if lines is None:
        return lineImage
    # Filter the lines and get the best left and right lanes
    linesAfterFilter = FilterLines(lines, image.shape[1])
    for line in linesAfterFilter:
        x1, y1, x2, y2 = line[0]
        # Calculate the lane boundaries to get the lane edges
        x1, y1, x2, y2 = CalculateLaneBoundaries(x1, y1, x2, y2, image.shape[0])
        # Check for lane switching according to the founded lane
        CheckForLaneSwitching(image.shape[1], x1, y1, x2, y2)
        # Draw the lane line in the line image
        cv2.line(lineImage, (x1, y1), (x2, y2), (255, 0, 255), 12)

    return lineImage

# Function to check if a bounding rectangle is inside a polygon
def IsInsidePolygon(rectangle, polygon):
    for point in rectangle:
        if cv2.pointPolygonTest(polygon, point, False) >= 0:
            return True
    return False

# A function to draw the given rectangles on the given image
def DrawRectangles(image, rectangles):
    imageWithRectangles = np.copy(image)
    for rectangle in rectangles:
        x, y = rectangle[0]
        color = (255, 0, 0)
        # Check if the rectangle is inside the danger zone
        if IsInsidePolygon(rectangle, dangerPolygon):
            # Add a red warning if the rectangle is inside the danger zone
            cv2.putText(imageWithRectangles, "WARNING", (x + 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            color = (0, 0, 255)
        # Draw the bounding rectangle around the car + warning if needed
        cv2.rectangle(imageWithRectangles, rectangle[0], rectangle[2], color, 6)
    
    return imageWithRectangles
    
# Function to check if a given rectangle is unique in the rectangles list
def UniqueRectangle(rectangleList, topLeft, bottomRight):
    for topLeftRectangle, _ , bottomRightRectangle, _ in rectangleList:
        # Check if there's an overlap between the rectangles of up to 120 pixels
        if topLeftRectangle[0] < topLeft[0] + 120 and topLeftRectangle[1] < topLeft[1] + 120 and bottomRightRectangle[0] + 120 > bottomRight[0] and bottomRightRectangle[1] + 120 > bottomRight[1]:
            return False
    return True    

# Function to find matches to templates in an image and return the bounding rectangles
def FindTemplateMatches(image):
    rectanglesList = []
    threshold = 0.82
    # Iterate through template list and look for matches
    for template in allTemplates:
        resultAfterMatching = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        # Use cv2.minMaxLoc() to extract the location of the best match
        _, maxVal, _, topLeft = cv2.minMaxLoc(resultAfterMatching)
        # Determine the bounding rectangle corners for the match
        width, height = (template.shape[1], template.shape[0])
        topRight = (topLeft[0] + width, topLeft[1])
        bottomRight = (topLeft[0] + width, topLeft[1] + height)
        bottomLeft = (topLeft[0], topLeft[1] + height)
        # Check if the confidence score is above the threshold and if the rectangle is unique
        if maxVal > threshold and UniqueRectangle(rectanglesList, topLeft, bottomRight):
            # Add the rectangle to the list
            rectanglesList.append((topLeft, topRight, bottomRight, bottomLeft))

    return rectanglesList

def ProcessFrame(frame):
    # Convert the frame to grayscale
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur on the grayscale image
    blurImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
    # Use Canny edge detection on the blurred image
    edgesImage = cv2.Canny(blurImage, 50, 150)
    # Get the edges in the region of interest (in the lane polygon)
    maskedEdges = RegionOfInterest(edgesImage)
    # Use Hough transform to detect lines from the edges
    lines = cv2.HoughLinesP(maskedEdges, 2, np.pi / 180, 70, minLineLength=5, maxLineGap=5)
    # Draw the lanes lines image
    lanesImage = DisplayLines(frame, lines)
    # Combine the original frame with the lanes image
    combinedImage = cv2.addWeighted(frame, 1, lanesImage, 1, 0)    
    # Display the lane switching message if there's a lane switch
    if switching:
        if switching == "Right":
            # Position the text on the right side of the frame (including a right arrow symbol)
            textPosition = (frame.shape[1] // 2, 75)  # Adjust the x-coordinate as needed
            message = f"{switching} Lane Switching ->"
        elif switching == "Left":
            # Position the text on the left side of the frame (including a left arrow symbol)
            textPosition = (50, 75)  # Near the left edge
            message = f"<- {switching} Lane Switching"
        # Draw the text (with the arrow symbol) on the combined image
        cv2.putText(combinedImage, message, textPosition, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)

    return combinedImage

########################################################################################

# Apply the video processing to our 3 videos:
laneSwitchingVideo = "LaneSwitchingVideo.mp4"
nightVideo = "NightVideo.mp4"
carDetectionVideo = "CarDetectionVideo.mp4"
videosNameArray = [laneSwitchingVideo, nightVideo, carDetectionVideo]

# Process each video
for i in range(len(videosNameArray)):
    videoPath = "LaneDetection/Videos/" + videosNameArray[i]
    cap = cv2.VideoCapture(videoPath)
    out = None
    # Error checking
    if not cap.isOpened():
        print("Could not open the video")
    
    # Process each frame in the video while it's active
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Check if this is the first frame
        if out is None:
            # Initialize the video writer and select video's global parameters
            out = cv2.VideoWriter('LaneDetection/OutputVideos/Output_' + videosNameArray[i],
                                  cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame.shape[1], frame.shape[0]))        
            # Set the global parameters according to the video
            if i == 0:
                # Lane switching video parameters
                polygonParameters = [0.4, 0.63, 0.55, 0.63, 0.8, 1, 0.1, 1]
                laneLength = 0.66
                glabalLanesCounter = 11
            elif i == 1:
                # Night video parameters
                polygonParameters = [0.44, 0.66, 0.54, 0.66, 0.8, 1, 0.13, 1]
                laneLength = 0.73
                glabalLanesCounter = 8
            else:
                # Car detection video parameters
                polygonParameters = [0.3, 0.6, 0.55, 0.6, 0.7, 0.95, 0.02, 0.95]
                laneLength = 0.66
                glabalLanesCounter = 7
                SelectDangerZone(frame)
            # Select the lane polygon for this video
            SelectLanePolygon(frame) 

        # Process the frame to get the frame with its lanes
        frameWithLanes = ProcessFrame(frame)
        
        if i == 2: # Car detection video
            # Find cars in the video and draw their bounding rectangles + warning message when needed
            cars = FindTemplateMatches(frame)
            frameWithLanes = DrawRectangles(frameWithLanes, cars)

        # Write the frame to the output video
        out.write(frameWithLanes)

        # Show the frames while running, for debugging purposes
        # cv2.imshow('out', frameWithLanes)

        # Break from the loop if the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release all resources
    out.release()
    cap.release()
    cv2.destroyAllWindows()

print("Videos processing complete!")