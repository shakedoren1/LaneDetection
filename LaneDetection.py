# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

rightLaneGlobal = [0, 0, 0, 0, 0]
leftLaneGlobal = [0, 0, 0, 0, 0]
switching = None
timer = 0

# %%
# Define a region of interest
def RegionOfInterest(edgesImages):
    frameHeight, frameWidth = edgesImages.shape
    mask = np.zeros_like(edgesImages)

    # Only focus on the bottom half of the screen
    polygon = np.array([[
        (frameWidth * 0.4, frameHeight * 0.63),
        (frameWidth * 0.55, frameHeight * 0.63),
        (frameWidth * 0.8, frameHeight),
        (frameWidth * 0.1, frameHeight),
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)
    maskedImage = cv2.bitwise_and(edgesImages, mask)
    return maskedImage

# %%
def CalculateLaneBoundaries(x1, y1, x2, y2, frameHeight):
    m = (x2-x1)/(y2-y1)
    n = x1 - m*y1

    return(int(frameHeight*m + n), int(frameHeight), int(frameHeight*0.66*m + n), int(frameHeight*0.66))

def Distance(coordinates):
    x1, y1, x2, y2 = coordinates
    return np.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))

def BestLeftLine(linesAfterFilter, frameWidth):
    bestLeftLine = [0, 0, 0, 0]
    """
    for line in linesAfterFilter:
        x1, y1, x2, y2 = line[0]
        if(x1 < frameWidth // 2 or x2 < frameWidth // 2) and (x2-x1)/(y2-y1) < 0:
            if Distance(line[0]) > Distance(bestLeftLine):
                bestLeftLine = line[0]"""
    
    linesAfterFilter = sorted(linesAfterFilter, key=lambda x:x[0][0])
    for line in linesAfterFilter:
        x1, y1, x2, y2 = line[0]
        if (x1 < frameWidth // 2 or x2 < frameWidth // 2) and (x2-x1)/(y2-y1) < 0:
            if Distance(line[0]) > 85:
                return line[0]
            
    return bestLeftLine

def BestRightLine(linesAfterFilter, frameWidth):
    bestRightLine = [0, 0, 0, 0]

    for line in linesAfterFilter:
        x1, y1, x2, y2 = line[0]
        if(x1 > frameWidth // 2 or x2 > frameWidth // 2):
            if Distance(line[0]) > Distance(bestRightLine):
                bestRightLine = line[0]
    """
    linesAfterFilter = sorted(linesAfterFilter, key=lambda x:x[0][0])
    for line in linesAfterFilter:
        x1, y1, x2, y2 = line[0]
        if x1 > frameWidth // 2 or x2 > frameWidth // 2:
            if Distance(line[0]) > 70:
                return line[0]"""

    return bestRightLine

def FilterLines(lines, frameWidth):
    linesAfterInitialFilter = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if(y1 == y2):
            continue
        elif(np.abs((x2-x1)/(y2-y1)) > 2):
            continue

        linesAfterInitialFilter.append(line)

    print("linesAfterInitialFilter:", linesAfterInitialFilter)

    global leftLaneGlobal
    global rightLaneGlobal

    LeftLane = BestLeftLine(linesAfterInitialFilter, frameWidth)
    RightLane = BestRightLine(linesAfterInitialFilter, frameWidth)       

    if(LeftLane[0] == 0):
        if(leftLaneGlobal[4] > 0):
            leftLaneGlobal[4] = leftLaneGlobal[4] - 1
            LeftLane = leftLaneGlobal[0:4]
    else:
        leftLaneGlobal = np.append(LeftLane, 11)
      
    if(RightLane[0] == 0):
        if(rightLaneGlobal[4] > 0):
            rightLaneGlobal[4] = rightLaneGlobal[4] - 1
            RightLane = rightLaneGlobal[0:4]
    else:
        rightLaneGlobal = np.append(RightLane, 11)
    
    if(LeftLane[0] != 0 and RightLane[0] != 0):
        return [[LeftLane], [RightLane]]
    elif(LeftLane[0] == 0 and RightLane[0] != 0):
        return [[RightLane]]
    elif(LeftLane[0] != 0 and RightLane[0] == 0):
        return [[LeftLane]]
    else:
        return []                   

# Function to draw selected lines on the image
def DisplayLines(image, lines):
    lineImage = np.zeros_like(image)
    if lines is None:
        return lineImage

    linesAfterFilter = FilterLines(lines, image.shape[1])
    for line in linesAfterFilter:
        x1, y1, x2, y2 = line[0]        
        x1, y1, x2, y2 = CalculateLaneBoundaries(x1, y1, x2, y2, image.shape[0])
        
        checkForLaneSwitching(image.shape[1], x1, y1, x2, y2)

        if(rightLaneGlobal[4] <= 1 or leftLaneGlobal[0] <= 1):
            cv2.line(lineImage, (x1, y1), (x2, y2), (0, 255, 0), 12)
        else:
            cv2.line(lineImage, (x1, y1), (x2, y2), (255, 0, 255), 12)
    
    return lineImage

# %%
# Function to detect lane switching
def checkForLaneSwitching(imageWidth, x1, y1, x2, y2):
    global switching, timer
    if not switching:
        middle_x = imageWidth / 2
        m = (x2-x1) / (y2-y1)
        if ((x1 < middle_x) and (x2 > middle_x)) or ((x1 > middle_x) and (x2 < middle_x)):
            switching = True
            timer = time.time()  # Start the timer
            switching = "Right" if m > 0 else "Left"
            print("Switching: ", switching)
    else:
        if time.time() - timer > 3:  # 3 seconds have passed
            switching = False  # Reset the switching state after 2 seconds

# %%
def processFrame(frame):
    # Convert to grayscale
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
    # Use Canny edge detection
    edgesImages = cv2.Canny(blurImage, 50, 150)

    maskedEdges = RegionOfInterest(edgesImages)

    # Use Hough transform to detect lines
    lines = cv2.HoughLinesP(maskedEdges, 2, np.pi / 180, 70, minLineLength=5, maxLineGap=5)
    # print(lines)

    # Draw the lines on the original image
    lineImage = DisplayLines(frame, lines)
    comboImage = cv2.addWeighted(frame, 0.8, lineImage, 1, 1)

    # Convert the final image to RGB
    comboImageRGB = cv2.cvtColor(comboImage, cv2.COLOR_BGR2RGB)

    frameWidth = frame.shape[1]
    # Display the lane switching message if applicable
    if switching:
        if switching == "Right":
            # Position the text on the right side of the frame
            text_position = (frameWidth - 400, 50)  # Adjust the x-coordinate as needed
        elif switching == "Left":
            # Position the text on the left side of the frame
            text_position = (50, 50)  # Near the left edge

        cv2.putText(comboImageRGB, f"{switching} Lane Switching", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    return comboImageRGB
    
# %%
# Function to process a frame with prints for debugging
def processframeWithPrints(frame):
    # Load the image
    plt.imshow(frame)
    plt.show()

    # Convert to grayscale
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # plt.imshow(grayImage)
    # plt.show()

    # Apply Gaussian blur
    blurImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
    # plt.imshow(blurImage)
    # plt.show()

    # Use Canny edge detection
    edgesImages = cv2.Canny(blurImage, 50, 150)
    plt.imshow(edgesImages)
    plt.show()

    maskedEdges = RegionOfInterest(edgesImages)
    plt.imshow(maskedEdges)
    plt.show()

    # Use Hough transform to detect lines
    lines = cv2.HoughLinesP(maskedEdges, 2, np.pi / 180, 70, minLineLength=5, maxLineGap=5)
    print(lines)

    # Draw the lines on the original image
    lineImage = DisplayLines(frame, lines)
    comboImage = cv2.addWeighted(frame, 0.8, lineImage, 1, 1)

    # Convert the final image to RGB
    comboImageRGB = cv2.cvtColor(comboImage, cv2.COLOR_BGR2RGB)

    frameWidth = frame.shape[1]
    # Display the lane switching message if applicable
    if switching:
        if switching == "Right":
            # Position the text on the right side of the frame
            text_position = (frameWidth - 400, 50)  # Adjust the x-coordinate as needed
        elif switching == "Left":
            # Position the text on the left side of the frame
            text_position = (50, 50)  # Near the left edge

        cv2.putText(comboImageRGB, f"{switching} Lane Switching", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Display the image
    plt.imshow(comboImageRGB)
    plt.show()

    return comboImageRGB

# %%
# Apply the process to a single image
# picture = cv2.imread("LanePicture5.png")
# processframeWithPrints(picture)

# %%
# Apply the same process to a video
videoPath = "LaneSwitchingVideo2_Fix.mp4"

cap = cv2.VideoCapture(videoPath)

if not cap.isOpened():
    print("Could not open the video")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    ## run as single frames
    # plt.imshow(processFrame(frame))
    # plt.show()

    ## run as video
    frameWithLanes = processFrame(frame)
    cv2.imshow('out', frameWithLanes)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()

# %%