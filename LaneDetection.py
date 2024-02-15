# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Lane detection variables
rightLaneGlobal = [0, 0, 0, 0, 0]
leftLaneGlobal = [0, 0, 0, 0, 0]
polygon_parameters = [0, 0, 0, 0, 0, 0, 0, 0]
lane_polygon = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.int32)
# Lane switching variables
switching = None
timer = 0
# Collision detection variables
danger_polygon = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.int32)
allTemplates = []
template_strings = []
for i in range(1, 6):
    template_strings.append('Templates\Car' + str(i) + '.png')

# Load more templates in different sizes for better detection
for template in template_strings:
        template = cv2.imread(template)
        for i in range(0, 50, 7):
           allTemplates.append(cv2.resize(template, (int(template.shape[1]*(1-i/100)), int(template.shape[0]*(1-i/100))), interpolation = cv2.INTER_AREA))

# Function to create a lane detection polygon
def select_lane_polygon(frame):
    frameHeight, frameWidth = frame.shape[:2]
    global lane_polygon
    # Define the polygon vertices
    lane_polygon = np.array([[
        (frameWidth * polygon_parameters[0], frameHeight * polygon_parameters[1]),
        (frameWidth * polygon_parameters[2], frameHeight * polygon_parameters[3]),
        (frameWidth * polygon_parameters[4], frameHeight * polygon_parameters[5]),
        (frameWidth * polygon_parameters[6], frameHeight * polygon_parameters[7]),
    ]], np.int32)


############### comment until here
    
# %%
# Define a region of interest
def RegionOfInterest(edgesImages):
    mask = np.zeros_like(edgesImages)
    cv2.fillPoly(mask, lane_polygon, 255)
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

    ## For debugging:
    # print("linesAfterInitialFilter:", linesAfterInitialFilter)

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

        cv2.line(lineImage, (x1, y1), (x2, y2), (255, 0, 255), 12)
    
    return lineImage

# %%
# Function to detect lane switching
def checkForLaneSwitching(imageWidth, x1, y1, x2, y2):
    global switching, timer
    if not switching:
        middle_x = imageWidth / 2
        m = (x2-x1) / (y2-y1)
        if ((x1 - 25 < middle_x) and (x2 + 25 > middle_x)) or (
            (x1 + 25 > middle_x) and (x2 - 25 < middle_x)):
            timer = time.time()  # Start the timer
            switching = "Right" if m > 0 else "Left"
            print("Switching: ", switching)
    else:
        if time.time() - timer > 4:  # 4 seconds have passed
            switching = None  # Reset the switching state
            timer = 0

# %%
def processFrame(frame):
    # Convert to grayscale
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
    # Use Canny edge detection
    edgesImage = cv2.Canny(blurImage, 50, 150)
    
    maskedEdges = RegionOfInterest(edgesImage)

    # Use Hough transform to detect lines
    lines = cv2.HoughLinesP(maskedEdges, 2, np.pi / 180, 70, minLineLength=5, maxLineGap=5)
    # print(lines)

    # Draw the lines on the original image
    lineImage = DisplayLines(frame, lines)
    comboImage = cv2.addWeighted(frame, 0.8, lineImage, 1, 1)

    frameWidth = frame.shape[1]
    # Display the lane switching message if applicable
    if switching:
        if switching == "Right":
            # Position the text on the right side of the frame, include a right arrow symbol
            text_position = (frameWidth // 2, 75)  # Adjust the x-coordinate as needed
            message = f"{switching} Lane Switching ->"
        elif switching == "Left":
            # Position the text on the left side of the frame, include a left arrow symbol
            text_position = (50, 75)  # Near the left edge
            message = f"<- {switching} Lane Switching"

        # Draw the text with the arrow symbol
        cv2.putText(comboImage, message, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)

    return comboImage
    
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

    frameWidth = frame.shape[1]
    # Display the lane switching message if applicable
    if switching:
        if switching == "Right":
            # Position the text on the right side of the frame, include a right arrow symbol
            text_position = (frameWidth // 2, 75)  # Adjust the x-coordinate as needed
            message = f"{switching} Lane Switching ->"
        elif switching == "Left":
            # Position the text on the left side of the frame, include a left arrow symbol
            text_position = (50, 75)  # Near the left edge
            message = f"<- {switching} Lane Switching"

        # Draw the text with the arrow symbol
        cv2.putText(comboImage, message, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                    1.5, (0, 255, 255), 3, cv2.LINE_AA)

    # Display the image
    plt.imshow(comboImage)
    plt.show()

    return comboImage

# %%
###############################################
# A function to draw bounding boxes on an image
def draw_boxes(img, boxes):
    img_copy = np.copy(img)
    for box in boxes:
        x, y = box[0]
        color = (255, 0, 0)
        # Add warning if the box is inside the danger zone
        if is_inside_polygon(box, danger_polygon):
            cv2.putText(img_copy, "WARNING", (x + 20, y - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            color = (0, 0, 255)
        # Draw the bounding box around the car
        cv2.rectangle(img_copy, box[0], box[2], color, 6)
    
    return img_copy
    
    
def uniqueBox(box_list, top_left, bottom_right):
    for top_left_box, _ , bottom_right_box, _ in box_list:
        if top_left_box[0] < top_left[0] and top_left_box[1] < top_left[1] and bottom_right_box[0] > bottom_right[0] and bottom_right_box[1] > bottom_right[1]:
            return False
    return True
        

# Function to find matches to templates in an image and return the bounding boxes
def find_matches(img):
    box_list = []
    # Iterate through template list

    for tmp in allTemplates:
        result_after_matching = cv2.matchTemplate(img, tmp, cv2.TM_CCOEFF_NORMED)
        # Use cv2.minMaxLoc() to extract the location of the best match
        _, max_val, _, top_left = cv2.minMaxLoc(result_after_matching)
        # Determine bounding box corners for the match
        w, h = (tmp.shape[1], tmp.shape[0])
        top_right = (top_left[0] + w, top_left[1])
        bottom_right = (top_left[0] + w, top_left[1] + h)
        bottom_left = (top_left[0], top_left[1] + h)
        threshold = 0.8
        # Check if confidence score is above threshold
        if max_val > threshold and uniqueBox(box_list, top_left, bottom_right):
            box_list.append((top_left, top_right, bottom_right, bottom_left))

    return box_list
##############################

# Function to check if a bounding box is inside a polygon
def is_inside_polygon(box, polygon):
    for point in box:
        if cv2.pointPolygonTest(polygon, point, False) >= 0:
            return True
    return False

# Function to create a danger polygon
def select_danger_zone(frame):
    frameHeight, frameWidth = frame.shape[:2]
    global danger_polygon
    # Define the polygon vertices
    danger_polygon = np.array([[
        (frameWidth * 0.39, frameHeight * 0.53),
        (frameWidth * 0.51, frameHeight * 0.53),
        (frameWidth * 0.7, frameHeight),
        (frameWidth * 0.1, frameHeight),
    ]], np.int32)

    # # Print the polygon vertices for debugging
    # cv2.polylines(frame, [danger_polygon], True, (0, 0, 255), 2)  # Draw in blue
    # plt.imshow(frame)
    # plt.show()
##############################


# %%
# Apply the process to a single image
# picture = cv2.imread("Crossroad1.png")
# processframeWithPrints(picture)

# %%
# Apply the same process to a video
# videoPath = "CrashVideo2_short.mp4"

# cap = cv2.VideoCapture(videoPath)

# if not cap.isOpened():
#     print("Could not open the video")

# out = None

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # In the first frame, initialize the video writer and select the 
#     # lane polygon and danger zone
#     if out is None:
#         out = cv2.VideoWriter('outputVideo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
#                               30, (frame.shape[1], frame.shape[0]))
#         select_lane_polygon(frame)
#         select_danger_zone(frame)
    
#     ## run as single frames
#     # plt.imshow(processFrame(frame))
#     # plt.show()

#     frameWithLanes = processFrame(frame)
#     ########### find cars in the video
#     boxes = find_matches(frame)
#     frameWithLanes = draw_boxes(frameWithLanes, boxes)
#     ###########
#     # cv2.imshow('out', frameWithLanes)
#     out.write(frameWithLanes)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# out.release()
# cap.release()
# cv2.destroyAllWindows()

# %%
# # Check the danger polygon
# picture = cv2.imread("crashPicture2.png")
# select_danger_zone(picture)

# %%

# %%
# Apply the same process to the 3 videos
Lane_Switching_Video = "LaneSwitchingVideo2.mp4"
Crash_Video = "CrashVideo2_short.mp4"
videoPath_array = [Lane_Switching_Video ,Crash_Video]

for videoPath in videoPath_array:
    cap = cv2.VideoCapture(videoPath)
    # Error handling
    if not cap.isOpened():
        print("Could not open the video")
    
    out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Set the global lane polygon parameters for each video
        if videoPath == Lane_Switching_Video:
            polygon_parameters = [0.4, 0.63, 0.55, 0.63, 0.8, 1, 0.1, 1]
        elif videoPath == Crash_Video:
            polygon_parameters = [0.3, 0.6, 0.55, 0.6, 0.7, 0.95, 0.02, 0.95]

        # In the first frame, initialize the video writer and select the 
        # lane polygon and danger zone
        if out is None:
            out = cv2.VideoWriter('output_' + videoPath, cv2.VideoWriter_fourcc(*'mp4v'), 
                                30, (frame.shape[1], frame.shape[0]))
            select_lane_polygon(frame)
            if videoPath == Crash_Video:
                select_danger_zone(frame)

        frameWithLanes = processFrame(frame)
        # Find cars in the video and draw bounding boxes and warning messages
        if videoPath == Crash_Video:
            cars = find_matches(frame)
            frameWithLanes = draw_boxes(frameWithLanes, cars)

        ## Print the frame to debug
        cv2.imshow('out', frameWithLanes)
            
        out.write(frameWithLanes)

        ## Break the loop if the user presses 'q' while the video is playing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    out.release()
    cap.release()
    cv2.destroyAllWindows()

# %%