# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %%
# Define a region of interest
def region_of_interest(edges_img):
    frame_height, frame_width = edges_img.shape
    mask = np.zeros_like(edges_img)

    # Only focus on the bottom half of the screen
    polygon = np.array([[
        (frame_width * 0.4, frame_height * 0.63),
        (frame_width * 0.55, frame_height * 0.63),
        (frame_width * 0.8, frame_height),
        (frame_width * 0.1, frame_height),
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(edges_img, mask)
    return masked_image

# %%
def calculateLaneBoundaries(x1, y1, x2, y2, frame_height):
    m = (x2-x1)/(y2-y1)
    n = x1 - m*y1

    return(int(frame_height*m + n), int(frame_height), int(frame_height*0.65*m + n), int(frame_height*0.65))

def distance(x1, y1, x2, y2):
    return np.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))

def FilterLines(lines):
    linesAfterFilter = []
    lines = sorted(lines, key=lambda x:x[0][0])
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if(y1 == y2):
            continue
        elif(np.abs((x2-x1)/(y2-y1)) > 5):
            continue

        SimilarLineInlinesAfterFilter = False
        for lineAfterFilter in linesAfterFilter:
            X1, Y1, X2, Y2 = lineAfterFilter[0]
            if(distance((X1-X2)/2, (Y1-Y2)/2, (x1-x2)/2, (y1-y2)/2) <= 40):
                SimilarLineInlinesAfterFilter = True
        if(not SimilarLineInlinesAfterFilter):
            linesAfterFilter.append(line)
    print("linesAfterFilter:", linesAfterFilter)
    return linesAfterFilter          
                    

# Function to draw selected lines on the image
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is None:
        return line_image

    linesAfterFilter = FilterLines(lines)
    #for line in lines:
    for line in linesAfterFilter:
        x1, y1, x2, y2 = line[0]
        # cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        # if(y1 == y2):
        #     continue
        # elif(np.abs((x2-x1)/(y2-y1)) > 5):
        #     continue
        
        x1, y1, x2, y2 = calculateLaneBoundaries(x1, y1, x2, y2, image.shape[0])
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 255), 10)
    return line_image

# %%
def processFrame(frame):
    # Convert to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    # Use Canny edge detection
    edges_img = cv2.Canny(blur_img, 50, 150)

    masked_edges = region_of_interest(edges_img)

    # Use Hough transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, 70, minLineLength=5, maxLineGap=5)
    print(lines)

    # Draw the lines on the original image
    line_img = display_lines(frame, lines)
    combo_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)

    # Convert the final image to RGB
    combo_img_rgb = cv2.cvtColor(combo_img, cv2.COLOR_BGR2RGB)

    return combo_img_rgb
    
# %%
# Function to process a frame with prints for debugging
def processFrame_withPrints(frame):
    # Load the image
    plt.imshow(frame)
    plt.show()

    # Convert to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray_img)
    # plt.show()

    # Apply Gaussian blur
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    # plt.imshow(blur_img)
    # plt.show()

    # Use Canny edge detection
    edges_img = cv2.Canny(blur_img, 50, 150)
    plt.imshow(edges_img)
    plt.show()

    masked_edges = region_of_interest(edges_img)
    plt.imshow(masked_edges)
    plt.show()

    # Use Hough transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, 70, minLineLength=5, maxLineGap=5)
    print(lines)

    # Draw the lines on the original image
    line_img = display_lines(frame, lines)
    combo_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)

    # Convert the final image to RGB
    combo_img_rgb = cv2.cvtColor(combo_img, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(combo_img_rgb)
    plt.show()

    return combo_img_rgb

# %%
# Apply the process to a single image
# picture = cv2.imread("LanePicture_Debug1.png")
# processFrame_withPrints(picture)

# %%
# Apply the same process to a video
video_path = "LaneSwitchingVideo2.mp4"

cap = cv2.VideoCapture(video_path)

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