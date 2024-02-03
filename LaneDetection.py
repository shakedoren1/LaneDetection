# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# # Load the image
# image_path = "LanePicture3.png"
# image = cv2.imread(image_path)

# # Convert to grayscale
# gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# plt.imshow(gray_img)
# plt.show()

# # Apply Gaussian blur
# blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
# plt.imshow(blur_img)
# plt.show()

# # Use Canny edge detection
# edges_img = cv2.Canny(blur_img, 50, 150)
# plt.imshow(edges_img)
# plt.show()

# %%
# Define a region of interest
def region_of_interest(edges_img):
    frame_height, frame_width = edges_img.shape
    mask = np.zeros_like(edges_img)

    # Only focus on the bottom half of the screen
    polygon = np.array([[
        (frame_width * 0.2, frame_height * 0.7),
        (frame_width * 0.8, frame_height * 0.7),
        (frame_width * 0.8, frame_height),
        (frame_width * 0.2, frame_height),
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(edges_img, mask)
    return masked_image

# masked_edges = region_of_interest(edges_img)
# plt.imshow(masked_edges)
# plt.show()

def calculateLaneBoundaries(x1, y1, x2, y2, frame_height):
    m = (x2-x1)/(y2-y1)
    n = x1 - m*y1

    return(int(frame_height*m + n), int(frame_height), int(frame_height*0.65*m + n), int(frame_height*0.65))



# # Use Hough transform to detect lines
# lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, 100, minLineLength=40, maxLineGap=5)
# print(lines)

# Function to draw selected lines on the image
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
                if(y1 == y2):
                    continue
                x1, y1, x2, y2 = calculateLaneBoundaries(x1, y1, x2, y2, image.shape[0])
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 255), 20)
    return line_image

# # Draw the lines on the original image
# line_img = display_lines(image, lines)
# combo_img = cv2.addWeighted(image, 0.8, line_img, 1, 1)

# # Convert the final image to RGB
# combo_img_rgb = cv2.cvtColor(combo_img, cv2.COLOR_BGR2RGB)

# # Display the image
# plt.imshow(combo_img_rgb)
# plt.show()

def processFrame(frame):
    # Convert to grayscale
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    # Use Canny edge detection
    edges_img = cv2.Canny(blur_img, 50, 150)

    masked_edges = region_of_interest(edges_img)

    # Use Hough transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi / 180, 100, minLineLength=20, maxLineGap=5)
    print(lines)

    # Draw the lines on the original image
    line_img = display_lines(frame, lines)
    combo_img = cv2.addWeighted(frame, 0.8, line_img, 1, 1)

    # Convert the final image to RGB
    combo_img_rgb = cv2.cvtColor(combo_img, cv2.COLOR_BGR2RGB)

    return combo_img_rgb

# %%
# Apply the same process to a video
video_path = "LaneSwitchingVideo.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open the video")

while cap.isOpened():
    ret, frame = cap.read()
    plt.imshow(processFrame(frame))
    plt.show()

cap.release()
cv2.destroyAllWindows()

# %%