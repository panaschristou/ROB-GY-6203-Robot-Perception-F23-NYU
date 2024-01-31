import cv2
import numpy as np

# Load the images
left_img = cv2.imread(r'C:\Users\pchristou\OneDrive - FOS Development Corp\Documents\Personal Documents\School Shit\Homework\RP_HW2\Task 3/left.jpg')
right_img = cv2.imread(r'C:\Users\pchristou\OneDrive - FOS Development Corp\Documents\Personal Documents\School Shit\Homework\RP_HW2\Task 3/right.jpg')

# PART A

# Convert images to grayscale
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

# Feature detection (using ORB, you can use other detectors as well)
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(left_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(right_gray, None)

# Match descriptors between the images
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Extract matched keypoints
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Estimate fundamental matrix
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.001, 0.99)


# PART B

def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(map(int, pt1[0])), 5, color, -1)
        img2 = cv2.circle(img2, tuple(map(int, pt2[0])), 5, color, -1)
    return img1, img2

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
left_img, _ = drawlines(left_gray, right_gray, lines1, pts1, pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
right_img, _ = drawlines(right_gray, left_gray, lines2, pts2, pts1)

# Resize images
scale_percent = 33 # percent of original size
width = int(left_img.shape[1] * scale_percent / 100)
height = int(left_img.shape[0] * scale_percent / 100)
dim = (width, height)
left_img_resized = cv2.resize(left_img, dim, interpolation = cv2.INTER_AREA)
right_img_resized = cv2.resize(right_img, dim, interpolation = cv2.INTER_AREA)

# Display images
cv2.imshow('left', left_img_resized)
cv2.imshow('right', right_img_resized)

# Move windows
cv2.moveWindow('left', 0, 0)
cv2.moveWindow('right', left_img_resized.shape[1] + 10, 0)

cv2.waitKey(0)
cv2.destroyAllWindows()


# PART C

# Add your calibration image paths here
image_paths = [ r'C:\Users\pchristou\OneDrive - FOS Development Corp\Documents\Personal Documents\School Shit\Homework\RP_HW2\Task 3\1.jpg', r'C:\Users\pchristou\OneDrive - FOS Development Corp\Documents\Personal Documents\School Shit\Homework\RP_HW2\Task 3\2.jpg', r'C:\Users\pchristou\OneDrive - FOS Development Corp\Documents\Personal Documents\School Shit\Homework\RP_HW2\Task 3\3.jpg', r'C:\Users\pchristou\OneDrive - FOS Development Corp\Documents\Personal Documents\School Shit\Homework\RP_HW2\Task 3\4.jpg', r'C:\Users\pchristou\OneDrive - FOS Development Corp\Documents\Personal Documents\School Shit\Homework\RP_HW2\Task 3\5.jpg', \
                           r'C:\Users\pchristou\OneDrive - FOS Development Corp\Documents\Personal Documents\School Shit\Homework\RP_HW2\Task 3\6.jpg', r'C:\Users\pchristou\OneDrive - FOS Development Corp\Documents\Personal Documents\School Shit\Homework\RP_HW2\Task 3\7.jpg', r'C:\Users\pchristou\OneDrive - FOS Development Corp\Documents\Personal Documents\School Shit\Homework\RP_HW2\Task 3\8.jpg', r'C:\Users\pchristou\OneDrive - FOS Development Corp\Documents\Personal Documents\School Shit\Homework\RP_HW2\Task 3\9.jpg', r'C:\Users\pchristou\OneDrive - FOS Development Corp\Documents\Personal Documents\School Shit\Homework\RP_HW2\Task 3\10.jpg', \
                            r'C:\Users\pchristou\OneDrive - FOS Development Corp\Documents\Personal Documents\School Shit\Homework\RP_HW2\Task 3\11.jpg']



calibration_images = []  # List to store calibration images

# Load calibration images
for path in image_paths:
    img = cv2.imread(path)
    calibration_images.append(img)

pattern_rows = 6  # Number of rows in the calibration pattern
pattern_cols = 8  # Number of columns in the calibration pattern

object_points = []  # List to store object points (3D)
image_points = []   # List to store image points (2D)

pattern_points = np.zeros((pattern_rows * pattern_cols, 3), np.float32)
pattern_points[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)

for img in calibration_images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (pattern_cols, pattern_rows), None)

    if ret:
        object_points.append(pattern_points)
        image_points.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (pattern_cols, pattern_rows), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()
# Get image dimensions
image_shape = gray.shape[::-1]

# Perform camera calibration
ret, K, distortion, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_shape, None, None)
k1, k2, *_ = distortion.ravel()  # Extract the top two distortion parameters

print("Camera Matrix (K):")
print(K)
print("Distortion Parameters (k1, k2):")
print(k1, k2)

# Compute Essential matrix from Fundamental matrix
E = np.dot(K.T, np.dot(F, K))

# Assuming you have matchedPoints1 and matchedPoints2 as the matched points from two images

_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

print("Rotation Matrix (R):")
print(R)
print("Translation Vector (t):")
print(t)