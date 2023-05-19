import cv2
import os

# Load the cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')  # adjust this path
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')  # adjust this path

# Create directories to save the features
if not os.path.exists('eyes'):
    os.makedirs('eyes')
if not os.path.exists('noses'):
    os.makedirs('noses')
if not os.path.exists('mouths'):
    os.makedirs('mouths')

# Specify the source directory for images
source_dir = './1'


# Function to extract eyes
def extract_eyes(roi_gray, roi_color, filename):
    eyes = eye_cascade.detectMultiScale(roi_gray)

    # Sort eyes by x coordinate
    eyes = sorted(eyes, key=lambda x: x[0])

    # Check if the eyes were detected
    if len(eyes) >= 2:
        left_eye, right_eye = eyes[:2]

        # Save the left eye
        ex, ey, ew, eh = left_eye
        cropped_eye = roi_color[ey:ey + eh, ex:ex + ew]
        cv2.imwrite(f'./eyes/{filename}_left_eye.png', cropped_eye)

        # Save the right eye
        ex, ey, ew, eh = right_eye
        cropped_eye = roi_color[ey:ey + eh, ex:ex + ew]
        cv2.imwrite(f'./eyes/{filename}_right_eye.png', cropped_eye)


# Function to extract nose
def extract_nose(roi_gray, roi_color, filename):
    noses = nose_cascade.detectMultiScale(roi_gray)

    # Check if the nose was detected
    if len(noses) >= 1:
        nx, ny, nw, nh = noses[0]
        cropped_nose = roi_color[ny:ny + nh, nx:nx + nw]
        cv2.imwrite(f'./noses/{filename}_nose.png', cropped_nose)


# Function to extract mouth
def extract_mouth(roi_gray, roi_color, filename):
    mouths = mouth_cascade.detectMultiScale(roi_gray)

    # Check if the mouth was detected
    if len(mouths) >= 1:
        mx, my, mw, mh = mouths[0]
        cropped_mouth = roi_color[my:my + mh, mx:mx + mw]
        cv2.imwrite(f'./mouths/{filename}_mouth.png', cropped_mouth)


# Iterate over the images
for filename in os.listdir(source_dir):
    img = cv2.imread(os.path.join(source_dir, filename))
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            extract_eyes(roi_gray, roi_color, filename)
            extract_nose(roi_gray, roi_color, filename)
            extract_mouth(roi_gray, roi_color, filename)

