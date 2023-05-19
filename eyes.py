import cv2
import os

# Load the cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Create directory to save eyes
if not os.path.exists('eyes'):
    os.makedirs('eyes')

# Specify the source directory for images
source_dir = './image_data'

for filename in os.listdir(source_dir):
    img = cv2.imread(os.path.join(source_dir, filename))
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Sort eyes by x coordinate
            eyes = sorted(eyes, key=lambda x: x[0])

            # Check if the eyes were detected
            if len(eyes) >= 2:
                left_eye = eyes[0]
                right_eye = eyes[1]

                # Save the left eye
                ex, ey, ew, eh = left_eye
                cropped_eye = roi_color[ey:ey+eh, ex:ex+ew]
                cv2.imwrite(f'./eyes/{filename}_left_eye.png', cropped_eye)

                # Save the right eye
                ex, ey, ew, eh = right_eye
                cropped_eye = roi_color[ey:ey+eh, ex:ex+ew]
                cv2.imwrite(f'./eyes/{filename}_right_eye.png', cropped_eye)
