import cv2
import os

# Load the cascades
# Load the cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

# Create directory to save mouth images
if not os.path.exists('mouths'):
    os.makedirs('mouths')

# Specify the source directory for images
source_dir = './image_data'

for filename in os.listdir(source_dir):
    img = cv2.imread(os.path.join(source_dir, filename))
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Adjust y coordinate for mouths (they will be lower in the image)
            roi_gray = roi_gray[int(0.5 * h):h, :]
            roi_color = roi_color[int(0.5 * h):h, :]

            mouths = mouth_cascade.detectMultiScale(roi_gray, 1.3, 5)

            for (mx, my, mw, mh) in mouths:
                cropped_mouth = roi_color[my:my + mh, mx:mx + mw]
                cv2.imwrite(f'./mouths/{filename}_mouth.png', cropped_mouth)
                break  # Save only one mouth per face
