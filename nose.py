import cv2
import os

# Load the cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml') # Adjust this path

# Create directory to save noses
if not os.path.exists('noses'):
    os.makedirs('noses')

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

            noses = nose_cascade.detectMultiScale(roi_gray)

            # Check if the nose was detected
            if len(noses) >= 1:
                nx, ny, nw, nh = noses[0]
                cropped_nose = roi_color[ny:ny+nh, nx:nx+nw]
                cv2.imwrite(f'./noses/{filename}_nose.png', cropped_nose)
