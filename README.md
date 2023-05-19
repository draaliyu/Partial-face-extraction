Face Feature Extractor

This Python script uses OpenCV to detect and extract eyes, noses, and mouths from images of faces. Extracted features are saved as individual .png images.
Dependencies

    Python 3
    OpenCV (cv2)

You can install OpenCV with pip: pip install opencv-python
Usage

    Download the haarcascade XML files from the OpenCV GitHub repository or other source.
        haarcascade_frontalface_default.xml
        haarcascade_eye.xml
        haarcascade_mcs_nose.xml
        haarcascade_mcs_mouth.xml

    Replace 'path_to_your_file/haarcascade_mcs_nose.xml' and 'path_to_your_file/haarcascade_mcs_mouth.xml' in the script with the path to your downloaded files.

    Save your source images in a directory.

    Modify source_dir = './image_data' in the script to the directory containing your source images.

    Run the script. Extracted features will be saved in the eyes, noses, and mouths directories.
