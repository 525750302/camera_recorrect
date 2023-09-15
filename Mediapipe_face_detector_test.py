# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import Mediapipe_face_dector_Visualization_utilities as Mf

# STEP 2: Create an FaceDetector object.
base_options = python.BaseOptions(model_asset_path='detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# STEP 3: Load the input image.
cap_path = "C:/Users/wuse/Desktop/camera_recorrect/camera_picture/anime3.mp4"
cap = cv2.VideoCapture(cap_path)
success = True
while success:
    success, image = cap.read()
    cv2.imwrite("C:/Users/wuse/Desktop/camera_recorrect/test.png", image)
    PATH = "C:/Users/wuse/Desktop/camera_recorrect/test.png"
    image = mp.Image.create_from_file(PATH)

    # STEP 4: Detect faces in the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    image_copy = np.copy(image.numpy_view())
    annotated_image = Mf.visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("TEST",rgb_annotated_image)
    cv2.waitKey(10)