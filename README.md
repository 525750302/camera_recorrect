# camera_recorrect
This program recognizes faces in images taken vertically from above using YOLOv8, mediapipe, and deepface models. And can recognize the age and gender of the face by using deepface.

# How to use
To use, please download the entire project.

1.Before running, please change the path ```C:/Users/XIR1SBY/Desktop/camera``` in all files to your own local file path!

2.To download the relevant packages, please refer to the following address
[YOLOV8](https://github.com/ultralytics/ultralytics),[Mediapipe](https://developers.google.com/mediapipe),[Deepface](https://github.com/serengil/deepface).This project uses ```yolov8s.pt``` as the yolo pre-training model by default.

3.Change code in deepface package (path ```.\Lib\site-packages\deepface```)

  + In ```Deepface.py```, add follow code in line 321.

    ```
    if img_objs == -1:
      return -1
    ```

  + In ```commons\Deepface.py```, change Line 114

    ```
    if len(face_objs) == 0 and enforce_detection is True:
      raise ValueError(
          "Face could not be detected. Please confirm that the picture is a face photo "
          + "or consider to set enforce_detection param to False."
      )
    ```

    to

    ```
    if len(face_objs) == 0 and enforce_detection is True:
      return -1
      #raise ValueError(
      #    "Face could not be detected. Please confirm that the picture is a face photo "
      #    + "or consider to set enforce_detection param to False."
      #)
    ```

  + In ```detectors\RetinaFaceWrapper.py```, change Line 19 to

    ```
    obj = RetinaFace.detect_faces(img, model=face_detector, threshold=0.8)
    ```

4. change input data path in ```main.py```
5. Run ```main.py```

# Problem
It is still difficult to recognize men and women from top angle. 

Due to the attribute of mediapipe, it is difficult to recognize people with long hair. Due to the fact that the retinaface uses the contours of the face to recognize the face, people with thinning hair may recognize other parts of the face as a human face.

By changing the angle at which the photo or video is taken, it is possible to increase the accuracy of the recognition by taking the photo from an angle where the face can be more easily seen.
