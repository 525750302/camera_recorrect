# camera_recorrect
This program recognizes faces in images taken vertically from above using YOLOv8, mediapipe, and deepface models. And can recognize the age and gender of the face by using deepface.

# How to use
To use, please download the entire project.

1.Before running, please change the path ```C:/Users/XIR1SBY/Desktop/camera``` in all files to your own local file path!

2.To download the relevant packages, please refer to the following address
[YOLOV8](https://github.com/ultralytics/ultralytics),[Mediapipe](https://developers.google.com/mediapipe),[Deepface](https://github.com/serengil/deepface).This project uses ```yolov8s.pt``` as the yolo pre-training model by default.

3.Change code in deepface package (path ```.\Lib\site-packages\deepface```)

  + In ```Deepface.py```, add the following code in line 321.

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

# Improvement

The YOLO v8 model has been further trained to recognize horizontal portraits of individuals entering the frame from a side angle in a vertical orientation. 

By utilizing MediaPipe for rapid facial data capture, the efficiency and accuracy of the DeepFace model's processing capabilities are enhanced. 

The detection accuracy is improved and false identifications are reduced by allocating weights of 0.66 to the DeepFace.RetinaFace model and 0.33 to the DeepFace.MediaPipe model.






# Result
Result will be saved in ```.\yolo``` file folder.


Age and gender result will be saved in ```.\yolo\result.txt```.
ID and location data of human box will be saved in ```.\yolo\box_data_X.txt```
For image data will change during running program, please check the result during running.
For better result output, please check python files in ```.\model and code``` file folder.

Python files in ```.\program for 30 degree``` are used for tilt angle cameras.

#####################################################

A recognition result. The person in the image has been successfully identified as a young male under 30 years old and is being continuously tracked.

[https://github.com/525750302/camera_recorrect/issues/1#issue-2131974745](https://github.com/525750302/camera_recorrect/assets/46802084/7791c781-381c-49db-a0c2-86a266fc6842)
