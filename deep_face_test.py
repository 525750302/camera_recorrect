from deepface import DeepFace

objs = DeepFace.analyze(img_path = "C:/Users/XIR1SBY/Desktop/camera/yolo/gape_picture_0.png", 
        actions = ['age', 'gender'],
        detector_backend="retinaface",
        enforce_detection = True)
print(objs)

