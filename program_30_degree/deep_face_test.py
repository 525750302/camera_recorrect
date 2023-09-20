from deepface import DeepFace
import cv2
img_path = "C:/Users/XIR1SBY/Desktop/camera/program_30_degree/face_0.png"
objs = DeepFace.analyze(img_path, 
        actions = ['age', 'gender'],
        detector_backend="mediapipe",
        enforce_detection = True)
print(objs)
x = objs[0]["region"]["x"]
y = objs[0]["region"]["y"]
w = objs[0]["region"]["w"]
h = objs[0]["region"]["h"]
gape_origin_frame = cv2.imread(img_path)
target = gape_origin_frame[max(int(y), 0):int(y + h),max(int(x), 0):int(x + w)]
cv2.imwrite("C:/Users/XIR1SBY/Desktop/camera/program_30_degree/detect_target.png", target)
print(objs)

