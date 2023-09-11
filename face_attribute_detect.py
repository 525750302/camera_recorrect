from deepface import DeepFace
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons import functions, realtime, distance as dst
import numpy as np

img_path = "C:/Users/XIR1SBY/Desktop/camera/yolo/face_0.png"

models = {}
models["age"] = DeepFace.build_model("Age")
models["gender"] = DeepFace.build_model("Gender")

img_objs = functions.extract_faces(img_path, detector_backend = "skip")
for img_content, img_region, _ in img_objs:
    obj = {}
    age_predictions = models["age"].predict(img_content, verbose=0)[0, :]
    apparent_age = Age.findApparentAge(age_predictions)
    # int cast is for exception - object of type 'float32' is not JSON serializable
    obj["age"] = int(apparent_age)
    gender_predictions = models["gender"].predict(img_content, verbose=0)[0, :]
    obj["gender"] = {}
    for i, gender_label in enumerate(Gender.labels):
        gender_prediction = 100 * gender_predictions[i]
        obj["gender"][gender_label] = gender_prediction
    obj["dominant_gender"] = Gender.labels[np.argmax(gender_predictions)]
    print(obj)