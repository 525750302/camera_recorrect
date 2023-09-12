from deepface import DeepFace
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons import functions, realtime, distance as dst
import numpy as np
import cv2
import time

class deep_face_detect():
    def __init__(self):
        self.img_PATH = "C:/Users/XIR1SBY/Desktop/camera/yolo/face_"
        self.show_path_origin = "C:/Users/XIR1SBY/Desktop/camera/yolo/origin_frame.png"
        self.show_path_rotate = "C:/Users/XIR1SBY/Desktop/camera/yolo/origin_frame_rotate.png"
        self.pTime = 0 
        self.models = {}
        self.models["age"] = DeepFace.build_model("Age")
        self.models["gender"] = DeepFace.build_model("Gender")
        self.location_data_PATH = "C:/Users/XIR1SBY/Desktop/camera/yolo/box_data_"

    def detect_age_and_gender(self, id):
        img_path = self.img_PATH + str(id) + ".png"
        #对输入的图片进行预处理，确保输入为224*224 不足的部分会进行补足
        img_objs = functions.extract_faces(img_path, detector_backend = "skip")
        img_content = img_objs[0][0]
        obj = {}
        age_predictions = self.models["age"].predict(img_content, verbose=0)[0, :]
        apparent_age = Age.findApparentAge(age_predictions)
        # int cast is for exception - object of type 'float32' is not JSON serializable
        obj["age"] = int(apparent_age)
        gender_predictions = self.models["gender"].predict(img_content, verbose=0)[0, :]
        obj["gender"] = {}
        for i, gender_label in enumerate(Gender.labels):
            gender_prediction = 100 * gender_predictions[i]
            obj["gender"][gender_label] = gender_prediction
        obj["dominant_gender"] = Gender.labels[np.argmax(gender_predictions)]
        return obj["age"],obj["dominant_gender"]
    
    def show_result(self, ages, genders,ids):
        #显示最终结果图片并且加上年龄和性别的结果 查看FPS
        cTime = time.time() #处理完一帧图像的时间
        fps = 1/(cTime-self.pTime)
        self.pTime = cTime  #重置起始时间

        annotated_origin_frame = cv2.imread(self.show_path_origin)
        annotated_rotated_frame = cv2.imread(self.show_path_rotate)
        # 在视频上显示fps信息，先转换成整数再变成字符串形式，文本显示坐标，文本字体，文本大小
        cv2.putText(annotated_origin_frame, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)  
        cv2.putText(annotated_rotated_frame, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)  
        
        id_num = len(ids)
        for i in range(id_num):
            id = ids[i]
            location_data_PATH = self.location_data_PATH + str(id) + ".txt"
            txt_file = open(location_data_PATH,'r')
            flag = int(txt_file.readline())
            center_x = int(txt_file.readline())
            center_y = int(txt_file.readline())
            if flag == 0:
                a = center_x
                center_x = center_y
                center_y = a
            txt_file.close()
            # 在视频上显示年龄和性别信息，结果的文本，文本显示坐标，文本字体，文本大小
            result_string = str(str(int(ages[i])) +"," + str(genders[i]))
            print("data：",result_string)
            cv2.putText(annotated_origin_frame, result_string, (center_x,center_y), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)  
            
        
        # 显示图像，输入窗口名及图像数据
        cv2.imshow('image_origin', annotated_origin_frame)   
        cv2.imshow('image_rotated', annotated_rotated_frame)   
        cv2.waitKey(10)