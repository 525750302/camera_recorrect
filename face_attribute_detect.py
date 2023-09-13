from deepface import DeepFace
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons import functions, realtime, distance as dst
import numpy as np
import cv2
import time

class deep_face_detect():
    def __init__(self):
        self.img_PATH = "C:/Users/XIR1SBY/Desktop/camera/yolo/gape_picture_"
        self.show_path_origin = "C:/Users/XIR1SBY/Desktop/camera/yolo/origin_frame.png"
        self.show_path_rotate = "C:/Users/XIR1SBY/Desktop/camera/yolo/origin_frame_rotate.png"
        self.pTime = 0 
        self.models = {}
        self.models["age"] = DeepFace.build_model("Age")
        self.models["gender"] = DeepFace.build_model("Gender")
        self.location_data_PATH = "C:/Users/XIR1SBY/Desktop/camera/yolo/box_data_"
        
        self.box_location_x = []
        self.box_location_y = []
        self.vote_age = []
        self.vote_gender = []
        self.vote_count = []

    def detect_age_and_gender(self, id):
        img_path = self.img_PATH + str(id) + ".png"
        #对输入的图片进行预处理，确保输入为224*224 不足的部分会进行补足
        #img_objs = functions.extract_faces(img_path, detector_backend = "skip")
        #img_content = img_objs[0][0]
        #cv2.imwrite("C:/Users/XIR1SBY/Desktop/camera/yolo/test.png",img_content)
        objs = DeepFace.analyze(img_path , 
        actions = ['age', 'gender'],
        detector_backend="mediapipe",
        enforce_detection = False)
        print(objs[0])
        return objs[0]["age"],objs[0]["dominant_gender"],objs[0]["gender"]
    
    def show_result(self, ages,dominant_genders, genders,ids):
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
            #投票决定真正的年龄和性别 并且实现追踪功能。
            #预计两次识别之间的帧数的移动距离不会超过1000
            vote_id = self.check_vote_data(center_x,center_y)
            
            if len(ages) > 0:    
                self.vote_data(vote_id,ages[i],dominant_genders[i])
                ages[i] = int(self.vote_age[vote_id])
                dominant_genders[i] = self.num_to_gender(self.vote_gender[vote_id])
                
            # 在视频上显示年龄和性别信息，结果的文本，文本显示坐标，文本字体，文本大小
            result_string = str(str(int(ages[i])) +"," + str(dominant_genders[i])) + "," + str(genders[i])
            print("data：",result_string)
            cv2.putText(annotated_origin_frame, result_string, (center_x,center_y), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)  
            
        
        # 显示图像，输入窗口名及图像数据
        cv2.imshow('image_origin', annotated_origin_frame)   
        cv2.imshow('image_rotated', annotated_rotated_frame)   
        cv2.waitKey(10)
    
    def check_vote_data(self, center_x, center_y):
        min_no = -1
        distance = 1000
        for i in range(len(self.box_location_x)):
            target_x = self.box_location_x[i]
            target_y = self.box_location_y[i]
            if (center_x - target_x)^2 + (center_y - target_y)^2 < distance:
                min_no = i 
                distance = (center_x - target_x)^2 + (center_y - target_y)^2
        if min_no == -1:
            self.box_location_x.append(center_x)
            self.box_location_y.append(center_y)
            self.vote_age.append(0)
            self.vote_gender.append(0)
            self.vote_count.append(0)
            min_no = len(self.box_location_x) - 1
        else:
            self.box_location_x[min_no] = center_x
            self.box_location_y[min_no] = center_y
        return min_no
    
    # 更新目标位置
    def vote_data(self,vote_id, age,gender):
        self.vote_age[vote_id] = (self.vote_age[vote_id] * self.vote_count[vote_id] + age) / (self.vote_count[vote_id] + 1)
        self.vote_gender[vote_id] = (self.vote_gender[vote_id] * self.vote_count[vote_id] + self.gender_to_num(gender)) / (self.vote_count[vote_id] + 1)
        self.vote_count[vote_id] = self.vote_count[vote_id] + 1
    
    def gender_to_num(self,gender):
        if gender == "Man":
            return 1
        elif gender == "Woman":
            return 0
    
    def num_to_gender(self,num):
        if num > 0.5:
            return "Man"
        else:
            return "Woman"