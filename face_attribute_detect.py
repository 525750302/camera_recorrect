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
        
        self.original_box_id = []
        self.rotate_box_id = []
        self.box_location_x = []
        self.box_location_y = []
        self.vote_age = []
        self.vote_gender = []
        self.vote_count = []
        
        self.image_center_x = 0
        self.image_center_y = 0

    def detect_age_and_gender(self, id):
        img_path = self.img_PATH + str(id) + ".png"
        #对输入的图片进行预处理，确保输入为224*224 不足的部分会进行补足
        #img_objs = functions.extract_faces(img_path, detector_backend = "skip")
        #img_content = img_objs[0][0]
        #cv2.imwrite("C:/Users/XIR1SBY/Desktop/camera/yolo/test.png",img_content)
        objs = DeepFace.analyze(img_path , 
        actions = ['age', 'gender'],
        detector_backend="mediapipe",
        enforce_detection = True)
        if objs == -1:
            return -1,-1,-1
        print(objs[0])
        x = objs[0]["region"]["x"]
        y = objs[0]["region"]["y"]
        w = objs[0]["region"]["w"]
        h = objs[0]["region"]["h"]
        gape_origin_frame = cv2.imread(img_path)
        target = gape_origin_frame[max(int(y), 0):int(y + h),max(int(x), 0):int(x + w)]
        cv2.imwrite("C:/Users/XIR1SBY/Desktop/camera/yolo/detect_target.png", target)
        return objs[0]["age"],objs[0]["dominant_gender"],objs[0]["gender"]
    
    def show_result(self, ages,dominant_genders, genders,ids,person_ids):
        #显示最终结果图片并且加上年龄和性别的结果 查看FPS
        cTime = time.time() #处理完一帧图像的时间
        fps = 1/(cTime-self.pTime)
        self.pTime = cTime  #重置起始时间

        annotated_origin_frame = cv2.imread(self.show_path_origin)
        annotated_rotated_frame = cv2.imread(self.show_path_rotate)
        # 在视频上显示fps信息，先转换成整数再变成字符串形式，文本显示坐标，文本字体，文本大小
        cv2.putText(annotated_origin_frame, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)  
        cv2.putText(annotated_rotated_frame, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3) 
        (self.image_center_x,self.image_center_y,_) = annotated_origin_frame.shape
        self.image_center_x = self.image_center_x / 2
        self.image_center_y = self.image_center_y / 2
                
        id_num = len(ids)
        # 在视频上显示年龄和性别信息，结果的文本，文本显示坐标，文本字体，文本大小
        for i in range(len(person_ids)):
            id = person_ids[i]
            location_data_PATH = self.location_data_PATH + str(id) + ".txt"
            txt_file = open(location_data_PATH,'r')
            flag = int(txt_file.readline())
            trak_id = int(txt_file.readline())
            center_x = int(txt_file.readline())
            center_y = int(txt_file.readline())
            if flag == 0:
                a = center_x
                center_x = center_y
                center_y = a
            txt_file.close()
            result_string = str()
            # 在此处更新BOX信息
            vote_id = -1
            vote_id = self.check_vote_data(center_x,center_y,trak_id,flag)
            # 如果识别出了年龄那么进行更新
            for j in range(id_num):
                if ids[j] == id and vote_id >= 0:
                    self.vote_data(vote_id,ages[j],dominant_genders[j],center_x, center_y)
            target_no_in_class = -1
            if flag == 1:
                for j in range(len(self.original_box_id)):
                    if trak_id == self.original_box_id[j]:
                        target_no_in_class = j
                if target_no_in_class>= 0 and self.vote_age[target_no_in_class] > 0:
                    result_string = str(str(int(self.vote_age[target_no_in_class])) +"," + str(self.num_to_gender(self.vote_gender[target_no_in_class])))
            elif flag == 0:
                for j in range(len(self.rotate_box_id)):
                    if trak_id == self.rotate_box_id[j]:
                        target_no_in_class = j
                if target_no_in_class>= 0 and self.vote_age[target_no_in_class] > 0:
                    result_string = str(str(int(self.vote_age[target_no_in_class])) +"," + str(self.num_to_gender(self.vote_gender[target_no_in_class])))
            cv2.putText(annotated_origin_frame, result_string, (center_x,center_y), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 2)  
            
        
        # 显示图像，输入窗口名及图像数据
        cv2.imshow('image_origin', annotated_origin_frame)   
        cv2.imshow('image_rotated', annotated_rotated_frame)   
        cv2.waitKey(10)
    
    # 持续追踪BOX并且进行更新
    def check_vote_data(self, center_x, center_y,trak_id,flag):
        min_no = -1
        distance = 5000
        
        if flag == 1:
            for j in range(len(self.original_box_id)):
                if trak_id == self.original_box_id[j]:
                    min_no = j
        elif flag == 0:
            for j in range(len(self.rotate_box_id)):
                if trak_id == self.rotate_box_id[j]:
                    min_no = j
        if min_no == -1:
            for i in range(len(self.box_location_x)):
                target_x = self.box_location_x[i]
                target_y = self.box_location_y[i]
                if pow(center_x - target_x,2) + pow(center_y - target_y,2) < distance:
                    min_no = i 
                    distance = pow(center_x - target_x,2) + pow(center_y - target_y,2)
        
        if min_no != -1:
            if flag == 1:
                # 之前已经有编号占据位置了 需要重新制作编号记录
                if self.original_box_id[min_no] != -1 and self.original_box_id[min_no]!=trak_id:
                    min_no = -1
                else:
                    if self.original_box_id[min_no] == -1:
                        self.original_box_id[min_no] = trak_id
                    self.box_location_x[min_no] = center_x
                    self.box_location_y[min_no] = center_y
            elif flag == 0:
                if self.rotate_box_id[min_no] != -1 and self.rotate_box_id[min_no]!=trak_id:
                    min_no = -1
                else:
                    if self.rotate_box_id[min_no] == -1:
                        self.rotate_box_id[min_no] = trak_id
                    self.box_location_x[min_no] = center_x
                    self.box_location_y[min_no] = center_y
        
        if min_no == -1:
            self.box_location_x.append(center_x)
            self.box_location_y.append(center_y)
            self.vote_age.append(0)
            self.vote_gender.append(0)
            self.vote_count.append(0)
            # 记录编号
            if flag == 1:
                self.original_box_id.append(trak_id)
                self.rotate_box_id.append(-1)
            elif flag == 0:
                self.original_box_id.append(-1)
                self.rotate_box_id.append(trak_id)
            min_no = len(self.box_location_x) - 1
        return min_no
    
    # 更新目标位置
    def vote_data(self,vote_id, age,gender,center_x,center_y):
        center_distance = abs(int(center_x - self.image_center_x)) + abs(int(center_y - self.image_center_y))
        self.vote_age[vote_id] = (self.vote_age[vote_id] * self.vote_count[vote_id] + age*center_distance) / (self.vote_count[vote_id] + center_distance)
        self.vote_gender[vote_id] = (self.vote_gender[vote_id] * self.vote_count[vote_id] + self.gender_to_num(gender) * center_distance) / (self.vote_count[vote_id] + center_distance)
        self.vote_count[vote_id] = self.vote_count[vote_id] + center_distance
    
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