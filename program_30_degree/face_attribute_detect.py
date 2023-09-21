from deepface import DeepFace
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons import functions, realtime, distance as dst
import numpy as np
import cv2
import time

class deep_face_detect():
    def __init__(self):
        self.img_gape_PATH = "C:/Users/XIR1SBY/Desktop/camera/program_30_degree/gape_picture_"
        self.img_face_PATH = "C:/Users/XIR1SBY/Desktop/camera/program_30_degree/gape_picture_"
        self.show_path_origin = "C:/Users/XIR1SBY/Desktop/camera/program_30_degree/origin_frame.png"
        self.show_path_rotate = "C:/Users/XIR1SBY/Desktop/camera/program_30_degree/origin_frame_rotate.png"
        self.pTime = 0 
        self.models = {}
        # build age and gender model
        self.models["age"] = DeepFace.build_model("Age")
        self.models["gender"] = DeepFace.build_model("Gender")
        self.location_data_PATH = "C:/Users/XIR1SBY/Desktop/camera/program_30_degree/box_data_"
        
        # get human box location data from box_data_X.txt
        # and record the id get from other program to record age and gender to vote
        self.original_box_id = []
        self.rotate_box_id = []
        self.box_location_x = []
        self.box_location_y = []
        
        # Using weighted methods to determine the age and gender of person
        self.vote_age = []
        self.vote_gender = []
        self.vote_count = []
        
        # save the size data of the wholly picture
        self.image_center_x = 0
        self.image_center_y = 0

    def detect_age_and_gender(self, id):
        img_gape_path = self.img_gape_PATH + str(id) + ".png"
        img_face_path = self.img_face_PATH + str(id) + ".png"
        #对输入的图片进行预处理，确保输入为224*224 不足的部分会进行补足
        #img_objs = functions.extract_faces(img_path, detector_backend = "skip")
        #img_content = img_objs[0][0]
        #cv2.imwrite("C:/Users/XIR1SBY/Desktop/camera/yolo/test.png",img_content)
        
        # analyze the age and gender using deepface
        # befrore analyze Deepface will use mediapipe Face detection model to check if there is a face image in the face_X.image
        # please note that the Face detection model is different to pose landmark detection model which is used in Mediapipe_recoginize.py
        # https://developers.google.com/mediapipe/solutions/vision/face_detector
        objs = DeepFace.analyze(img_face_path , 
        actions = ['age', 'gender'],
        detector_backend="mediapipe",
        enforce_detection = True)
        # the weight of the result from using mediapipe is set as 0.5
        flag_model = 0.5
        # if the mediapipe Face detection model failed to detect face then use the retinaface model
        # https://github.com/serengil/retinaface
        if objs == -1:
            objs = DeepFace.analyze(img_gape_path ,
            actions = ['age', 'gender'],
            detector_backend="retinaface",
            enforce_detection = True)
            # the weight of the result from using mediapipe is set as 1
            flag_model = 1
            # if it still failed in detecting face then return -1 to show the fail
            if objs == -1:
                return -1,-1,-1,-1
            
        #-----------------------------------------------
        print("--------------------objs:",objs[0])
        print("--------------------objs:",objs[0]["age"],objs[0]["dominant_gender"],objs[0]["gender"],flag_model)
        x = objs[0]["region"]["x"]
        y = objs[0]["region"]["y"]
        w = objs[0]["region"]["w"]
        h = objs[0]["region"]["h"]
        
        # gape the face which is inputed to the Deepface model
        if flag_model == 1:
            gape_origin_frame = cv2.imread(img_gape_path)
        else:
            gape_origin_frame = cv2.imread(img_face_path)
        target = gape_origin_frame[max(int(y), 0):int(y + h),max(int(x), 0):int(x + w)]
        cv2.imwrite("C:/Users/XIR1SBY/Desktop/camera/yolo/detect_target.png", target)
        # return the age and gender and the weight of the result
        return objs[0]["age"],objs[0]["dominant_gender"],objs[0]["gender"],flag_model
    
    def show_result(self, ages,dominant_genders, genders,ids,person_ids,flag_model):
        # Show final result images and add age and gender results with FPS
        # calculate FPS
        cTime = time.time() #处理完一帧图像的时间
        fps = 1/(cTime-self.pTime)
        self.pTime = cTime  #重置起始时间

        annotated_origin_frame = cv2.imread(self.show_path_origin)
        annotated_rotated_frame = cv2.imread(self.show_path_rotate)
        # show FPS in the video
        cv2.putText(annotated_origin_frame, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)  
        cv2.putText(annotated_rotated_frame, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3) 
        
        # get size data and the center ponit location of the original image
        (self.image_center_y,self.image_center_x,_) = annotated_origin_frame.shape
        self.image_center_x = self.image_center_x / 2
                
        id_num = len(ids)
        # get box data from box_data_X.txt
        for i in range(len(person_ids)):
            id = person_ids[i]
            location_data_PATH = self.location_data_PATH + str(id) + ".txt"
            txt_file = open(location_data_PATH,'r')
            flag = int(txt_file.readline())
            trak_id = int(txt_file.readline())
            center_x = int(txt_file.readline())
            center_y = int(txt_file.readline())
            
            # if the box data was get from rotated image the rotate it 
            if flag == 0:
                a = center_x
                center_x = center_y
                center_y = a
            txt_file.close()
            result_string = str()
            
            # Find out if there are past records for BOX with this number
            # if not create record to save the Box
            vote_id = -1
            vote_id = self.check_vote_data(center_x,center_y,trak_id,flag)
            print("vote_id:",vote_id,self.vote_age)
            
            # decide the age and gender of human using weight of the result from deepface
            for j in range(id_num):
                if ids[j] == id and vote_id >= 0:
                    self.vote_data(vote_id,ages[j],dominant_genders[j],genders[j],center_x, center_y,flag_model[j])
            
            # out put vote result and show in video
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
    
    # Find out if there are any past records for BOX with this number
    # if not create record to save the Box
    def check_vote_data(self, center_x, center_y,trak_id,flag):
        min_no = -1
        distance = 5000
        
        # check if there is a record
        if flag == 1:
            for j in range(len(self.original_box_id)):
                if trak_id == self.original_box_id[j]:
                    min_no = j
        elif flag == 0:
            for j in range(len(self.rotate_box_id)):
                if trak_id == self.rotate_box_id[j]:
                    min_no = j
        
        # if not check if there is a record which is near 
        # if there is then it may considered as same no
        if min_no == -1:
            for i in range(len(self.box_location_x)):
                target_x = self.box_location_x[i]
                target_y = self.box_location_y[i]
                if pow(center_x - target_x,2) + pow(center_y - target_y,2) < distance:
                    min_no = i 
                    distance = pow(center_x - target_x,2) + pow(center_y - target_y,2)
        
        # if find a record which is near
        if min_no != -1:
            if flag == 1:
                # If there is other data get from the same image then it needs to creat a new record
                if self.original_box_id[min_no] != -1 and self.original_box_id[min_no]!=trak_id:
                    min_no = -1
                else:
                    # If the data in the record from the original image is null then fill in 
                    if self.original_box_id[min_no] == -1:
                        self.original_box_id[min_no] = trak_id
                    self.box_location_x[min_no] = center_x
                    self.box_location_y[min_no] = center_y
            elif flag == 0:
                # If there is other data get from the same image then it needs to creat a new record
                if self.rotate_box_id[min_no] != -1 and self.rotate_box_id[min_no]!=trak_id:
                    min_no = -1
                else:
                    # If the data in the record from the rotated image is null then fill in 
                    if self.rotate_box_id[min_no] == -1:
                        self.rotate_box_id[min_no] = trak_id
                    self.box_location_x[min_no] = center_x
                    self.box_location_y[min_no] = center_y
        
        # still no creat new record
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
        
        # return record no to vote age and gender data
        return min_no
    
    # 更新目标位置
    def vote_data(self,vote_id, age,gender,log_gender,center_x,center_y,flag_model):
        # the futher the human from the center point the higher of weight will be 
        center_distance = int(flag_model * (int(pow(center_x - self.image_center_x,2)) + int(pow(center_y - self.image_center_y,2))))
        print("center_distance:",center_distance,center_x,self.image_center_x,center_y,self.image_center_y)
        if center_distance<int(250000 * flag_model):
            return 
        center_distance = center_distance - int(flag_model * 100000)
        
        # save data and output
        txt_path = "C:/Users/XIR1SBY/Desktop/camera/yolo/result.txt"
        txt_file = open(txt_path,'a')
        text = str(age) + "," + str(center_distance) + "," + str(log_gender) + "," + str(int(center_x)) + "," + str(int(self.image_center_x)) + "," + str(int(center_y)) + "," +  str(int(self.image_center_y)) + "," + str(int(self.vote_count[vote_id])) +"," +str(flag_model)
        txt_file.write(str(text))
        txt_file.write('\r')
        txt_file.close()
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