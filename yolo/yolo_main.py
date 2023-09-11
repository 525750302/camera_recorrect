import cv2
from ultralytics import YOLO
import os
import time
from collections import defaultdict
import numpy as np
from PIL import Image
import cut_picture_from_top

class yolo():
    def __init__(self, cap):
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        # Load the YOLOv8 model
        self.model_origin = YOLO('C:/Users/XIR1SBY/Desktop/camera/yolo/yolov8s.pt')
        self.model_rotated = YOLO('C:/Users/XIR1SBY/Desktop/camera/yolo/yolov8s.pt')
        # Open the video file
        self.cap = cap
        # self.cut_function = cut_picture_from_top.cut_picture_from_top_function()
        self.gape_picture_PATH = "C:/Users/XIR1SBY/Desktop/camera/yolo/"
        self.pTime = 0  # 设置第一帧开始处理的起始时间
    def get_one_picture(self):
        # Read a frame from the video
        success, origin_frame = self.cap.read()
        #success = True
        #origin_frame = cv2.imread(self.cap)
        PATH = self.gape_picture_PATH + "origin_frame.png"
        cv2.imwrite(PATH, origin_frame)
        #test
        origin_frame = cv2.imread(PATH)
        
        # origin_frame = self.cut_function.change_from_top(origin_frame)
        print(success)
        
        #--------------------------------------
        rotated_frame = cv2.flip(origin_frame, 1)
        rotated_frame = cv2.rotate(rotated_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        PATH = self.gape_picture_PATH + "origin_frame_rotate.png"
        cv2.imwrite(PATH, rotated_frame)
        track_ids_origin = []
        track_ids_rotated = []
        if success:
            # Run YOLOv8 tracking on the origin_frame, persisting tracks between origin_frames
            results_origin = self.model_origin.track(origin_frame, persist=True, conf=0.3, iou=0.5)
            # Visualize the results_origin on the origin_frame
            annotated_origin_frame = results_origin[0].plot()
            # Get the boxes and track IDs
            boxes_origin = results_origin[0].boxes.xywh.cpu()
            print("id:",results_origin[0].boxes.id)
            if results_origin[0].boxes.id != None:
                track_ids_origin = results_origin[0].boxes.id.int().cpu().tolist()
                class_name_origin = results_origin[0].boxes.cls.int().cpu().tolist()
                
            # Run YOLOv8 tracking on the origin_frame, persisting tracks between origin_frames
            results_rotated = self.model_rotated.track(rotated_frame, persist=True, conf=0.3, iou=0.5)
            # Visualize the results_origin on the origin_frame
            annotated_rotated_frame = results_rotated[0].plot()
            # Get the boxes and track IDs
            boxes_rotated = results_rotated[0].boxes.xywh.cpu()
            print("id:",results_rotated[0].boxes.id)
            if results_rotated[0].boxes.id != None:
                track_ids_rotated = results_rotated[0].boxes.id.int().cpu().tolist()
                class_name_rotated = results_rotated[0].boxes.cls.int().cpu().tolist()
            

        # 查看FPS
        cTime = time.time() #处理完一帧图像的时间
        fps = 1/(cTime-self.pTime)
        self.pTime = cTime  #重置起始时间

        # 在视频上显示fps信息，先转换成整数再变成字符串形式，文本显示坐标，文本字体，文本大小
        cv2.putText(annotated_origin_frame, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)  
        cv2.putText(annotated_rotated_frame, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)  
        # 显示图像，输入窗口名及图像数据
        cv2.imshow('image_YOLO_origin', annotated_origin_frame)   
        cv2.imshow('image_YOLO_rotated', annotated_rotated_frame)   
        cv2.waitKey(10)
        
        #-----------------------------
        no_origin = 0
        person_ids_origin = []
        person_x_origin = []
        person_y_origin = []
        person_w_origin = []
        person_h_origin = []
        person_ids_rotated = []
        person_x_rotated = []
        person_y_rotated = []
        person_w_rotated = []
        person_h_rotated = []
        
        if len(track_ids_origin) > 0:
            for boxes_origin, track_id in zip(boxes_origin, track_ids_origin):
                #检测是否是人类
                # If you want to get screenshots of other objects from the image you can add judgment conditions here
                # the classifation no of dog is 16 ; the classifation no of human is 0
                if class_name_origin[no_origin] == 0:
                    person_ids_origin = person_ids_origin.append(track_ids_origin[no_origin])
                    x, y, w, h = boxes_origin
                    x = int(x.item())
                    y = int(y.item())
                    w = int(w.item())
                    h = int(h.item())
                    person_x_origin.append(x)
                    person_y_origin.append(y)
                    person_w_origin.append(w)
                    person_h_origin.append(h)
                    no_origin = no_origin + 1
        
        no_rotated = 0
        if len(track_ids_rotated) > 0:
            for boxes_rotated, track_id in zip(boxes_rotated, track_ids_rotated):
                if class_name_rotated[no_rotated] == 0:
                    person_ids_rotated = person_ids_rotated.append(track_ids_rotated[no_rotated])
                    x, y, w, h = boxes_rotated
                    x = int(x.item())
                    y = int(y.item())
                    w = int(w.item())
                    h = int(h.item())
                    person_x_rotated.append(x)
                    person_y_rotated.append(y)
                    person_w_rotated.append(w)
                    person_h_rotated.append(h)
                    no_rotated = no_rotated + 1
        
        person_ids = []
        person_no = 0
        #如果原图像有人的识别，那么以原图像的数据为基础
        if no_origin > 0:
            for track_id in range(no_origin):
                print(track_id)
                self.save_person_inf(annotated_origin_frame, person_no, person_x_origin[track_id], person_y_origin[track_id], person_w_origin[track_id], person_h_origin[track_id])
                person_ids.append(person_no)
                person_no = person_no + 1
        #如果原图像没有人的识别，那么以旋转图像的数据为基础， 并且记得替换xywh
        elif no_rotated > 0:
            for track_id in range(no_rotated):
                self.save_person_inf(annotated_rotated_frame, person_no, person_x_rotated[track_id], person_y_rotated[track_id], person_w_rotated[track_id], person_h_rotated[track_id])
                person_ids.append(person_no)
                person_no = person_no + 1
        
        #如果原图和旋转图都识别到东西，那么检验是否有重复或者未识别
        if no_origin > 0 and no_rotated > 0: 
            for i in range(no_rotated):
                flag = 0
                for j in range(no_origin):
                    distance_x = person_x_origin[j] - person_y_rotated[i]
                    distance_y = person_y_origin[j] - person_x_rotated[i]
                    if distance_x*distance_x + distance_y*distance_y <= 100:
                        flag = 1
                        break
                #存在未识别到的图像
                if flag == 0:
                    self.save_person_inf(annotated_rotated_frame,person_no, person_x_rotated[i], person_y_rotated[i], person_w_rotated[i], person_h_rotated[i])
                    person_ids.append(person_no)
                    person_no = person_no + 1
        return person_ids
    
    #为了从图片中切出人脸，决定所有的都已竖直的正常人像为储存数据
    def save_person_inf(self, img, id, x, y, w, h):
        (max_img_size_y,max_img_size_x,img_c) = img.shape
        pictureName = self.gape_picture_PATH + "gape_picture_" + str(id) + ".png"
        cropped = img[max(int(y - h/2), 0):min(int(y + h/2), max_img_size_y),max(int(x - w/2), 0):min(int(x + w/2), max_img_size_x)]
        cv2.imwrite(pictureName, cropped)