import cv2
from ultralytics import YOLO
import os
import time
from collections import defaultdict
import numpy as np
from PIL import Image

class yolo():
    def __init__(self, cap):
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        # Load the YOLOv8 model
        self.model = YOLO('C:/Users/XIR1SBY/Desktop/bosch_avp_camera_recorrect/yolo/yolov8n.pt')
        # Open the video file
        self.cap = cap

        self.gape_picture_PATH = "C:/Users/XIR1SBY/Desktop/bosch_avp_camera_recorrect/yolo/"
        self.pTime = 0  # 设置第一帧开始处理的起始时间

    def get_one_picture(self):
        # Read a frame from the video
        success, frame = self.cap.read()
        cv2.imwrite("C:/Users/XIR1SBY/Desktop/bosch_avp_camera_recorrect/yolo/frame.png", frame)
        print(success)
        (max_img_size_y,max_img_size_x,img_c) = frame.shape
        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = self.model.track(frame, persist=True)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

        # 查看FPS
        cTime = time.time() #处理完一帧图像的时间
        fps = 1/(cTime-self.pTime)
        self.pTime = cTime  #重置起始时间

        # 在视频上显示fps信息，先转换成整数再变成字符串形式，文本显示坐标，文本字体，文本大小
        cv2.putText(annotated_frame, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)  
        # 显示图像，输入窗口名及图像数据
        cv2.imshow('image_YOLO', annotated_frame)    
        cv2.waitKey(10)
        for box, track_id in zip(boxes, track_ids):
            pictureName = self.gape_picture_PATH + "gape_picture_" + str(track_id) + ".png"
            print(pictureName)
            x, y, w, h = box
            x = int(x.item())
            y = int(y.item())
            w = int(w.item())
            h = int(h.item())
            #print (max(int(y - h/2), 0),min(int(y + h/2), max_img_size_y),max(int(x - w/2), 0), min(int(x + w/2), max_img_size_x))
            #cropped = annotated_frame[0:int(max_img_size_x/2), 0:int(max_img_size_y/2)]
            cropped = annotated_frame[max(int(y - h/2), 0):min(int(y + h/2), max_img_size_y),max(int(x - w/2), 0):min(int(x + w/2), max_img_size_x)]
            cv2.imwrite(pictureName, cropped)
            