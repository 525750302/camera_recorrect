import cv2
from PIL import Image
import math

class cut_face_from_picture():
    def __init__(self):
        self.raw_pictiure_path = "C:/Users/XIR1SBY/Desktop/bosch_avp_camera_recorrect/yolo/gape_picture_"
        self.location_data_PATH = "C:/Users/XIR1SBY/Desktop/bosch_avp_camera_recorrect/yolo/data_"
        self.result_PATH = "C:/Users/XIR1SBY/Desktop/bosch_avp_camera_recorrect/yolo/face_"
    def cut_picture(self,id):
        raw_pictiure_path = self.raw_pictiure_path + str(id) + ".png"
        location_data_PATH = self.location_data_PATH + str(id) + ".txt"
        self.txt_file = open(location_data_PATH,'r')
        result_PATH = self.result_PATH + str(id) + ".png"
        point_index = [0,2,5,7,8,9, 10]
        point_location = []
        self.txt_file.seek(0, 0)
        #x横向 y竖向
        for i in point_index:
            x = int(self.txt_file.readline())
            y = int(self.txt_file.readline())
            point_location.append([x,y])
        #w为横向的宽 h 为纵向的长
        right_w = abs(point_location[0][0] - point_location[3][0]) + max(abs(point_location[0][0] - point_location[4][0])/2 , abs(point_location[0][0] - point_location[6][0])/2)
        left_w = abs(point_location[0][0] - point_location[4][0]) + max(abs(point_location[0][0] - point_location[3][0])/2 , abs(point_location[0][0] - point_location[5][0])/2)
        up_h = abs(point_location[0][1] - (point_location[1][1] + point_location[2][1]) / 2)*6 
        down_h = abs(point_location[0][1] - (point_location[5][1] + point_location[6][1]) / 2)*3
        center_point_x =  point_location[0][0]
        center_point_y =  point_location[0][1]

        img = cv2.imread(raw_pictiure_path)
        cut_face_image = img[max(0,int(center_point_y - up_h)):int(center_point_y + down_h),int(center_point_x - left_w):int(center_point_x + right_w)]
        cv2.imwrite(result_PATH, cut_face_image)
        print("finish")