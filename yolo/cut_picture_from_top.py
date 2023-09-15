import cv2
import numpy as np
import math

class cut_picture_from_top_function():
    
    def __init__(self):
        pass
    
    def change_from_top(self,ori_img):
        PI = math.pi
        # 扫描线的长度 等效于新图像的宽度
        # 角度分割 360/角度单位 等效于新图的长度
        scan_length = 800
        scan_delt_angel = 0.2
        scan_width = int(360/scan_delt_angel)

        #注意y为竖向，x为横向
        (max_img_size_y,max_img_size_x,img_c) = ori_img.shape
        # print(max_img_size_y, max_img_size_x)
        center_y = max_img_size_y / 2
        center_x = max_img_size_x / 2

        new_img = np.zeros((scan_length, scan_width, 3), dtype=np.uint8)
        # 按照圆的极坐标赋值
        for row in range(new_img.shape[0]):
            for col in range(new_img.shape[1]):
                # 角度，最后的-0.1是用于优化结果，可以自行调整
                theta = PI * 2 / scan_width * (col + 1) - 0.2
                # 半径，减1防止超界
                CIRCLE_RADIUS = min(1/max(0.001,abs(math.cos(theta)))*center_x - 1, 1/max(abs(math.sin(theta)),0.001)*center_y - 1)
# 
                rho = max(0, CIRCLE_RADIUS -row -1 )

                # rho = center_y - row - 1
                x = int(center_x + rho * math.cos(theta) + 0.0)
                y = int(center_y - rho * math.sin(theta) + 0.0)
                # 赋值
                new_img[row, col, :] = ori_img[y, x, :]

        # 展示图片
        # cv2.imshow("Src frame", ori_img)
        # cv2.imshow("Log-Polar", new_img)
        cv2.imwrite("C:/Users/wuse/Desktop/camera_recorrect/camera_picture/Linear-Polar.png", new_img)
        return new_img