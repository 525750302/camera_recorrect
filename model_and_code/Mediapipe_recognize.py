import cv2
import mediapipe as mp
import time

class mediapipe_model():
    def __init__(self) -> None:
        # 导入姿态跟踪方法
        self.mpPose = mp.solutions.pose  # 姿态识别方法
        # decide how to recogenize face image here
        # the higher the min_detection_confidence is the harder to detect face image
        self.pose = self.mpPose.Pose(static_image_mode=True, # 静态图模式，False代表置信度高时继续跟踪，True代表实时跟踪检测新的结果
                           #upper_body_only=True,  # 是否只检测上半身
                           model_complexity = 1,
                           smooth_landmarks=False,  # 平滑，一般为True
                           min_detection_confidence=0.6, # 检测置信度
                           min_tracking_confidence=0.5)  # 跟踪置信度
        # 检测置信度大于0.5代表检测到了，若此时跟踪置信度大于0.5就继续跟踪，小于就沿用上一次，避免一次又一次重复使用模型

        # 导入绘图方法
        self.mpDraw = mp.solutions.drawing_utils

        self.pTime = 0  # 设置第一帧开始处理的起始时间
        #（2）处理每一帧图像
        self.lmlist = [] # 存放人体关键点信息
        
        self.picture_PATH = "C:/Users/XIR1SBY/Desktop/camera/yolo/"
        self.txt_PATH = "C:/Users/XIR1SBY/Desktop/camera/yolo/"
        
    def check_feacture(self, id):    
        #根据收到的ID来决定输入的数据
        #print("id:", id)
        picture_path = self.picture_PATH + "gape_picture_" + str(id) +".png"
        # 接收图片是否导入成功、帧图像
        img = cv2.imread(picture_path)
        txt_path = self.txt_PATH + "data_" + str(id) + ".txt"
        txt_file = open(txt_path,'a')
        # 将导入的BGR格式图像转为RGB格式
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 将图像传给姿态识别模型
        results = self.pose.process(imgRGB)

        # 查看体态关键点坐标，返回x,y,z,visibility
        # print(results.pose_landmarks)

        # 记录是否检测到人脸

        # 如果检测到体态就执行下面内容，没检测到就不执行
        # check result data and save data_X.txt
        if results.pose_landmarks:

            # 绘制姿态坐标点，img为画板，传入姿态点坐标，坐标连线
            self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

            # 获取32个人体关键点坐标, index记录是第几个关键点 事先清除文本的内容
            txt_file.truncate(0)
            for index, lm in enumerate(results.pose_landmarks.landmark):

                # 保存每帧图像的宽、高、通道数
                h, w, c = img.shape

                # 得到的关键点坐标x/y/z/visibility都是比例坐标，在[0,1]之间
                # 转换为像素坐标(cx,cy)，图像的实际长宽乘以比例，像素坐标一定是整数
                cx, cy = int(lm.x * w), int(lm.y * h)

                # save useful point data no.0 2 5 8 7 9 10
                # for details please check https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
                if index == 0 or index == 2 or index == 5 or index == 8 or index == 7 or index == 9 or index == 10:
                    txt_file.write(str(cx))
                    txt_file.write('\r')
                    txt_file.write(str(cy))
                    txt_file.write('\r')
                # 保存坐标信息
                self.lmlist.append((cx, cy))

                # 在关键点上画圆圈，img画板，以(cx,cy)为圆心，半径5，颜色绿色，填充圆圈
                cv2.circle(img, (cx,cy), 3, (0,255,0), cv2.FILLED)
            txt_file.close()
            PATH = self.picture_PATH + "recognize_result_"+ str(id) +".png"
            cv2.imwrite(PATH, img)
    
            #如果成功检测返回true
            return True
        # 查看FPS
        #cTime = time.time() #处理完一帧图像的时间
        #fps = 1/(cTime-self.pTime)
        #self.pTime = cTime  #重置起始时间

        # 在视频上显示fps信息，先转换成整数再变成字符串形式，文本显示坐标，文本字体，文本大小
        #cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)  

        # 显示图像，输入窗口名及图像数据
        #cv2.imshow('image', img)
        #cv2.waitKey(10)    
        txt_file.close()
        return False