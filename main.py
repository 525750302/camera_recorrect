import sys
sys.path.append(r'C:/Users/wuse/Desktop/camera_recorrect/yolo')
import yolo.yolo_main
import Mediapipe_recognize
import cut_face_from_picture
import face_attribute_detect
import threading
import time
import cv2

class resource_stack():
    id_stack = []
    YOLO_id = 0
    Mediapipe_id = 0
    cut_picture_id = 0
    deep_face_id = 0
    successful_checked_ids = []
    
    def change_YOLO_id(self,id):
        self.YOLO_id = id
    
    def Change_Mediapipe_id(self,id):
        self.Mediapipe_id = id
    
    def Change_cut_picture_id(self,id):
        self.cut_picture_id = id
        
    def Change_deep_face_id(self,id):
        self.deep_face_id = id
        
    def update_id_stack(self, ids):
        self.id_stack = ids
        
    def return_id(self,index):
        if index >= len(self.id_stack):
            print("error id")
            return -1
        return self.id_stack[index]
    
    def clear_id(self):
        self.id_stack.clear()
        self.successful_checked_ids.clear()
    
    def get_len_ids(self):
        return len(self.id_stack)
    
    def add_successful_checked_ids(self,id):
        if self.successful_checked_ids.count(id) > 0:
            return False
        else:
            self.successful_checked_ids.append(id)
    
    def remove_successful_checked_ids(self,id):
        if self.successful_checked_ids.count(id) > 0:
            self.successful_checked_ids.remove(id)
        else:
            return False
            
    def check_successful_checked_ids(self,id):
        if self.successful_checked_ids.count(id) > 0:
            return True
        else:
            return False

class Thread_YOLO (threading.Thread):
    def __init__(self, threadID, name, counter, cap):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.model = yolo.yolo_main.yolo(cap)
        self.name = name
        self.counter = counter
    def run(self):
        while True:
            print("Starting " + self.name)
           # 获得锁，成功获得锁定后返回True
           # 可选的timeout参数不填时将一直阻塞直到获得锁定
           # 否则超时后将返回False
            lockYOLO.acquire()
            global resource_controler
            ids = self.model.get_one_picture()
            resource_controler.update_id_stack(ids)
            # 释放锁
            print("END " + self.name)
            lockMedia.release()
            #print_time(self.name, self.counter)
        
class Thread_Mediapipe (threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.model = Mediapipe_recognize.mediapipe_model()
        self.threadID = threadID
        self.name = name
        self.counter = counter
    def run(self):
        while True:
            print("Starting " + self.name)
           # 获得锁，成功获得锁定后返回True
           # 可选的timeout参数不填时将一直阻塞直到获得锁定
           # 否则超时后将返回False
            lockMedia.acquire()
            global resource_controler
            resource_num = resource_controler.get_len_ids()
            if resource_num > 0:
                for i in range(resource_num):
                    id = resource_controler.return_id(i)
                    resource_controler.Change_Mediapipe_id(id)
                    flag_checked = self.model.check_feacture(id)
                    # 检查是否检测到特征点并且进行管理
                    if flag_checked == True:
                        resource_controler.add_successful_checked_ids(id)
                    elif flag_checked == False:
                        resource_controler.remove_successful_checked_ids(id)
            # 释放锁
            print("END " + self.name)
            lockcut.release()
            #print_time(self.name, self.counter)
            
class Thread_cut_face (threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.model = cut_face_from_picture.cut_face_from_picture()
        self.threadID = threadID
        self.name = name
        self.counter = counter
    def run(self):
        while True:
            print("Starting " + self.name)
           # 获得锁，成功获得锁定后返回True
           # 可选的timeout参数不填时将一直阻塞直到获得锁定
           # 否则超时后将返回False
            lockcut.acquire()
            global resource_controler
            resource_num = resource_controler.get_len_ids()
            if resource_num>0:
                for i in range(resource_num):
                    id = resource_controler.return_id(i)
                    if resource_controler.check_successful_checked_ids(id) == False:
                        continue
                    resource_controler.Change_cut_picture_id(id)
                    self.model.cut_picture(id)
            # 释放锁
            print("END " + self.name)
            lockdeepface.release()
            #print_time(self.name, self.counter)
            
class Thread_deep_face(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.model = face_attribute_detect.deep_face_detect()
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.ages = []
        self.dominant_genders = []
        self.genders = []
        
    def run(self):
        while True:
            print("Starting " + self.name)
            # 获得锁，成功获得锁定后返回True
            # 可选的timeout参数不填时将一直阻塞直到获得锁定
            # 否则超时后将返回False
            lockdeepface.acquire()
            global resource_controler
            resource_num = resource_controler.get_len_ids()
            self.ages.clear()
            self.dominant_genders.clear()
            self.genders.clear()
            usable_ids = []
            # 记录所有识别为人的BOX输出文件的编号
            person_ids = []
            if resource_num>0:
                for i in range(resource_num):
                    id = resource_controler.return_id(i)
                    person_ids.append(id)
                    if resource_controler.check_successful_checked_ids(id) == False:
                        continue
                    resource_controler.Change_deep_face_id(id)
                    #获得检测得到的年龄和性别
                    age, dominant_gender, gender = self.model.detect_age_and_gender(id)
                    if age < 0:
                        continue
                    self.ages.append(age)
                    self.dominant_genders.append(dominant_gender)
                    self.genders.append(gender)
                    usable_ids.append(id)
            
            #显示结果
            print("result:",self.ages,self.genders,usable_ids,person_ids)
            self.model.show_result(self.ages,self.dominant_genders,self.genders,usable_ids,person_ids)
            resource_controler.clear_id()
            # 释放锁
            print("END " + self.name)
            lockYOLO.release()
            #print_time(self.name, self.counter)
 
def print_time(threadName, delay):
    time.sleep(delay)
    t = time.time()
    nowTime = int(round(t * 1000))
    #print ("%s: %s  ms:%d" % (threadName, time.ctime(time.time()), nowTime))
 
lockYOLO=threading.Lock()
lockMedia=threading.Lock()
lockcut = threading.Lock()
lockdeepface = threading.Lock()
lockMedia.acquire()
lockcut.acquire()
lockdeepface.acquire()
threads = []

cap_path = "C:/Users/wuse/Desktop/camera_recorrect/camera_picture/xie.mp4"
cap = cv2.VideoCapture(cap_path)
# 创建新线程
thread1 = Thread_YOLO(1, "Thread-yolo", 0.01, cap)
thread2 = Thread_Mediapipe(2, "Thread-mediapipe", 0.01)
thread3 = Thread_cut_face(3, "Thread-cut-face", 0.01)
thread4 = Thread_deep_face(4, "Thread_deep_face", 0.01)

# 创建资源管理
resource_controler = resource_stack()
# 开启新线程
thread1.start()
thread2.start()
thread3.start()
thread4.start()
 
# 添加线程到线程列表
threads.append(thread1)
threads.append(thread2)
threads.append(thread3)
threads.append(thread4)
for t in threads:
    t.join()
