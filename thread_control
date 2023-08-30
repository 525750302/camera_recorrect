import sys
sys.path.append(r'C:/Users/XIR1SBY/Desktop/bosch_avp_camera_recorrect/yolo')
import yolo.yolo_main
import Mediapipe_recognize
import cut_face_from_picture
import threading
import time
import cv2

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
            self.model.get_one_picture()
            # 释放锁
            lockMedia.release()
            print_time(self.name, self.counter)
        
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
            self.model.check_feacture()
            # 释放锁
            lockcut.release()
            print_time(self.name, self.counter)
            
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
            self.model.cut_picture()
            # 释放锁
            lockYOLO.release()
            print_time(self.name, self.counter)
 
def print_time(threadName, delay):
    time.sleep(delay)
    print ("%s: %s" % (threadName, time.ctime(time.time())))
 
lockYOLO=threading.Lock()
lockMedia=threading.Lock()
lockcut = threading.Lock()
lockMedia.acquire()
lockcut.acquire()
threads = []

cap = cv2.VideoCapture(0)
# 创建新线程
thread1 = Thread_YOLO(1, "Thread-yolo", 0.1, cap)
thread2 = Thread_Mediapipe(2, "Thread-mediapipe", 0.1)
thread3 = Thread_cut_face(2, "Thread-mediapipe", 0.1)
# 开启新线程
thread1.start()
thread2.start()
thread3.start()
 
# 添加线程到线程列表
threads.append(thread1)
threads.append(thread2)
threads.append(thread3)
for t in threads:
    t.join()
