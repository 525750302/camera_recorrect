from PIL import Image
import os
import cv2

def colorful_to_single(input_img_path, output_img_path):
    """
    彩色图转单色图
    :param input_img_path: 图片路径
    :param output_img_path: 输出图片路径
    """
 
    img = cv2.imread(input_img_path)
    # 转化为黑白图片
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cv2.imwrite(output_img_path,img) 
    
dataset_dir = 'C:/datasets/VisDrone/VisDrone2019-DET-train/images_origin'
output_dir = 'C:/datasets/VisDrone/VisDrone2019-DET-train/images'
 
# 获得需要转化的图片路径并生成目标路径
image_filenames = [(
    os.path.join(dataset_dir, file_dir),
    os.path.join(output_dir, file_dir)
) for file_dir in os.listdir(dataset_dir)]
 
# 转化所有图片
for path in image_filenames:
    colorful_to_single(path[0], path[1])