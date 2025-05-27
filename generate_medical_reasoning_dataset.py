from datasets import load_dataset, load_from_disk

path_dataset = r'/data/@Zilong_Works/Data/SourceData/Dataset/VQA/vqa-rad'
 


# 从本地加载数据集
dataset = load_dataset(path_dataset)
print(dataset)


 # 取出训练集
dataset_tr = dataset['train']
print(dataset_tr)
print(dataset_tr[989]) # 查看单个样本

# 保存一张image到目标path下
path_result=r'/data/@Zilong_Works/Data/SourceData/Image/img.png'
image=dataset_tr[989]['image']
image.save(path_result)
import cv2
image_np = cv2.imread(path_result)
# 将image转化为类似cv2的格式
from PIL import Image
import numpy as np
image_PIL_to_np = np.array(image)
# 比较image_PIL_to_np和image_np是否完全一样
if np.array_equal(image_PIL_to_np, image_np):
    print("image_PIL_to_np和image_np完全一样")


pass