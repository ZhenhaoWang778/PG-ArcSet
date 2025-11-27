import os
import cv2
import PIL
import torch
import pandas
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import TensorDataset, ConcatDataset

# 数据集路径
root_path = 'electronic_datas/Blurred_datas/'
branch_path = 'branch_type'
net_path = 'net_type'
line_path = 'line_type'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_path = 'electronic_datas/processed_datas/'


def extract_arc_region(image_path, output_datas_path, threshold_value):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像 {image_path}")
        return

        # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用阈值
    if threshold_value is None:
        threshold_value, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # print(f"使用OTSU自动阈值: {threshold_value}")
    else:
        _, binary_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        # print(f"使用手动阈值: {threshold_value}")

    # 形态学处理
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 查找所有亮部区域的轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # print("未检测到亮部区域")

        img = PIL.Image.open(image_path)
        img = img.resize((224, 224))
        img.save(output_datas_path)
        os.remove(output_datas_path)
        # print(f"裁切后的原图区域已保存至: {output_datas_path}")
        return img

        # 找到所有亮部区域的最小边界矩形
    x_min, y_min = image.shape[1], image.shape[0]
    x_max, y_max = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

        # 添加一些边距
    margin = 10
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(image.shape[1], x_max + margin)
    y_max = min(image.shape[0], y_max + margin)

    # 计算裁切区域的宽度和高度
    crop_width = x_max - x_min
    crop_height = y_max - y_min

    # 确保裁切区域至少为224×224
    if crop_width < 224 or crop_height < 224:
        # 计算需要扩展的尺寸
        expand_x = max(0, (224 - crop_width) // 2)
        expand_y = max(0, (224 - crop_height) // 2)

        # 扩展裁切区域，确保不超过图像边界
        x_min = max(0, x_min - expand_x)
        y_min = max(0, y_min - expand_y)
        x_max = min(image.shape[1], x_max + expand_x)
        y_max = min(image.shape[0], y_max + expand_y)

        # 如果扩展后仍然小于224，则继续调整
        if (x_max - x_min) < 224:
            if x_min == 0:
                x_max = min(image.shape[1], 224)
            else:
                x_min = max(0, x_max - 224)

        if (y_max - y_min) < 224:
            if y_min == 0:
                y_max = min(image.shape[0], 224)
            else:
                y_min = max(0, y_max - 224)
                # 直接从原图裁切区域（不进行亮部提取处理）
    cropped_region = image[y_min:y_max, x_min:x_max]

    # 保存裁切后的原图区域
    # cv2.imwrite(output_datas_path, cropped_region)
    # print(f"裁切后的原图区域已保存至: {output_datas_path}")
    cropped_region = Image.fromarray(cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB))
    cropped_region = cropped_region.resize((224, 224))
    cropped_region.save(output_datas_path)
    return cropped_region


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, label_path, train=True, transform=None):
        self.root_path = root_path
        self.label_path = label_path
        self.path = os.path.join(root_path, label_path)
        print(self.path)
        self.train_flag = train
        if transform is None:
            self.transform = torchvision.transforms.Compose(
                [
                    # torchvision.transforms.Resize(size=(100, 100)),  # 尺寸规范
                    torchvision.transforms.Resize(size=(224, 224)),  # 尺寸规范
                    torchvision.transforms.ToTensor(),  # 转化为tensor
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
                ])
        else:
            self.transform = transform
        self.path_list = os.listdir(self.path)  # 列出所有图片命名

    def __getitem__(self, idx):
        img_name = self.path_list[idx]
        img_item_path = os.path.join(self.root_path, self.label_path, img_name)
        label = self.label_path
        output_datas_path = os.path.join(output_path, label, img_name)
        # print(output_datas_path)
        img = extract_arc_region(img_item_path, output_datas_path, threshold_value=65)
        img = self.transform(img)  # 把图片转换成tensor
        label = self.label_path
        if label == "branch_type":
            label = 0
        elif label == "line_type":
            label = 1
        else:
            label = 2
        label = torch.tensor(label, dtype=torch.int64)  # 把标签转换成int64
        return img, label

    def __len__(self):
        return len(self.path_list)


branch_type_data = MyDataset(root_path, branch_path)
net_type_data = MyDataset(root_path, net_path)
line_type_data = MyDataset(root_path, line_path)

Img_PIL_Tensor = branch_type_data[0][0]
new_img_PIL = torchvision.transforms.ToPILImage()(Img_PIL_Tensor).convert('RGB')
plt.imshow(new_img_PIL)
plt.show()

# 80%训练集  20%测试集
train_datas = ConcatDataset([branch_type_data, net_type_data, line_type_data])
'''                                                                                                          
test_datas = ConcatDataset([branch_type_data, net_type_data,line_type_data])    #设置测试集                       
'''
train_size = int(0.8 * len(train_datas))
validate_size = len(train_datas) - train_size
train_datas, validate_datas = torch.utils.data.random_split(train_datas, [train_size, validate_size])

# 数据分批
# batch_size=32 每一个batch大小为32
# shuffle=True 打乱分组
# pin_memory=True 锁页内存，数据不会因内存不足，交换到虚拟内存中，能加快数据读入到GPU显存中.
# num_workers 线程数。 num_worker设置越大，加载batch就会很快，训练迭代结束可能下一轮batch已经加载好
# win10 设置会多线程可能会出现问题，一般设置0.
train_loader = torch.utils.data.DataLoader(train_datas, batch_size=32,
                                           shuffle=True, pin_memory=True, num_workers=0)
validate_loader = torch.utils.data.DataLoader(validate_datas, batch_size=32,
                                              shuffle=True, pin_memory=True, num_workers=0)
'''                                                                                                          
test_loader = torch.utils.data.DataLoader(test_datas, batch_size=32,                                         
                                            shuffle=False, pin_memory=True, num_workers=0) #载入测试集数据          
'''

# 处理所有图片
print("开始处理所有训练图片...")
for (images, labels) in enumerate(train_loader):
    print('处理中')
for batch_idx, (images, labels) in enumerate(validate_loader):
    print('处理中')
print("所有图片处理完成！")










