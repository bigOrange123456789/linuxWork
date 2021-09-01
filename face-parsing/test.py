#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], 
                   [255, 85, 0], 
                   [255, 170, 0],
                   [255, 0, 85], 
                   [255, 0, 170],
                   [0, 255, 0], #5
                   [85, 255, 0], 
                   [170, 255, 0],
                   [0, 255, 85], 
                   [0, 255, 170],
                   [0, 0, 255], #10
                   [85, 0, 255], 
                   [170, 0, 255],
                   [0, 85, 255], 
                   [0, 170, 255],
                   [255, 255, 0], #15
                   [255, 255, 85], 
                   [255, 255, 170],
                   [255, 0, 255], 
                   [255, 85, 255], 
                   [255, 170, 255],#20
                   [0, 255, 255], 
                   [85, 255, 255], 
                   [170, 255, 255]
                   ]#len(part_colors)=24
    print("part_colors.shape")
    print(len(part_colors))
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite('../output/result.png', vis_parsing_anno)
        cv2.imwrite('../output/result.jpg', vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im
def save(parsing):
            print(parsing.shape)
            result=parsing.tolist()
            
            file=open("../output/result.json","w")
            import json
            j={"result":result}
            json.dump(j,file)

def evaluate( path=os.getcwd()+'/../input/4.jpg', cp='79999_iter.pth'):
    n_classes = 19 # 分为19个区域
    net = BiSeNet(n_classes=n_classes)#创建网络
    net.cuda()#GPU计算
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()#进行预测

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
            img = Image.open(path)#读取图片资源
            image = img.resize((500, 500), Image.BILINEAR)#将图片的大小标准化
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            print("net(img)")
            print(len(net(img)))
            out = net(img)[0]#net(img)是元组类型，有3个元素
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            save(parsing)
            
            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path="../output")



if __name__ == "__main__":
    evaluate()


