import os
import time
import torch

def delete_file():
    '''
        Scans the static directory and deletes the model's output image and
        explaination results after 20 minutes by comparing their timestamp 
        with current time. Done so tool size doesn't grow with time.
    '''
    for file in os.listdir('./static'):
            if file.endswith('.jpg'):
                num_ = len(file.split("_"))
                if num_ == 1:
                    file_name = file[:-5]
                else :
                    file_name = file[:-9]
            
                diff = time.time()*100000-int(file_name)
                if diff>120000000:
                    address = os.path.join('static',file)
                    os.remove(address)
                    
    f = open('./static/read.txt','r')
    if len(f.read().split(" "))>600:
        content = f.read().split(" ")[-67:]
        f = open('./static/read.txt','w')
        f.truncate(0)
        for c in content:
            f.write(c)
            f.write(" ")

def box_iou(box1, box2):
    '''
        calculates the iou of for each box in box1 and box2.
        
        Args
            box1: tensor-predictions 
            box2: tensor-predictions 
    '''
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  
