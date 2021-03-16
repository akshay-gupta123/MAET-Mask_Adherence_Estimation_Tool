import torch
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from utilities import delete_file
from model import get_model
import numpy as np


def get_prediction(img_byte):
      '''
            Args
                 img_byte : image bytes of uploaded image
            Returns
                 img_address: timestamp of uploaded image
                 num_mask: number of people wearing mask in uploaded image
                 num_nmask: number of people not wearing mask in uploaded image 
                 level: safety level of uploaded image
                 model_img_address: timestamp of model's output image
                 prediction[0].shape: total number of faces detected in uploaded image 
      '''
      delete_file()

      img = Image.open(io.BytesIO(img_byte))
      img = img.convert('RGB')
      img_address = round(time.time()*1000000)
      img.save("./static/"+str(img_address)+".jpg")

      orign_img = cv2.imread("./static/"+str(img_address)+".jpg")[:,:,::-1] 
      resize_img = cv2.resize(orign_img,(640,640),interpolation=cv2.INTER_CUBIC)

      ratiox,ratioy = 150*640/(resize_img.shape[1]+300),150*640/(resize_img.shape[0]+300)
      pad_image = cv2.copyMakeBorder(resize_img,150,150,150,150,cv2.BORDER_CONSTANT)
      pad_image = cv2.resize(pad_image,(416,416),interpolation=cv2.INTER_CUBIC)

      results = get_model([resize_img,pad_image])
      prediction1 = results.xyxy[0]
      prediction1 = prediction1[prediction1[:,0].argsort()]
      prediction2 = results.xyxy[1]
      prediction2 = prediction2[prediction2[:,0].argsort()]

      if prediction1.shape[0] > prediction2.shape[0] or prediction1.shape[0]>= 10:
            prediction = prediction1
            plot_img = resize_img
            chosen_img =0
      else :
            prediction = prediction2
            plot_img = pad_image
            chosen_img = 1

      num_non_mask =torch.sum(prediction[:,5]==1).numpy()
      num_mask= prediction.shape[0]-num_non_mask
      proportn = num_mask/(num_mask+num_non_mask) if (num_mask+num_non_mask)!=0 else 0
      proportn = round(proportn,2)

      if proportn >=0.8:
            level="High"
      elif proportn>=0.5:
            level="Medium"
      else:
            level = "Low"  
            

      fig = plt.figure(figsize=(10, 10))
      ax = fig.add_subplot(1, 1, 1)
      fig.tight_layout()
      plt.imshow(plot_img,aspect='auto')
      plt.axis('off')

      for i, detection in enumerate(prediction):
            label = str(i)
            if int(detection[5])==1:
                  bbox = patches.Rectangle(detection[:2], detection[2]-detection[0], detection[3]-detection[1], linewidth=4, edgecolor='r', facecolor='none')
            else:
                  bbox = patches.Rectangle(detection[:2], detection[2]-detection[0], detection[3]-detection[1], linewidth=4, edgecolor='g', facecolor='none')

            plt.text(detection[0], detection[1], label, color="blue",fontsize=10, ha="center", va="center",bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
            ax.add_patch(bbox)

      model_img_address = round(time.time()*1000000)
      plt.savefig("./static/"+str(model_img_address)+".jpg")

      if chosen_img==1:
            ploted_img = cv2.imread("./static/"+str(model_img_address)+".jpg")
            ploted_img = cv2.resize(ploted_img,(640,640),interpolation=cv2.INTER_CUBIC)
            resizd_ploted = ploted_img[int(ratioy):-int(ratioy)-10,int(ratiox)+15:-int(ratiox)]
            model_img_address = round(time.time()*1000000)
            cv2.imwrite("./static/"+str(model_img_address)+".jpg",resizd_ploted)

      return img_address,num_mask,num_non_mask,level,model_img_address,prediction.shape[0]