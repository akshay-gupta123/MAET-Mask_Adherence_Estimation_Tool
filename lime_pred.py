import torch
from lime import lime_image
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
import cv2
from model import get_model
import time
from utilities import box_iou

explainer = lime_image.LimeImageExplainer()

target = 0
target_pred = 0

def lime_result(img_num,address,target_id=0):
  '''
    Prediction for the selected target is obtained, and it's area is calculated
    If image size is upto 100 times of predicted box area, we padded the image 
    for making LIME to generate good explanation results. Else go with original oneself.
    
    "LimeImageExplainer" : Explains predictions on Image (i.e. matrix) data. 
    "explain_instance" : Generates explanations for a prediction. First, generate neighborhood 
    data by randomly perturbing features from the instance, then learn locally weighted linear 
    models on this neighborhood data to explain each of the classes in an interpretable way.
    "get_image_and_mask" : Return an image is a 3d numpy array and mask is a 2d numpy array that 
    can be used with skimage.segmentation.mark_boundaries. Quickshift algorithm is used for 
    segmentation. Top 3 features are saved.
    
    Args
        img_num: String- the timestamp of image uploaded by user
        address: String- the timestamp of YOLO model's output image
        target_id: int- targeted face-index for selection 

  '''
  global target_pred     
  target = target_id
  
  img = cv2.imread('./static/'+str(img_num)+'.jpg')[:,:,::-1]
  img = cv2.resize(img,(640,640),interpolation=cv2.INTER_CUBIC)
  lime_img = img
  result1 = get_model(img)
  prediction1 = result1.xyxy[0]
  prediction1 = prediction1[prediction1[:,0].argsort()]
  
  try:
      target_pred = prediction1[target]
      area = (img.shape[0]*img.shape[1])/((target_pred[2]-target_pred[0])*(target_pred[3]-target_pred[1]))
      ratiox,ratioy = 1,1
  except :
      area = 0    
          
  if prediction1.shape[0]<target or area < 100 : 
    ratiox,ratioy = 150*416/(img.shape[1]+300),150*416/(img.shape[0]+300)
    img2 = cv2.copyMakeBorder(img,150,150,150,150,cv2.BORDER_CONSTANT)
    img2 = cv2.resize(img2,(416,416),interpolation=cv2.INTER_CUBIC)
    result2 = get_model(img2)
    prediction2 = result2.xyxy[0]
    prediction2 = prediction2[prediction2[:,0].argsort()]
    
    if prediction2.shape[0]>target : 
      target_pred = prediction2[target]
      lime_img = img2
    else :
      target_pred = prediction1[target]
      lime_img = img
    
  explanation = explainer.explain_instance(np.array(lime_img), lime_pred , top_labels=2, hide_color=0, num_samples=100,batch_size=100)
  
  for i in range(1,4):
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=i, hide_rest=False)
    
    temp = temp[int(ratioy):-int(ratioy),int(ratiox):-int(ratiox)]
    mask = mask[int(ratioy):-int(ratioy),int(ratiox):-int(ratiox)]

    img_boundry = mark_boundaries(temp/255.0, mask)
    f, axarr = plt.subplots(1,1)
    axarr.axis('off')
    axarr.imshow(img_boundry,aspect="auto")
    f.tight_layout()
    name = f'./static/{address}_{target}_{i}.jpg'
    plt.savefig(name)
    
    plt.close(f)
    
  return 

def lime_pred(images):
  '''
    Firstly predictions are generated for a batch of image and for each image maximum iou box with
    targeted box (target_pred) is selected having same class and updated it's confidence as iou*confidence
    
    Args
        images: a batch of 3D numpy array
    Returns
        prediction: a numpy array
  '''    
  global target_pred
  prediction = []
  
  for image in images:
    logits = get_model(image)
    output = logits.xyxy[0]
    
    if output.shape[0]:
      correct_class_predictions = output[...,5] == target_pred[5]
      correctly_labeled_boxes = output[correct_class_predictions]
      
      if correctly_labeled_boxes.shape[0]:
        iou_with_target, _idx = box_iou(correctly_labeled_boxes[:,:4], target_pred.unsqueeze(0)[:,:4]).max(1)
        index_best_box_in_correct_class = torch.argmax(iou_with_target)
        index_best_box_in_output = torch.where(output[...,5] == target_pred[5])[0][index_best_box_in_correct_class]

        result =  output[index_best_box_in_output][4]*iou_with_target[index_best_box_in_correct_class]
        prediction.append([result])
        
      else :
          prediction.append([0])
    else :
      prediction.append([0])
  return np.array(prediction)

