# MAET-Mask_Adherence_Estimation_Tool

This repository is the official implementation of paper <em>"A Robust System to Detect and Explain Public Mask Wearing Behavior"</em> and it contains the code and the trained weights.

## Abstract 
COVID19 is a global health crisis during which mask wearing has emerged as an effective tool to combat the spread of disease. During this time, non-technical  users like health officials and school administrators need tools to know how widely people are wearing masks in public. We present a robust and efficient <strong>Mask Adherence Estimation Tool MAET</strong> based on the pre-trained <em>YOLOv5s</em> object detection model, <em>CSPNet</em> feature extractor, <em>PANet</em> as neck and combine it with <em>LIME</em> explanation method to help the user understand the mask adherence at an individual and aggregate level. Authorities around the world can use the tool by uploading the image and see results with the explanation. In times of the COVID-19 pandemic, with the world looking back to normalcy and people resuming in-person work, such monitoring of face masks at public places will make them safer.

## Project Setting
Install the required Python packages.
```
pip3 install -r requirements.txt
```
Local Deployment
```
python app.py
```
## Method
![Flow](https://drive.google.com/uc?export=view&id=1Eyrje5nNPh_268wIxUkbwx5SaAuvGLgk)

The proposed workflow is shown in figure above. The detection model takes an arbitrary sized image, outputs an object localization, object confidence and class for each detected faces in the image. Then, a particular face is selected, and LIME runs for that target. LIME returns a segmented mask using Quickshift segmentation algorithm, with green segments contributing positively, red negatively towards prediction. In the tool, instead of making users counting the proportion of mask adherence, we provide composite labels about the degree of mask adherence in an image based on configurable parameters.

## Results
<p float="left">
  <img src="https://drive.google.com/uc?export=view&id=1wUYdsyXbfwGXE3stdKaMdssruIc97Zor" width="450" />
  <img src="https://drive.google.com/uc?export=view&id=1AvrYA35CFw1Nk9MhJGeChvxuHMAfW15B" width="450" /> 
</p>

Above are the two screenshots of the tool. For full demo video refer to <a href="https://drive.google.com/file/d/18f6BSZ3Ck4P9syDWY3hn37PLIxu8oW2K/view">DEMO</a>

## Refrences
* [Prof. Biplav Srivastava's Github Repo](https://github.com/biplav-s)
* [Training Notebooks Repositories](https://github.com/akshay-gupta123/Face-Mask-Detection)