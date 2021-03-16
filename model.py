import torch

model_det = torch.hub.load('ultralytics/yolov5','custom', path_or_model='./static/weights.pt').autoshape()

def get_model(img):
    '''
        Args:
            img: 3d numpy array or a batch of 3d numpy array 
        Returns:
            A detection object tensor 
    '''
    model_det.conf = 0.35
    return model_det(img)
 
