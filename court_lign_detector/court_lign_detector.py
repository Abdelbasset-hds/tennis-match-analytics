import torch
import cv2
from torchvision import models, transforms
import pickle

class CourtLignDetector():
    def __init__(self,model_path):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features,14*2)
        self.model.load_state_dict(torch.load(model_path,map_location='cpu'))
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])

    def predict(self,img) :
            img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img_tensor = self.transforms(img_rgb).unsqueeze(0)
            with torch.no_grad() :
                output = self.model(img_tensor)
            keypoint = output.squeeze().cpu().numpy()
            original_h,original_w = img.shape[:2]
            keypoint[::2] *= original_w/224
            keypoint[1::2] *= original_h/224
            return keypoint
    
    def draw_key_point(self,img,keypoint):
        for i in range(0,len(keypoint),2) :
            x = int(keypoint[i])
            y = int(keypoint[i+1])

            cv2.circle(img,(x,y),2,(255,0,0),-1)
            cv2.putText(img,f"{i//2}",(x,y-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),3)

        return img
    
    def draw_key_point_on_video(self,frames,keypoint) :
        output = list()
        for frame in frames:
            keypoint_on_frame = self.draw_key_point(frame,keypoint)
            output.append(keypoint_on_frame)

        return output