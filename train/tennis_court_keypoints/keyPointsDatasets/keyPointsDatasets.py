from torchvision import  transforms
import json
import cv2
import numpy as np

class KeyPointDatasets :
    def __init__(self,img_dir,data_file):
        self.img_dir = img_dir
        with open(data_file, "r") as f :
            self.data = json.load(f)

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) :
        return len(self.data)
    
    def __getitem__(self,idx) :
        item = self.data[idx]
        img = cv2.imread(f"{self.img_dir}/{item['id']}.png")
        h,w = img.shape[:2]

        img = img.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(img)
        kps = np.array(item['kps']).flatten()
        kps = kps.astype(np.float32)

        kps[::2] *= 224/w
        kps[1::2] *= 224/h

        return img,kps

