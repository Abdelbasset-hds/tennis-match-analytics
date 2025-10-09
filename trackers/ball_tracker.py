from ultralytics import YOLO
import cv2
import pickle

class Ball_tracker :
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def detect_frame(self,frame) :
        results = self.model.predict(frame,conf=0.15)[0]
        ball_dict = dict()
        for box in results.boxes :
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        return ball_dict


    def detect_frames(self,frames,read_from_stab = False,stab_path=None) :

        if read_from_stab and stab_path is not None :
            with open(stab_path,'rb') as f :
                ball_detctions = pickle.load(f)
                return ball_detctions
            
        ball_detctions = list()
        for frame in frames :
            ball_detcted = self.detect_frame(frame)
            ball_detctions.append(ball_detcted)
        if stab_path is not None :
            with open(stab_path,'wb') as f :
                pickle.dump(ball_detctions,f)

        return ball_detctions

    def draw_boxes(self,video_frame,ball_detections):
        output = list()
        for frame,ball in zip(video_frame,ball_detections) :
            for ball_id , box in ball.items() :
                x1,y1,x2,y2 = map(int,box)
                cv2.putText(frame,f'ball id : {ball_id}',(int(x1),int(y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),3)
            output.append(frame)
        return output