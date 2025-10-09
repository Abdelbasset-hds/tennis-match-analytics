from ultralytics import YOLO
import cv2
import pickle

class Player_tracker :
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def detect_frame(self,frame) :
        results = self.model.track(frame,conf=0.70,persist=True)[0]
        player_dict = dict()
        for box in results.boxes :
            player_id = int(box.id.tolist()[0])
            class_id = box.cls.tolist()[0]
            result = box.xyxy.tolist()[0]
            score = box.conf.tolist()[0]
            if results.names[class_id] == "person" :
                player_dict[player_id] = (result,score)
        return player_dict


    def detect_frames(self,frames,read_from_stab = False,stab_path=None) :

        if read_from_stab and stab_path is not None :
            with open(stab_path,'rb') as f :
                player_detctions = pickle.load(f)
                return player_detctions
            
        player_detctions = list()
        for frame in frames :
            player_detcted = self.detect_frame(frame)
            player_detctions.append(player_detcted)
        if stab_path is not None :
            with open(stab_path,'wb') as f :
                pickle.dump(player_detctions,f)

        return player_detctions

    def draw_boxes(self,video_frame,player_detections):
        output = list()
        for frame,player in zip(video_frame,player_detections) :
            for player_id , info in player.items() :
                x1,y1,x2,y2 = map(int,info[0])
                cv2.putText(frame,f'player id : {player_id} score {info[1]:.2f}',(int(x1),int(y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
            output.append(frame)
        return output