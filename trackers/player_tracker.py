from ultralytics import YOLO
import cv2

class Player_tracker :
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def detect_frame(self,frame) :
        results = self.model.track(frame,persist=True)[0]
        player_dict = dict()
        for box in results.boxes :
            player_id = int(box.id.tolist()[0])
            class_id = box.cls.tolist()[0]
            result = box.xyxy.tolist()[0]
            if results.names[class_id] == "person" :
                player_dict[player_id] = result
        return player_dict


    def detect_frames(self,frames) :
        player_detctions = list()
        for frame in frames :
            player_detcted = self.detect_frame(frame)
            player_detctions.append(player_detcted)
        return player_detctions

    def draw_boxes(self,video_frame,player_detections):
        output = list()
        for frame,player in zip(video_frame,player_detections) :
            for player_id , box in player.items() :
                x1,y1,x2,y2 = map(int,box)
                cv2.putText(frame,f'player id : {player_id}',(int(x1),int(y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),3)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
        output.append(frame)
        return output