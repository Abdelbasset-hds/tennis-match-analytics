from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../')
from utils.bbox_utils import center_of_the_box, calc_distance

class Player_tracker :
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def choose_and_filtre_player(self,court_point,player_list):
        player_detection_first_frame = player_list[0]
        chosen_player = self.filter_players(court_point,player_detection_first_frame)
        filtre_player_detection = []
        for player in player_list :
            dict_player = {id : box for id , box in player.items() if id in chosen_player}
            filtre_player_detection.append(dict_player)
        return filtre_player_detection

    def filter_players(self,court_points,player_dict):
        distance = []
        for id,bbox in player_dict.items() : 
            min_distance = float('inf')
            for i in range(0,len(court_points),2) :
                court_point = (court_points[i],court_points[i+1])
                dist = calc_distance(court_point,center_of_the_box(bbox))
                if dist < min_distance :
                    min_distance = dist
            distance.append((id,min_distance))
        distance.sort(key=lambda x : x[0])
        player_chosen = [distance[0][0],distance[1][0]]
        return player_chosen

    def detect_frame(self,frame) :
        results = self.model.track(frame,conf=0.70,persist=True)[0]
        player_dict = dict()
        for box in results.boxes :
            player_id = int(box.id.tolist()[0])
            class_id = box.cls.tolist()[0]
            result = box.xyxy.tolist()[0]
            if results.names[class_id] == "person" :
                player_dict[player_id] = result
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
            for player_id , bbox in player.items() :
                x1,y1,x2,y2 = map(int,bbox)
                cv2.putText(frame,f'player id : {player_id}',(int(x1),int(y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
            output.append(frame)
        return output