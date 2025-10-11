import cv2
import numpy as np
from utils.video_utils import combin_frames
from utils.bbox_utils import center_of_the_box, get_foot_position


class CourtOverlay :
    def __init__(self,src_point,point_dst) :
        src_point = src_point.reshape((14,2))
        src_point = np.array([src_point],dtype=np.float32)
        point_dst = np.array([point_dst],dtype=np.float32)
        self.H , _ = cv2.findHomography(src_point,point_dst,cv2.RANSAC)
    
    def map_point(self,point) :
        src_point = np.array([[[point[0],point[1]]]],dtype=np.float32)
        mapped = cv2.perspectiveTransform(src_point,self.H)
        return (mapped[0][0])

    def draw_ball(self,frame,ball_position) :
        x,y = self.map_point(ball_position)
        cv2.circle(frame,(int(x),int(y)+60),2,(0,255,0),-1)
    
    def draw_on_video(self,frames,ball_positions,player_liste):
        tennis_court = cv2.imread('data/tennis_court.jpg')
        tennis_court = cv2.rotate(tennis_court,cv2.ROTATE_90_CLOCKWISE)
        output = list()
        for frame,position,players in zip(frames,ball_positions,player_liste) :        
            court_img = tennis_court.copy()
            values = list(players.values())
            for i in range(min(2,len(values))) :
                bbox = values[i]
                player_position = get_foot_position(bbox)
                self.draw_player(court_img,player_position)
            ball_position = position.get(1)
            ball_position = center_of_the_box(ball_position)
            self.draw_ball(court_img,ball_position)
            result = combin_frames(frame,court_img)
            output.append(result)
        return output

    def draw_player(self,frame,player_position) :
        x,y = self.map_point(player_position)
        cv2.circle(frame,(int(x),int(y)),3,(0,0,255),-1)


