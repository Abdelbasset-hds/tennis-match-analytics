import pandas as pd
import numpy as np
import cv2
from utils.bbox_utils import center_of_the_box, calc_distance_meters ,get_foot_position

class Stats:
    def __init__(self,ball_positions,players_positions) :
        self.ball_positions = ball_positions
        self.players_positions = players_positions
        self.df = pd.DataFrame()

    def get_ball_stats(self):
        #print("Ball Positions:", self.ball_positions)
        data = [x.get(1,[]) for x in self.ball_positions ]
        data_adjust = list()
        #print(f"ball data :{data}")
        for i in data :
            x,y = center_of_the_box(i)
            data_adjust.append([x,y])

        new_data = pd.DataFrame(data_adjust,columns=['x','y'])
        
        new_data['player1_hit'] = 0
        new_data['player2_hit'] = 0
        new_data['hits'] = 0
        new_data['delta_y'] = new_data['y'].diff()
        for i in range (len(new_data)-1) :
            if new_data.loc[i,'delta_y'] > 0 and new_data.loc[i+1,'delta_y'] < 0:
                new_data.loc[i+1,'player1_hit'] = 1
            if new_data.loc[i,'delta_y'] < 0 and new_data.loc[i+1,'delta_y'] > 0:
                new_data.loc[i+1,'player2_hit'] = 1
        for i in range (len(new_data)) :
            if new_data.loc[i,'player2_hit'] == 1 or new_data.loc[i,'player1_hit'] == 1 :
                new_data.loc[i,'hits'] =1
        self.df = pd.concat([self.df,new_data],ignore_index = True)
        return self.df

    def get_players_stats(self) :
        data = []
        for player in self.players_positions :
            for id, bbox in player.items() :
                data.append(bbox)
        temp = []
        for i in range(0,len(data),2) :
            bbox1 = data[i]
            bbox2=data[i+1] if i+1 < len(data) else None
            pos1 = get_foot_position(bbox1)
            pos2 = get_foot_position(bbox2) if bbox2 is not None and len(bbox2) == 4 else (np.nan,np.nan)
            temp.append([pos1,pos2])
        new_data = pd.DataFrame(temp,columns=['player1','player2'])
        new_data['player1_dist'] = 0
        new_data['player2_dist'] = 0
        for i in range(0,len(new_data)-1,2) :
            if new_data.loc[i+1,'player1'] is not None :
                point1 = new_data.loc[i,'player1']
                point2 = new_data.loc[i+1,'player1'] 
                distance = calc_distance_meters(point1,point2)
                new_data.loc[i,'player1_dist'] = distance
            if new_data.loc[i+1,'player2'] is not None :
                point1 = new_data.loc[i,'player2']
                point2 =new_data.loc[i+1,'player2']
                distance = calc_distance_meters(point1,point2)
                new_data.loc[i,'player2_dist'] = distance    
        max_length = max(len(self.df), len(new_data))
        self.df = pd.concat([self.df.reindex(range(max_length)), new_data.reindex(range(max_length))], axis=1, ignore_index=False)   
        return self.df

    def draw_interface(self,frames) :
        output = list()
        x1 = frames[1].shape[1]*2/3
        y1 = frames[1].shape[0]/4
        x2 = x1+280
        y2 = frames[1].shape[0]*3/4
        for frame in frames :
            overlay = frame.copy()
            cv2.rectangle(overlay,(int(x1),int(y1)),(int(x2),int(y2)),(50,50,50),-1)
            frame = cv2.addWeighted(frame,0.3,overlay,0.7,0)
            output.append(frame)
        return output

    def draw_hits(self,frame,i):
        x1 = frame.shape[1]*2/3 + 50
        y1 = frame.shape[0]/4 + 50
        cv2.putText(frame,f'number of hits : {i}',(int(x1),int(y1)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,210,255),2,cv2.LINE_AA)
      

    def draw_stats(self,frames):
        output = list()
        frames = self.draw_interface(frames)
        stats = self.get_ball_stats()
        stats = self.get_players_stats()
        stats.to_excel("data/stats_tennis.xlsx",index=False)
        count = 0
        for i,frame in enumerate(frames) :
            
            count += stats.loc[i,'hits']
            self.draw_hits(frame,count)
            output.append(frame)
        return output

    