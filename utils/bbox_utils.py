from math import sqrt

def center_of_the_box(box) :
    x1 , y1 , x2 , y2 = box
    x_center = int((x1+x2)/2)
    y_center = int((y1+y2)/2)
    return (x_center,y_center)

def calc_distance(p1,p2):
    return sqrt(((p1[0]+p2[0])**2)+((p1[1]+p2[1])**2))

def get_min_distance_player_keycourt(court_points,id,bbox): 
    min_distance = float('inf')
    for i in range(0,len(court_points),2) :
            court_point = (court_points[i],court_points[i+1])
            dist = calc_distance(court_point,center_of_the_box(bbox))
            if dist < min_distance :
                min_distance = dist
    return (id,min_distance)

def get_min_distances_players_keycourt(court_points,player_dict):
    distance = []
    for id,bbox in player_dict.items() : 
        distance.append(get_min_distance_player_keycourt(court_points,id,bbox))
    return distance

def get_closest_keypoint_index(court_points,bbox):
    min_distance = float('inf')
    for i in range(0,len(court_points),2) :
            court_point = (court_points[i],court_points[i+1])
            dist = calc_distance(court_point,center_of_the_box(bbox))
            if dist < min_distance :
                min_distance = dist
                closest_keypoint_index = (i,i+1)
    return closest_keypoint_index


def get_foot_position(bbox) :
    return((bbox[0]+bbox[2])/2,bbox[3])