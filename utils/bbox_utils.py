from math import sqrt

def center_of_the_box(box) :
    x1 , y1 , x2 , y2 = box
    x_center = int((x1+x2)/2)
    y_center = int((y1+y2)/2)
    return (x_center,y_center)

def calc_distance(p1,p2):
    return sqrt(((p1[0]+p2[0])**2)+((p1[1]+p2[1])**2))
