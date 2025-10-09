from utils.video_utils import read_video, save_video
from court_lign_detector.court_lign_detector import CourtLignDetector
from trackers import Player_tracker , Ball_tracker
from ultralytics import YOLO
import cv2

def main():
    input_video_path = "data/tennis_match_fixe.mp4"
    video_frame = read_video(input_video_path)

    player_track = Player_tracker("yolov8x.pt")
    ball_track = Ball_tracker("models/last.pt")
    keypoint_detector = CourtLignDetector("models/keypoints_model.pth")

    
    keypoints_detect = keypoint_detector.predict(video_frame[0])
    ball_result = ball_track.detect_frames(video_frame,read_from_stab=True,stab_path="tracker_stubs/ball_detections.pkl")
    ball_result = ball_track.interpolate(ball_result)
    player_result = player_track.detect_frames(video_frame,read_from_stab=True,stab_path="tracker_stubs/player_detections.pkl")
    player_filtred = player_track.choose_and_filtre_player(keypoints_detect,player_result)

    output_video = player_track.draw_boxes(video_frame,player_filtred)
    output_video = ball_track.draw_boxes(output_video,ball_result)
    output_video = keypoint_detector.draw_key_point_on_video(output_video,keypoints_detect)

    for i , frame in enumerate(output_video) :
        cv2.putText(frame,f"frame : {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

    save_video(output_video,"output_video/output_video_test3.avi")

if __name__ == "__main__" :
    main()