from utils.video_utils import read_video, save_video
from court_lign_detector.court_lign_detector import CourtLignDetector
from trackers import Player_tracker , Ball_tracker
from ultralytics import YOLO

def main():
    input_video_path = "data/tennis_match_fixe.mp4"
    video_frame = read_video(input_video_path)
    player_track = Player_tracker("yolov8x.pt")
    ball_track = Ball_tracker("models/last.pt")
    keypoint_detector = CourtLignDetector("models/keypoints_model.pth")

    
    keypoints_detect = keypoint_detector.predict(video_frame[0])
    ball_result = ball_track.detect_frames(video_frame,read_from_stab=True,stab_path="tracker_stubs/ball_detections.pkl")
    player_result = player_track.detect_frames(video_frame,read_from_stab=True,stab_path="tracker_stubs/player_detections.pkl")
    output_video = player_track.draw_boxes(video_frame,player_result)
    output_video = ball_track.draw_boxes(output_video,ball_result)
    output_video = keypoint_detector.draw_key_point_on_video(output_video,keypoints_detect)

    save_video(output_video,"output_video/output_video_test3.avi")

if __name__ == "__main__" :
    main()