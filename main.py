from utils.video_utils import read_video, save_video
from trackers.player_tracker import Player_tracker 
from ultralytics import YOLO

def main():
    input_video_path = "data/tennis_match_lite.mp4"
    video_frame = read_video(input_video_path)
    player_detect = Player_tracker("yolov8x.pt")
    result = player_detect.detect_frames(video_frame)
    output_video = player_detect.draw_boxes(video_frame,result)
    print(result)

    save_video(output_video,"output_video/output_video_test1.avi")

if __name__ == "__main__" :
    main()