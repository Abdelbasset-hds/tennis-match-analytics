import cv2
import numpy as np

def read_video(video_path) :
    cap = cv2.VideoCapture(video_path)
    frames = list()
    while True :
        ret , frame = cap.read()
        if not ret :
            break
        frames.append(frame)
    cap.release()
    print(f'frame input {frames[0].shape}')
    return frames

def save_video(output_video,output_path):
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'MJPG'),
        24,
        (output_video[0].shape[1],output_video[0].shape[0])
    )
    print(f'frame input {output_video[0].shape}')
    for frame in output_video :
        out.write(frame)
    out.release()

    print(f"Video saved to {output_path}")

def combin_frames(frame1,frame2) :
    img_h , img_w = frame2.shape[:2]
    frame_h,frame_w = frame1.shape[:2]
    img_resize = cv2.resize(frame2,(int((img_w*frame_h)/img_h),frame_h))
    combined = np.hstack([frame1,img_resize])
    return combined