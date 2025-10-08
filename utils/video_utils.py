import cv2


def read_video(video_path) :
    cap = cv2.VideoCapture(video_path)
    frames = list()
    while True :
        ret , frame = cap.read()
        if not ret :
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video,output_path):
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'MJPG'),
        24,
        (output_video[0].shape[1],output_video[0].shape[0])
    )
    for frame in output_video :
        out.write(frame)
    out.release()
    print(f"Video saved to {output_path}")
