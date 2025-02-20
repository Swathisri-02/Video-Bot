import cv2
import os
import json
import ffmpeg

def extract_frames(video_path, output_folder, frame_interval=50):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_id}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_id += 1
        frame_count += 1 

    cap.release()

def extract_audio(video_path, output_audio_path):
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    ffmpeg.input(video_path).output(output_audio_path).run(overwrite_output=True)

def extract_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    metadata = {
        "frame_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "frame_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / int(cap.get(cv2.CAP_PROP_FPS))),
    }
    cap.release()
    return metadata

if __name__ == "__main__":
    video_path = "data/videos/sample.mp4"
    extract_frames(video_path, "data/frames/")
    extract_audio(video_path, "data/audio/sample.wav")
    metadata = extract_metadata(video_path)

    with open("data/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("Extraction complete!")
