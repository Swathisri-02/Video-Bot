import os
import json
import numpy as np
import whisper
import ffmpeg
import torch
import torchvision.transforms as transforms
from PIL import Image
from sentence_transformers import SentenceTransformer
from ultralytics import YOLO
import fractions

# ✅ Load models
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
yolo_model = YOLO("yolov8n.pt")  # YOLOv8 Nano for fast inference

def detect_objects(image_path):
    results = yolo_model(image_path)
    detected_objects = set()

    for result in results:
        if hasattr(result, "boxes") and result.boxes is not None:
            for box in result.boxes.data:
                cls = int(box[5]) if len(box) > 5 else -1  # ✅ Prevent index error
                if cls in result.names:
                    detected_objects.add(result.names[cls])  # ✅ Get class name safely
    return list(detected_objects)

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    
    # Convert image to tensor before encoding
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    image_embedding = embedding_model.encode(image_tensor)  # ✅ Encode actual image data
    return image_embedding

def transcribe_audio(audio_path):
    model = whisper.load_model("base")  
    result = model.transcribe(audio_path)
    return result["text"]

def get_audio_embedding(audio_path):
    transcript = transcribe_audio(audio_path)
    audio_embedding = embedding_model.encode([transcript])
    return transcript, audio_embedding

def extract_video_metadata(video_path):
    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")

        metadata = {
            "title": video_path.split("/")[-1],
            "duration": float(probe["format"]["duration"]),
            "resolution": f"{video_info['width']}x{video_info['height']}",
            "fps": float(fractions.Fraction(video_info["r_frame_rate"])),  # ✅ Safe conversion
            "codec": video_info["codec_name"]
        }
        return metadata
    except Exception as e:
        print("Error extracting metadata:", str(e))
        return {}

def store_embeddings():
    frame_embeddings = {}
    frame_objects = {}  # ✅ Store detected objects per frame
    metadata = extract_video_metadata("data/videos/sample.mp4")

    transcript, audio_embedding = get_audio_embedding("data/audio/sample.wav")

    for frame_file in os.listdir("data/frames/"):
        frame_path = os.path.join("data/frames/", frame_file)
        frame_embeddings[frame_file] = get_image_embedding(frame_path).tolist()
        frame_objects[frame_file] = detect_objects(frame_path)  # ✅ Detect objects

    np.save("data/embeddings/frame_embeddings.npy", frame_embeddings)
    np.save("data/embeddings/audio_embedding.npy", audio_embedding)
    np.save("data/embeddings/frame_objects.npy", frame_objects)  # ✅ Use `.npy`

    with open("data/embeddings/video_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    with open("data/embeddings/frame_objects.json", "w") as f:
        json.dump(frame_objects, f, indent=4)

    with open("data/embeddings/video_transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)

if __name__ == "__main__":
    store_embeddings()
    print("Embeddings and object detections stored successfully!")
