import os
from scripts.extract_video_data import extract_frames, extract_audio, extract_metadata
from scripts.generate_embeddings import store_embeddings
from scripts.query_video import generate_response

def main():
    video_path = "data/videos/Sample.mp4"

    print("Processing video...")
    extract_frames(video_path, "data/frames/")
    extract_audio(video_path, "data/audio/sample.wav")
    metadata = extract_metadata(video_path)

    print("Generating embeddings...")
    store_embeddings()

    print("\nChatbot is ready! Ask questions about the video.")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = generate_response(query)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()
