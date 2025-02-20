import numpy as np
import json
import groq
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# ✅ Load .env variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ✅ Configure Groq API
groq_api_key = os.getenv("GROQ_API_KEY")
client = groq.Groq(api_key=groq_api_key)

# ✅ Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ Load stored embeddings, metadata, and object detections
frame_embeddings = np.load("data/embeddings/frame_embeddings.npy", allow_pickle=True).item()
audio_embedding = np.load("data/embeddings/audio_embedding.npy", allow_pickle=True)

with open("data/embeddings/video_metadata.json", "r") as f:
    video_metadata = json.load(f)

with open("data/embeddings/video_transcript.txt", "r") as f:
    video_transcript = f.read()

with open("data/embeddings/frame_objects.json", "r") as f:
    frame_objects = json.load(f)  # ✅ Load detected objects

def find_most_relevant_embedding(query_embedding):
    query_embedding = np.array(query_embedding).reshape(-1)  # ✅ Flatten to (384,)
    best_match = None
    highest_similarity = -1

    for frame_file, embedding in frame_embeddings.items():
        embedding = np.array(embedding).reshape(-1)  # ✅ Flatten to (384,)
        similarity = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = frame_file

    return best_match, highest_similarity

def generate_response(query):
    query_embedding = embedding_model.encode([query])
    best_frame, similarity = find_most_relevant_embedding(query_embedding)

    # # ✅ Find objects in the most relevant frame
    # detected_objects = frame_objects.get(best_frame, [])

    # ✅ Collect objects from all frames
    all_detected_objects = set()
    for obj_list in frame_objects.values():
        all_detected_objects.update(obj_list)

    detected_objects = list(all_detected_objects)  # Convert to list

    # ✅ Create a prompt for LLM
    prompt = f"""
    You are an AI that provides information about a video. 
    The video is titled "{video_metadata['title']}".
    The transcript of the video includes: "{video_transcript[:500]}"... (truncated).
    Across all frames, the detected objects are: {', '.join(detected_objects)}.
    
    Question: "{query}"
    Answer:
    """

    # ✅ Generate a response using Groq's API
    response = client.chat.completions.create(
        model="llama3-8b-8192",  # ✅ Replace with Groq model name
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )

    return response.choices[0].message.content.strip()
