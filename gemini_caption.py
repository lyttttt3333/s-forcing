from google import genai
from google.genai import types
import os
import json
from tqdm import tqdm

client = genai.Client(api_key="AIzaSyD_cVbBFmJB8v-MDquZu1viuAZnpRo_1_0")

def generate_text_from_video(video_file_path):
    video_bytes = open(video_file_path, 'rb').read()

    response = client.models.generate_content(
        model='models/gemini-2.0-flash',
        contents=types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                ),
                types.Part(text='Please summarize the video in 3 sentences.')
            ]
        )
    )
    return response.candidates[0].content.parts[0].text

def process_videos_in_folder(folder_path, output_json_path, output_txt_path):
    video_texts = {}
    text_lines = []

    filenames = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
    for filename in tqdm(filenames, desc="Processing videos"):
        video_path = os.path.join(folder_path, filename)
        print(f"Processing: {video_path}")
        try:
            text = generate_text_from_video(video_path)
            video_texts[filename] = text
            text_lines.append(text)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Save JSON
    with open(output_json_path, 'w') as json_file:
        json.dump(video_texts, json_file, indent=2)

    # Save TXT
    with open(output_txt_path, 'w') as txt_file:
        for line in text_lines:
            txt_file.write(line + '\n')

if __name__ == "__main__":
    folder = "data/sekai-game-drone"
    json_output = "data/sekai-game-drone/video_texts.json"
    txt_output = "data/sekai-game-drone/video_texts.txt"
    process_videos_in_folder(folder, json_output, txt_output)