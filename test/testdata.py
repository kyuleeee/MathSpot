import re
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from datasets import Dataset, load_dataset
from tqdm import tqdm
from huggingface_hub import HfApi, HfFolder
import os
import pandas as pd
import json
from pathlib import Path

# Hugging Face 토큰 설정
huggingface_token = "__"  
os.environ["HUGGINGFACE_TOKEN"] = huggingface_token
HfFolder.save_token(huggingface_token)


#openai.api_key = "-"
#openai.api_base = "-"
openai.api_type = '-'
openai.api_version = '-'


# OpenAI 클라이언트 설정 (주석 처리된 부분)
# client = AzureOpenAI(
#    
# ) 

class MathAnnotator:
    def __init__(self, output_dir: str = "annotations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_for_annotation(self, dataset_entry: dict) -> dict:
        """Prepare a dataset entry for annotation by splitting into words."""
        words = dataset_entry['text'].strip().split()
        return {
            'video_id': dataset_entry['video_id'],
            'text': dataset_entry['text'],
            'words': words,
            'labels': ['O'] * len(words)  # Initialize with 'O' labels
        }

    def save_for_annotation(self, data: dict, filename: str):
        """Save data in CSV format for manual annotation."""
        # Create CSV with words and labels
        rows = []
        for word, label in zip(data['words'], data['labels']):
            rows.append({
                'video_id': data['video_id'],
                'word': word,
                'label': label  # Will be 'O' initially
            })
        
        df = pd.DataFrame(rows)
        csv_path = self.output_dir / filename
        df.to_csv(csv_path, index=False)
        print(f"Saved for annotation: {csv_path}")
        return csv_path

    def load_annotations(self, csv_path: str) -> dict:
        """Load annotated CSV and convert back to dataset format."""
        df = pd.read_csv(csv_path)
        video_id = df['video_id'].iloc[0]
        labels = df['label'].tolist()
        
        return {
            'video_id': video_id,
            'labels': labels
        }

def process_youtube_video(url):
    """Process a single YouTube video."""
    try:
        yt = YouTube(url)
        video_id = yt.video_id
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry['text'] for entry in transcript])
        
        return {
            'video_id': video_id,
            'text': text
        }
    except Exception as e:
        print(f"Error processing video {url}: {e}")
        return None

def main():
    # Initialize annotator
    annotator = MathAnnotator()
    
    # Process YouTube videos
    youtube_urls = [
        "https://www.youtube.com/watch?v=nHlE7EgJFds",
        "https://www.youtube.com/watch?v=bBVDSUZDCWY"
    ]
    
    youtube_data = [process_youtube_video(url) for url in youtube_urls]
    youtube_data = [item for item in youtube_data if item]
    
    # Process existing dataset
    raw_data = load_dataset("jeongyoun/youtube-transcript-dataset")
    processed_data = []
    
    print("Processing dataset entries...")
    for item in tqdm(raw_data['train']):
        processed_data.append({
            'video_id': item['video_id'],
            'text': item['text']
        })
    
    # Combine data
    all_data = youtube_data + processed_data
    
    # Prepare for annotation
    print("\nPreparing files for annotation...")
    annotation_files = []
    for i, entry in enumerate(all_data):
        prepared_data = annotator.prepare_for_annotation(entry)
        csv_path = annotator.save_for_annotation(
            prepared_data,
            f"transcript_{i}_{entry['video_id']}.csv"
        )
        annotation_files.append(csv_path)
    
    print("\nAnnotation Instructions:")
    print("1. Open the CSV files in the 'annotations' directory")
    print("2. For each word, fill in the 'label' column with:")
    print("   - O: Non-mathematical word")
    print("   - B-MATH: First word of a mathematical expression")
    print("   - I-MATH: Continuation of a mathematical expression")
    print("3. Save the CSV files")
    print("4. Run this script again with --upload flag to upload to Hugging Face")
    
    # If --upload flag is provided, load annotations and upload to Hugging Face
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    
    if args.upload:
        print("\nLoading annotations and uploading to Hugging Face...")
        annotated_data = []
        
        for csv_path in annotation_files:
            if csv_path.exists():
                annotations = annotator.load_annotations(csv_path)
                entry_idx = int(csv_path.stem.split('_')[1])
                annotated_data.append({
                    'video_id': all_data[entry_idx]['video_id'],
                    'text': all_data[entry_idx]['text'],
                    'labels': annotations['labels']
                })
        
        # Create Hugging Face dataset
        dataset = Dataset.from_dict({
            'video_id': [item['video_id'] for item in annotated_data],
            'text': [item['text'] for item in annotated_data],
            'labels': [item['labels'] for item in annotated_data]
        })
        
        # Upload to Hugging Face
        repo_name = "jeongyoun/VideoTranscript_Math_NER"
        dataset.push_to_hub(repo_name)
        
        print(f"\nDataset uploaded to: https://huggingface.co/datasets/{repo_name}")
        print(f"Total annotated samples: {len(annotated_data)}")

if __name__ == "__main__":
    main()
