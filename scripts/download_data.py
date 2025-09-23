#!/usr/bin/env python3
"""
Data acquisition and curation for Bangladeshi Bangla TTS
Downloads and processes OpenSLR and Common Voice datasets
"""

import os
import sys
import requests
import tarfile
import zipfile
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import json
from urllib.parse import urlparse

class BanglaDataDownloader:
    def __init__(self, data_dir="bangla_tts"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.datasets_dir = self.data_dir / "datasets"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.datasets_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs
        self.dataset_urls = {
            "openslr_53": {
                "url": "https://www.openslr.org/resources/53/bn_bd_male.tar.gz",
                "description": "OpenSLR 53 - Bangladeshi Male Speaker",
                "type": "openslr"
            },
            "openslr_54": {
                "url": "https://www.openslr.org/resources/54/bn_bd_female.tar.gz", 
                "description": "OpenSLR 54 - Bangladeshi Female Speaker",
                "type": "openslr"
            }
        }
    
    def download_file(self, url, destination):
        """Download file with progress bar"""
        print(f"Downloading: {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=destination.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ Downloaded: {destination}")
    
    def extract_archive(self, archive_path, extract_to):
        """Extract tar.gz or zip archive"""
        print(f"Extracting: {archive_path}")
        
        if archive_path.suffix == '.gz' and archive_path.stem.endswith('.tar'):
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_to)
        elif archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")
        
        print(f"✅ Extracted to: {extract_to}")
    
    def download_openslr_datasets(self):
        """Download OpenSLR Bangladeshi datasets"""
        print("Downloading OpenSLR datasets...")
        
        for dataset_name, dataset_info in self.dataset_urls.items():
            try:
                # Download
                url = dataset_info["url"]
                filename = Path(urlparse(url).path).name
                download_path = self.raw_dir / filename
                
                if not download_path.exists():
                    self.download_file(url, download_path)
                else:
                    print(f"Already exists: {download_path}")
                
                # Extract
                extract_dir = self.raw_dir / dataset_name
                if not extract_dir.exists():
                    self.extract_archive(download_path, extract_dir)
                else:
                    print(f"Already extracted: {extract_dir}")
                
            except Exception as e:
                print(f"❌ Failed to download {dataset_name}: {e}")
    
    def analyze_audio_file(self, audio_path):
        """Analyze individual audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Basic properties
            duration = len(y) / sr
            
            # Quality checks
            is_too_short = duration < 0.5
            is_too_long = duration > 10.0
            is_silent = np.max(np.abs(y)) < 0.01
            
            return {
                'file_path': str(audio_path),
                'duration': duration,
                'sample_rate': sr,
                'num_samples': len(y),
                'max_amplitude': np.max(np.abs(y)),
                'rms_energy': np.sqrt(np.mean(y**2)),
                'is_too_short': is_too_short,
                'is_too_long': is_too_long,
                'is_silent': is_silent,
                'is_valid': not (is_too_short or is_too_long or is_silent)
            }
            
        except Exception as e:
            return {
                'file_path': str(audio_path),
                'error': str(e),
                'is_valid': False
            }
    
    def process_openslr_dataset(self, dataset_name):
        """Process OpenSLR dataset"""
        print(f"Processing {dataset_name}...")
        
        dataset_dir = self.raw_dir / dataset_name
        if not dataset_dir.exists():
            print(f"Dataset directory not found: {dataset_dir}")
            return None
        
        # Find audio files and transcripts
        audio_files = list(dataset_dir.rglob("*.wav"))
        transcript_files = list(dataset_dir.rglob("*.txt"))
        
        print(f"Found {len(audio_files)} audio files")
        print(f"Found {len(transcript_files)} transcript files")
        
        # Process files
        processed_data = []
        
        for audio_file in tqdm(audio_files, desc=f"Processing {dataset_name}"):
            # Analyze audio
            audio_info = self.analyze_audio_file(audio_file)
            
            # Find corresponding transcript
            transcript_file = audio_file.with_suffix('.txt')
            transcript = ""
            
            if transcript_file.exists():
                try:
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        transcript = f.read().strip()
                except Exception as e:
                    print(f"Error reading transcript {transcript_file}: {e}")
            
            processed_data.append({
                **audio_info,
                'transcript': transcript,
                'dataset': dataset_name,
                'speaker_id': self.extract_speaker_id(audio_file),
                'has_transcript': bool(transcript)
            })
        
        # Save processed bangla_tts
        output_file = self.processed_dir / f"{dataset_name}_processed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Processed bangla_tts saved to: {output_file}")
        return processed_data
    
    def extract_speaker_id(self, audio_path):
        """Extract speaker ID from file path"""
        # This is dataset-specific logic
        parts = audio_path.parts
        for part in parts:
            if 'speaker' in part.lower() or part.startswith('spk'):
                return part
        return "unknown"
    
    def create_dataset_summary(self, all_processed_data):
        """Create summary statistics for all datasets"""
        print("Creating dataset summary...")
        
        summary = {
            'total_files': 0,
            'total_duration': 0,
            'valid_files': 0,
            'datasets': {},
            'duration_distribution': {},
            'sample_rate_distribution': {},
            'speaker_distribution': {}
        }
        
        for dataset_name, data in all_processed_data.items():
            if not data:
                continue
                
            valid_data = [item for item in data if item.get('is_valid', False)]
            
            dataset_summary = {
                'total_files': len(data),
                'valid_files': len(valid_data),
                'total_duration': sum(item.get('duration', 0) for item in valid_data),
                'avg_duration': np.mean([item.get('duration', 0) for item in valid_data]) if valid_data else 0,
                'speakers': len(set(item.get('speaker_id', 'unknown') for item in data))
            }
            
            summary['datasets'][dataset_name] = dataset_summary
            summary['total_files'] += len(data)
            summary['valid_files'] += len(valid_data)
            summary['total_duration'] += dataset_summary['total_duration']
        
        # Save summary
        summary_file = self.processed_dir / "dataset_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Dataset summary saved to: {summary_file}")
        
        # Print summary
        print("\nDATASET SUMMARY")
        print("=" * 30)
        print(f"Total files: {summary['total_files']}")
        print(f"Valid files: {summary['valid_files']}")
        print(f"Total duration: {summary['total_duration']:.2f} seconds ({summary['total_duration']/3600:.2f} hours)")
        
        for dataset_name, dataset_info in summary['datasets'].items():
            print(f"\n{dataset_name}:")
            print(f"  Files: {dataset_info['valid_files']}/{dataset_info['total_files']}")
            print(f"  Duration: {dataset_info['total_duration']:.2f}s")
            print(f"  Avg duration: {dataset_info['avg_duration']:.2f}s")
            print(f"  Speakers: {dataset_info['speakers']}")
    
    def run_data_acquisition(self):
        """Run complete bangla_tts acquisition pipeline"""
        print("🚀 Starting Bangladeshi Bangla bangla_tts acquisition...")
        print("=" * 50)
        
        # Download datasets
        self.download_openslr_datasets()
        
        # Process datasets
        all_processed_data = {}
        
        for dataset_name in self.dataset_urls.keys():
            processed_data = self.process_openslr_dataset(dataset_name)
            if processed_data:
                all_processed_data[dataset_name] = processed_data
        
        # Create summary
        if all_processed_data:
            self.create_dataset_summary(all_processed_data)
            print("\n✅ Data acquisition completed!")
        else:
            print("❌ No datasets were successfully processed!")

def main():
    """Main function"""
    print("Bangladeshi Bangla TTS Data Acquisition")
    print("=" * 40)
    
    downloader = BanglaDataDownloader()
    downloader.run_data_acquisition()

if __name__ == "__main__":
    main()
