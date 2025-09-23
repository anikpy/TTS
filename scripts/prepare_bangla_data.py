#!/usr/bin/env python3
"""
Bangladeshi Bangla Data Preparation for CPU TTS Training
Minimal dependency approach for bangla_tts processing
"""

import os
import sys
import json
import wave
import struct
import urllib.request
import tarfile
import zipfile
from pathlib import Path
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BanglaDataPreparator:
    def __init__(self, data_dir="bangla_tts"):
        """Initialize Bangla bangla_tts preparator"""
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # OpenSLR dataset URLs
        self.dataset_urls = {
            "openslr_53": "https://www.openslr.org/resources/53/data_utt.tgz",
            "openslr_54": "https://www.openslr.org/resources/54/data_utt.tgz"
        }
        
        logger.info("Initialized Bangla Data Preparator")
    
    def download_dataset(self, dataset_name, force_download=False):
        """Download OpenSLR Bangla dataset"""
        if dataset_name not in self.dataset_urls:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        url = self.dataset_urls[dataset_name]
        filename = f"{dataset_name}.tgz"
        filepath = self.raw_dir / filename
        
        if filepath.exists() and not force_download:
            logger.info(f"Dataset {dataset_name} already downloaded")
            return True
        
        logger.info(f"Downloading {dataset_name} from {url}")
        
        try:
            # Download with progress
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    if block_num % 100 == 0:  # Log every 100 blocks
                        logger.info(f"Download progress: {percent}%")
            
            urllib.request.urlretrieve(url, filepath, progress_hook)
            logger.info(f"Downloaded {dataset_name} successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
            return False
    
    def extract_dataset(self, dataset_name):
        """Extract downloaded dataset"""
        filename = f"{dataset_name}.tgz"
        filepath = self.raw_dir / filename
        extract_dir = self.raw_dir / dataset_name
        
        if not filepath.exists():
            logger.error(f"Dataset file not found: {filepath}")
            return False
        
        if extract_dir.exists():
            logger.info(f"Dataset {dataset_name} already extracted")
            return True
        
        logger.info(f"Extracting {dataset_name}")
        
        try:
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(self.raw_dir)
            
            # Rename extracted directory
            extracted_dirs = [d for d in self.raw_dir.iterdir() 
                            if d.is_dir() and d.name != dataset_name]
            if extracted_dirs:
                extracted_dirs[0].rename(extract_dir)
            
            logger.info(f"Extracted {dataset_name} successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract {dataset_name}: {e}")
            return False
    
    def get_audio_info(self, audio_path):
        """Get basic audio information without external libraries"""
        try:
            with wave.open(str(audio_path), 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / sample_rate
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                
                return {
                    'duration': duration,
                    'sample_rate': sample_rate,
                    'channels': channels,
                    'sample_width': sample_width,
                    'frames': frames,
                    'is_valid': duration > 0.5 and duration < 15.0  # Valid range
                }
        except Exception as e:
            logger.warning(f"Could not read audio file {audio_path}: {e}")
            return {
                'duration': 0,
                'sample_rate': 0,
                'channels': 0,
                'sample_width': 0,
                'frames': 0,
                'is_valid': False
            }
    
    def clean_bangla_text(self, text):
        """Clean and normalize Bangla text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove non-Bangla characters (keep basic punctuation)
        # Bangla Unicode range: \u0980-\u09FF
        text = re.sub(r'[^\u0980-\u09FF\s\.\,\!\?\;\:]', '', text)
        
        # Remove multiple punctuation
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[,]{2,}', ',', text)
        
        return text.strip()
    
    def process_dataset(self, dataset_name):
        """Process a single dataset"""
        dataset_dir = self.raw_dir / dataset_name
        
        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return []
        
        logger.info(f"Processing dataset: {dataset_name}")
        
        processed_data = []
        
        # Look for audio files and transcripts
        audio_files = list(dataset_dir.rglob("*.wav"))
        logger.info(f"Found {len(audio_files)} audio files")
        
        for audio_file in audio_files:
            # Look for corresponding transcript
            transcript_file = audio_file.with_suffix('.txt')
            
            if not transcript_file.exists():
                # Try alternative transcript naming
                transcript_file = audio_file.parent / f"{audio_file.stem}.trans.txt"
            
            transcript = ""
            if transcript_file.exists():
                try:
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        transcript = f.read().strip()
                except:
                    try:
                        with open(transcript_file, 'r', encoding='latin-1') as f:
                            transcript = f.read().strip()
                    except:
                        logger.warning(f"Could not read transcript: {transcript_file}")
            
            # Clean transcript
            cleaned_transcript = self.clean_bangla_text(transcript)
            
            # Get audio info
            audio_info = self.get_audio_info(audio_file)
            
            # Create bangla_tts entry
            data_entry = {
                'file_path': str(audio_file),
                'transcript': cleaned_transcript,
                'cleaned_transcript': cleaned_transcript,
                'duration': audio_info['duration'],
                'sample_rate': audio_info['sample_rate'],
                'channels': audio_info['channels'],
                'is_valid': (audio_info['is_valid'] and 
                           len(cleaned_transcript) > 5 and
                           len(cleaned_transcript) < 200),
                'has_transcript': len(cleaned_transcript) > 0,
                'dataset': dataset_name,
                'speaker_id': self.extract_speaker_id(audio_file)
            }
            
            processed_data.append(data_entry)
        
        # Save processed bangla_tts
        output_file = self.processed_dir / f"{dataset_name}_processed.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        # Log statistics
        valid_count = sum(1 for item in processed_data if item['is_valid'])
        total_duration = sum(item['duration'] for item in processed_data if item['is_valid'])
        
        logger.info(f"Dataset {dataset_name} processed:")
        logger.info(f"  Total files: {len(processed_data)}")
        logger.info(f"  Valid files: {valid_count}")
        logger.info(f"  Total duration: {total_duration:.2f} seconds")
        logger.info(f"  Saved to: {output_file}")
        
        return processed_data
    
    def extract_speaker_id(self, audio_path):
        """Extract speaker ID from file path"""
        # Try to extract speaker ID from filename or path
        path_parts = Path(audio_path).parts
        
        # Look for speaker patterns
        for part in path_parts:
            if 'speaker' in part.lower():
                return part
            if re.match(r'sp\d+', part.lower()):
                return part
            if re.match(r's\d+', part.lower()):
                return part
        
        # Default to filename without extension
        return Path(audio_path).stem
    
    def create_training_splits(self, processed_data, train_ratio=0.8, val_ratio=0.1):
        """Create training, validation, and test splits"""
        valid_data = [item for item in processed_data if item['is_valid']]
        
        if len(valid_data) < 10:
            logger.warning("Not enough valid bangla_tts for proper splits")
            return valid_data, [], []
        
        # Shuffle bangla_tts
        import random
        random.shuffle(valid_data)
        
        # Calculate split indices
        total = len(valid_data)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_data = valid_data[:train_end]
        val_data = valid_data[train_end:val_end]
        test_data = valid_data[val_end:]
        
        logger.info(f"Data splits created:")
        logger.info(f"  Training: {len(train_data)} samples")
        logger.info(f"  Validation: {len(val_data)} samples")
        logger.info(f"  Test: {len(test_data)} samples")
        
        return train_data, val_data, test_data
    
    def prepare_all_data(self):
        """Prepare all available datasets"""
        all_processed_data = []
        
        for dataset_name in self.dataset_urls.keys():
            logger.info(f"Processing {dataset_name}...")
            
            # Download and extract
            if self.download_dataset(dataset_name):
                if self.extract_dataset(dataset_name):
                    # Process dataset
                    processed_data = self.process_dataset(dataset_name)
                    all_processed_data.extend(processed_data)
                else:
                    logger.error(f"Failed to extract {dataset_name}")
            else:
                logger.error(f"Failed to download {dataset_name}")
        
        if not all_processed_data:
            logger.error("No bangla_tts was processed successfully")
            return
        
        # Create splits
        train_data, val_data, test_data = self.create_training_splits(all_processed_data)
        
        # Save splits
        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        for split_name, split_data in splits.items():
            if split_data:
                output_file = self.processed_dir / f"{split_name}_data.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(split_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {split_name} split: {len(split_data)} samples")
        
        # Save summary
        summary = {
            'total_datasets': len(self.dataset_urls),
            'total_files': len(all_processed_data),
            'valid_files': len([item for item in all_processed_data if item['is_valid']]),
            'total_duration': sum(item['duration'] for item in all_processed_data if item['is_valid']),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data)
        }
        
        summary_file = self.processed_dir / "data_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info("Data preparation completed!")
        logger.info(f"Summary: {summary}")

def main():
    """Main bangla_tts preparation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Bangladeshi Bangla TTS Data")
    parser.add_argument("--bangla_tts-dir", default="bangla_tts", help="Data directory")
    parser.add_argument("--download-only", action="store_true", 
                       help="Only download datasets, don't process")
    
    args = parser.parse_args()
    
    preparator = BanglaDataPreparator(args.data_dir)
    
    if args.download_only:
        for dataset_name in preparator.dataset_urls.keys():
            preparator.download_dataset(dataset_name)
            preparator.extract_dataset(dataset_name)
    else:
        preparator.prepare_all_data()

if __name__ == "__main__":
    main()
