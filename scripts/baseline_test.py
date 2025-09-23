#!/usr/bin/env python3
"""
Baseline TTS model testing for Bangladeshi Bangla
Tests different models and generates sample outputs for comparison
"""

import os
import sys
import time
import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path
from TTS.api import TTS
import matplotlib.pyplot as plt
import seaborn as sns

# Set CPU-only mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(4)  # Optimize for CPU

class BaselineTester:
    def __init__(self, output_dir="outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Test sentences in Bangladeshi Bangla
        self.test_sentences = [
            "আমি বাংলাদেশে থাকি।",  # I live in Bangladesh
            "ঢাকা আমাদের রাজধানী।",  # Dhaka is our capital
            "বাংলা আমাদের মাতৃভাষা।",  # Bangla is our mother tongue
            "আজ আবহাওয়া খুব সুন্দর।",  # Today's weather is very nice
            "তুমি কেমন আছো?"  # How are you?
        ]
        
        self.models_to_test = [
            "tts_models/multilingual/multi-dataset/xtts_v2",
            "tts_models/multilingual/multi-dataset/xtts_v1", 
            "tts_models/bn/custom/vits"  # If available
        ]
    
    def test_model_availability(self):
        """Test which models are available"""
        print("Testing model availability...")
        available_models = []
        
        for model_name in self.models_to_test:
            try:
                print(f"Testing: {model_name}")
                tts = TTS(model_name=model_name, progress_bar=False)
                available_models.append((model_name, tts))
                print(f"✅ {model_name} - Available")
            except Exception as e:
                print(f"❌ {model_name} - Not available: {str(e)}")
        
        return available_models
    
    def generate_baseline_samples(self, model_name, tts_model):
        """Generate baseline samples for a given model"""
        print(f"\nGenerating samples for {model_name}...")
        
        model_output_dir = self.output_dir / "baseline" / model_name.replace("/", "_")
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, sentence in enumerate(self.test_sentences):
            try:
                print(f"Generating: {sentence}")
                
                # Measure generation time
                start_time = time.time()
                
                # Generate audio
                if "xtts" in model_name.lower():
                    # XTTS models need speaker reference
                    wav = tts_model.tts(text=sentence, language="bn")
                else:
                    wav = tts_model.tts(text=sentence)
                
                generation_time = time.time() - start_time
                
                # Save audio
                output_path = model_output_dir / f"sample_{i+1}.wav"
                tts_model.tts_to_file(text=sentence, file_path=str(output_path))
                
                # Analyze audio properties
                audio_info = self.analyze_audio(output_path)
                
                results.append({
                    'sentence': sentence,
                    'file_path': output_path,
                    'generation_time': generation_time,
                    **audio_info
                })
                
                print(f"✅ Generated: {output_path} ({generation_time:.2f}s)")
                
            except Exception as e:
                print(f"❌ Failed to generate for '{sentence}': {str(e)}")
                results.append({
                    'sentence': sentence,
                    'error': str(e)
                })
        
        return results
    
    def analyze_audio(self, audio_path):
        """Analyze audio properties"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Basic properties
            duration = len(y) / sr
            
            # Spectral features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            
            # Fundamental frequency
            f0 = librosa.yin(y, fmin=50, fmax=400)
            f0_mean = np.mean(f0[f0 > 0]) if np.any(f0 > 0) else 0
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'mfcc_mean': np.mean(mfccs),
                'mfcc_std': np.std(mfccs),
                'spectral_centroid_mean': np.mean(spectral_centroid),
                'f0_mean': f0_mean
            }
            
        except Exception as e:
            print(f"Warning: Could not analyze {audio_path}: {e}")
            return {}
    
    def create_comparison_report(self, all_results):
        """Create a comparison report of all models"""
        print("\nCreating comparison report...")
        
        report_path = self.output_dir / "baseline_comparison_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("BASELINE TTS MODEL COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, results in all_results.items():
                f.write(f"Model: {model_name}\n")
                f.write("-" * 30 + "\n")
                
                successful_generations = [r for r in results if 'error' not in r]
                failed_generations = [r for r in results if 'error' in r]
                
                f.write(f"Successful generations: {len(successful_generations)}/{len(results)}\n")
                
                if successful_generations:
                    avg_time = np.mean([r['generation_time'] for r in successful_generations])
                    avg_duration = np.mean([r['duration'] for r in successful_generations if 'duration' in r])
                    
                    f.write(f"Average generation time: {avg_time:.2f}s\n")
                    f.write(f"Average audio duration: {avg_duration:.2f}s\n")
                
                if failed_generations:
                    f.write("Failed generations:\n")
                    for fail in failed_generations:
                        f.write(f"  - '{fail['sentence']}': {fail['error']}\n")
                
                f.write("\n")
        
        print(f"Report saved to: {report_path}")
    
    def run_baseline_tests(self):
        """Run complete baseline testing"""
        print("🚀 Starting baseline TTS model testing...")
        print("=" * 50)
        
        # Test model availability
        available_models = self.test_model_availability()
        
        if not available_models:
            print("❌ No models available for testing!")
            return
        
        # Generate samples for each available model
        all_results = {}
        
        for model_name, tts_model in available_models:
            try:
                results = self.generate_baseline_samples(model_name, tts_model)
                all_results[model_name] = results
            except Exception as e:
                print(f"❌ Failed to test {model_name}: {e}")
        
        # Create comparison report
        if all_results:
            self.create_comparison_report(all_results)
            print("\n✅ Baseline testing completed!")
            print(f"Check outputs in: {self.output_dir}")
        else:
            print("❌ No successful model tests!")

def main():
    """Main function"""
    print("CPU-only Bangladeshi Bangla TTS Baseline Testing")
    print("=" * 50)
    
    # Verify CPU setup
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print("⚠️  GPU detected but we're using CPU-only mode")
    
    # Run baseline tests
    tester = BaselineTester()
    tester.run_baseline_tests()

if __name__ == "__main__":
    main()
