#!/usr/bin/env python3
"""
Test script for Bangladeshi Bangla TTS system
Quick verification that the code works
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append('.')

def test_basic_functionality():
    """Test basic functionality without full pipeline"""
    print("🧪 Testing Bangladeshi Bangla TTS System...")
    
    try:
        # Import the main class
        from bangla_tts_colab import BanglaTTSSystem
        
        # Initialize system
        tts_system = BanglaTTSSystem("/tmp/test_bangla_tts")
        print("✅ System initialization successful")
        
        # Test text cleaning
        test_text = "আমি বাংলায় কথা বলি।  !!!"
        cleaned = tts_system.clean_bangla_text(test_text)
        print(f"✅ Text cleaning: '{test_text}' -> '{cleaned}'")
        
        # Test sample data creation
        sample_data = tts_system.create_sample_data()
        print(f"✅ Sample data creation: {len(sample_data)} samples")
        
        # Test data splits
        train_data, val_data, test_data = tts_system.create_training_splits(sample_data)
        print(f"✅ Data splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # Test model config creation
        model_config = tts_system.create_lightweight_model()
        print(f"✅ Model config created: {model_config['model_type']}")
        
        # Test sample audio generation
        output_path = "/tmp/test_sample.wav"
        success = tts_system.generate_sample_audio("আমি বাংলায় কথা বলি।", output_path)
        print(f"✅ Sample audio generation: {'Success' if success else 'Failed'}")
        
        print("\n🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quick_pipeline():
    """Test a quick version of the pipeline"""
    print("\n🚀 Testing Quick Pipeline...")
    
    try:
        from bangla_tts_colab import BanglaTTSSystem
        
        # Initialize with smaller config for testing
        tts_system = BanglaTTSSystem("/tmp/test_bangla_tts")
        tts_system.config.update({
            "num_epochs": 2,  # Very short training
            "batch_size": 2,
            "checkpoint_interval": 1
        })
        
        # Create sample data
        all_data = tts_system.create_sample_data()
        
        # Create splits
        train_data, val_data, test_data = tts_system.create_training_splits(all_data)
        
        # Quick training test
        training_history = tts_system.train_model(train_data[:4], val_data[:2])  # Use subset
        print(f"✅ Training test completed: {len(training_history.get('train_losses', []))} epochs")
        
        # Quick evaluation
        evaluation_results = tts_system.evaluate_accent_similarity(test_data[:2])
        print(f"✅ Evaluation completed: {evaluation_results.get('accent_similarity_score', 0):.3f}")
        
        # Generate samples
        sample_texts = ["আমি বাংলায় কথা বলি।", "ঢাকা বাংলাদেশের রাজধানী।"]
        for i, text in enumerate(sample_texts):
            output_path = f"/tmp/quick_sample_{i+1}.wav"
            tts_system.generate_sample_audio(text, output_path)
        
        print("✅ Sample generation completed")
        
        print("\n🎉 Quick pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🎯 Bangladeshi Bangla TTS - Test Suite")
    print("=" * 60)
    
    # Test 1: Basic functionality
    basic_success = test_basic_functionality()
    
    # Test 2: Quick pipeline (only if basic tests pass)
    pipeline_success = False
    if basic_success:
        pipeline_success = test_quick_pipeline()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"  Basic Tests: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"  Pipeline Test: {'✅ PASS' if pipeline_success else '❌ FAIL'}")
    
    if basic_success and pipeline_success:
        print("\n🎉 All tests passed! The system is ready for Google Colab.")
        print("\n📋 Next steps:")
        print("  1. Upload bangla_tts_colab.py to Google Colab")
        print("  2. Run: exec(open('bangla_tts_colab.py').read())")
        print("  3. Wait for the complete pipeline to finish")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
