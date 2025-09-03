#!/usr/bin/env python3

from datasets import load_dataset
import requests

def test_hf_connection():
    print("🔍 Testing HuggingFace connection...")
    
    # Test basic connection to huggingface.co
    try:
        response = requests.get("https://huggingface.co", timeout=10)
        print(f"✅ Basic connection to huggingface.co: {response.status_code}")
    except Exception as e:
        print(f"❌ Basic connection failed: {e}")
        return False
    
    # Test dataset access
    datasets_to_test = [
        ("wikitext", "wikitext-2-raw-v1"),
        ("openwebtext", None),
        ("HuggingFaceFW/fineweb", None),
        ("allenai/c4", None)
    ]
    
    for dataset_name, config in datasets_to_test:
        try:
            print(f"\n🧪 Testing {dataset_name}...")
            
            if config:
                ds = load_dataset(dataset_name, config, split="train", streaming=True)
            else:
                ds = load_dataset(dataset_name, split="train", streaming=True)
                
            # Try to get first sample
            first_sample = next(iter(ds))
            print(f"✅ {dataset_name} accessible - Sample: {str(first_sample)[:100]}...")
            
        except Exception as e:
            print(f"❌ {dataset_name} failed: {e}")
    
    return True

if __name__ == "__main__":
    test_hf_connection()