#!/usr/bin/env python3

from datasets import load_dataset
import os
import json
import signal
import sys

class FineWebDownloader:
    def __init__(self, cache_dir="./fineweb_cache"):
        self.dataset_name = "HuggingFaceFW/fineweb"
        self.cache_dir = cache_dir
        self.progress_file = os.path.join(cache_dir, "download_progress.json")
        self.interrupted = False
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Setup signal handler for graceful interruption
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, _signum, _frame):
        print("\nDownload interrupted. Progress saved. Run script again to resume.")
        self.interrupted = True
        sys.exit(0)
    
    def load_progress(self):
        """Load download progress from file"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {"downloaded_count": 0}
    
    def save_progress(self, progress):
        """Save download progress to file"""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f)
    
    def download(self, max_documents=None):
        """Download FineWeb dataset with pause/resume capability"""
        
        print(f"Downloading {self.dataset_name}...")
        print(f"Cache directory: {self.cache_dir}")
        
        # Load previous progress
        progress = self.load_progress()
        start_count = progress["downloaded_count"]
        
        if start_count > 0:
            print(f"Resuming download from document {start_count}")
        
        # Load dataset in streaming mode
        dataset = load_dataset(
            self.dataset_name,
            name="default",
            streaming=True,
            split="train",
            cache_dir=self.cache_dir
        )
        
        count = 0
        try:
            for example in dataset:
                if count < start_count:
                    count += 1
                    continue
                
                # Save document to cache
                doc_file = os.path.join(self.cache_dir, f"doc_{count:08d}.json")
                with open(doc_file, 'w') as f:
                    json.dump(example, f)
                
                count += 1
                
                # Save progress every 100 documents
                if count % 100 == 0:
                    progress["downloaded_count"] = count
                    self.save_progress(progress)
                    print(f"Downloaded {count} documents...")
                
                # Check for interruption
                if self.interrupted:
                    break
                
                # Stop if we've reached max documents
                if max_documents and count >= max_documents:
                    print(f"Reached maximum of {max_documents} documents")
                    break
                    
        except KeyboardInterrupt:
            print(f"\nDownload paused at document {count}")
            progress["downloaded_count"] = count
            self.save_progress(progress)
            print(f"Progress saved. Run script again to resume from document {count}")
        
        # Final progress save
        progress["downloaded_count"] = count
        self.save_progress(progress)
        print(f"Download complete! Downloaded {count} documents to {self.cache_dir}")

def main():
    downloader = FineWebDownloader()
    
    # Download first 1000 documents as example (remove limit for full dataset)
    downloader.download(max_documents=1000)

if __name__ == "__main__":
    main()