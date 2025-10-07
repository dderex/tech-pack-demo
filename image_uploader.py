"""
Image Upload Manager for External Hosting

Uses free image hosting service (imgbb) to create persistent URLs.
Optimized with parallel uploads, retry logic, and progress tracking.
"""

import requests
import base64
from typing import Optional, Dict, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class ImageUploader:
    """Manager for uploading images to external hosting service."""
    
    def __init__(self, api_key: Optional[str] = None, max_workers: int = 3):
        """
        Initialize the image uploader.
        
        Args:
            api_key: API key for imgbb (free at https://api.imgbb.com/)
            max_workers: Maximum number of parallel upload threads
        """
        self.api_key = api_key
        self.upload_url = "https://api.imgbb.com/1/upload"
        self.max_workers = max_workers
        self.session = requests.Session()  # Reuse connections
    
    def is_configured(self) -> bool:
        """Check if uploader is properly configured."""
        return self.api_key is not None and len(self.api_key) > 0
    
    def upload_image(self, image_data: bytes, name: str = "sketch", max_retries: int = 3) -> Optional[Dict]:
        """
        Upload image to imgbb with retry logic.
        
        Args:
            image_data: Image bytes
            name: Name for the image
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dict with URL info or None if failed
        """
        if not self.is_configured():
            return None
        
        # Convert image bytes to base64 once
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Prepare upload payload
        payload = {
            'key': self.api_key,
            'image': image_base64,
            'name': name
        }
        
        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                # Upload to imgbb using session for connection reuse
                response = self.session.post(self.upload_url, data=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get('success'):
                        data = result['data']
                        return {
                            'url': data['url'],
                            'display_url': data['display_url'],
                            'delete_url': data.get('delete_url'),
                            'thumb_url': data.get('thumb', {}).get('url'),
                            'size': data.get('size'),
                            'expiration': data.get('expiration', 'No expiration')
                        }
                
                # If not successful but no exception, retry
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    
            except requests.RequestException as e:
                print(f"Upload attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
            except Exception as e:
                print(f"Unexpected error during upload: {e}")
                return None
        
        return None
    
    def _upload_single(self, img_data: dict, base_name: str, index: int) -> dict:
        """
        Helper method to upload a single image (used for parallel processing).
        
        Args:
            img_data: Dict with 'image_data' and metadata
            base_name: Base name for the image
            index: Image index
            
        Returns:
            Dict with image data and upload result
        """
        page_num = img_data.get('page', index)
        img_index = img_data.get('image_index', 1)
        name = f"{base_name}_page{page_num}_img{img_index}"
        
        upload_result = self.upload_image(img_data['image_data'], name)
        
        return {
            **img_data,
            'upload': upload_result,
            'index': index  # Preserve order
        }
    
    def upload_multiple(self, images: list, base_name: str = "sketch", 
                       progress_callback: Optional[Callable[[int, int], None]] = None) -> list:
        """
        Upload multiple images in parallel with progress tracking.
        
        Args:
            images: List of dicts with 'image_data' and metadata
            base_name: Base name for images
            progress_callback: Optional callback function(completed, total)
            
        Returns:
            List of upload results in original order
        """
        if not images:
            return []
        
        results = [None] * len(images)  # Pre-allocate list to maintain order
        completed = 0
        total = len(images)
        
        # Use ThreadPoolExecutor for parallel uploads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all upload tasks
            future_to_index = {
                executor.submit(self._upload_single, img_data, base_name, i): i
                for i, img_data in enumerate(images, 1)
            }
            
            # Process completed uploads
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    # Place result in correct position
                    results[result['index'] - 1] = result
                    completed += 1
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(completed, total)
                        
                except Exception as e:
                    print(f"Error processing upload: {e}")
                    index = future_to_index[future]
                    results[index - 1] = {
                        **images[index - 1],
                        'upload': None
                    }
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, total)
        
        # Remove index field from results
        for result in results:
            if result and 'index' in result:
                del result['index']
        
        return results
    
    def close(self):
        """Close the session and cleanup resources."""
        if hasattr(self, 'session'):
            self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

