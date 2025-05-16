import torch
import clip
from PIL import Image
import numpy as np
import os
import ssl

# SSL context configuration
ssl._create_default_https_context = ssl._create_unverified_context

def generate_image_embeddings_clip(images):
    """
    Generate embeddings for a list of images using CLIP.
    
    Args:
        images (list): List of image paths
        
    Returns:
        numpy.ndarray: Matrix of embeddings for each image
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting manual model download...")
        cache_dir = os.path.expanduser("~/.cache/clip")
        os.makedirs(cache_dir, exist_ok=True)
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False, download_root=cache_dir)
    
    # Preprocess and encode images
    image_features = []
    for image_path in images:
        image = Image.open(image_path)
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_feature = model.encode_image(image_input)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            image_features.append(image_feature)
    
    return torch.cat(image_features).cpu().numpy()

# Example usage
if __name__ == "__main__":
    # Example image paths (to be replaced with actual paths)
    image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
    
    # Generate image embeddings
    image_embeddings = generate_image_embeddings_clip(image_paths)
    print(f"Image embeddings shape: {image_embeddings.shape}")
    print("(number of images, CLIP embedding dimension)") 