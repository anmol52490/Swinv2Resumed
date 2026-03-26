import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

def generate_smoothed_weights():
    print("Calculating exact pixel distribution for Class Weights...")
    dataset = load_dataset("EduardoPacheco/FoodSeg103", cache_dir=r"D:\swinv2resumed\FoodSegWithUnet\data")
    train_data = dataset['train']
    
    num_classes = 104
    pixel_counts = np.zeros(num_classes, dtype=np.int64)

    for item in tqdm(train_data):
        mask = np.array(item['label'])
        unique_classes, counts = np.unique(mask, return_counts=True)
        for cls, count in zip(unique_classes, counts):
            if cls < num_classes:
                pixel_counts[cls] += count

    frequencies = pixel_counts / pixel_counts.sum()
    
    # Logarithmic smoothing: prevents minority classes from causing gradient explosions
    # The 1.02 constant ensures the denominator never hits 0 or goes negative
    weights = 1.0 / np.log(1.02 + frequencies)
    
    # Normalize so the most common class has a weight of 1.0
    weights = weights / weights.min()
    
    tensor_weights = torch.tensor(weights, dtype=torch.float32)
    torch.save(tensor_weights, "class_weights.pt")
    
    print("\nWeights generated successfully:")
    print(f"Background Weight (Class 0): {tensor_weights[0]:.4f}")
    print(f"Max Minority Weight: {tensor_weights.max():.4f}")

if __name__ == "__main__":
    generate_smoothed_weights()