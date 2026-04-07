import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from datasets import load_dataset

# Import YOUR model architecture
from model import SwinUperNet

# -----------------------------
# Config
# -----------------------------
CHECKPOINT_PATH = r"D:\swinv2resumed\Swinv2UpernetFoodseg\epochs_200_640_improvedFPN\models\46.65MIOU_0.14Loss_82.42pixAcc_58.53mAcc_model.pth.tar"
DATASET_NAME = "EduardoPacheco/FoodSeg103"
CACHE_DIR = "../FoodSegWithUnet/data/"
SPLIT_NAME = "validation"
NUM_CLASSES = 104
BASE_SIZE = (640, 640)

# PIL compatibility for resize interpolation
NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST

# -----------------------------
# Metric Calculation Utilities
# -----------------------------
def fast_hist(a, b, n):
    """Calculates confusion matrix for a single batch/image."""
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def compute_metrics(hist):
    """Calculates standard segmentation metrics from the confusion matrix."""
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-8)
    valid_classes = hist.sum(1) > 0
    miou = np.nanmean(iu[valid_classes])

    pixel_acc = np.diag(hist).sum() / (hist.sum() + 1e-8)

    acc_cls = np.diag(hist) / (hist.sum(1) + 1e-8)
    macc = np.nanmean(acc_cls[valid_classes])

    return miou * 100, pixel_acc * 100, macc * 100, iu

# -----------------------------
# TTA Engine
# -----------------------------
def predict_with_tta_batch(model, image_tensor, base_size=(640, 640)):
    scales = [0.75, 1.0, 1.25]
    flips = [False, True]

    augmented_images = []

    for scale in scales:
        for flip in flips:
            h, w = int(base_size[0] * scale), int(base_size[1] * scale)

            img = F.interpolate(image_tensor, size=(h, w), mode='bilinear', align_corners=False)

            if flip:
                img = torch.flip(img, dims=[3])

            img = F.interpolate(img, size=base_size, mode='bilinear', align_corners=False)

            augmented_images.append(img)

    batch = torch.cat(augmented_images, dim=0)  # shape: (6, 3, 640, 640)

    with torch.no_grad():
        logits = model(batch)  # ONE forward pass

    # reverse flips
    idx = 0
    restored = []

    for scale in scales:
        for flip in flips:
            logit = logits[idx:idx+1]
            if flip:
                logit = torch.flip(logit, dims=[3])
            restored.append(logit)
            idx += 1

    final_logits = torch.stack(restored).mean(dim=0)
    final_mask = torch.argmax(final_logits, dim=1)

    return final_mask

# -----------------------------
# Main Evaluation Loop
# -----------------------------
def evaluate_dataset():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading dataset split '{SPLIT_NAME}'...")
    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT_NAME,
        cache_dir=CACHE_DIR
    )

    print(f"Loading Architecture and Weights into {DEVICE}...")
    model = SwinUperNet(num_classes=NUM_CLASSES).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Must match training preprocessing
    transform = transforms.Compose([
        transforms.Resize(BASE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    total_hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

    print(f"Starting TTA Evaluation on {len(dataset)} validation images...")

    for sample in tqdm(dataset):
        # Load image
        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        image = image.convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Load mask
        mask = sample["label"]
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.array(mask))

        mask = mask.resize(BASE_SIZE, NEAREST)
        mask_np = np.array(mask, dtype=np.int64)

        try:
            pred_mask = predict_with_tta_batch(model, img_tensor, base_size=BASE_SIZE)
            pred_mask_np = pred_mask.squeeze(0).cpu().numpy().astype(np.int64)

            total_hist += fast_hist(mask_np.flatten(), pred_mask_np.flatten(), NUM_CLASSES)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\nCUDA OOM during TTA. Remove the 1.5 and 1.75 scales.")
                return
            raise e

    miou, pix_acc, macc, _ = compute_metrics(total_hist)

    print("\n" + "=" * 40)
    print("TTA EVALUATION RESULTS")
    print("=" * 40)
    print(f"Final TTA mIoU:      {miou:.2f}%")
    print(f"Final TTA Pixel Acc:  {pix_acc:.2f}%")
    print(f"Final TTA mAcc:       {macc:.2f}%")
    print("=" * 40)

if __name__ == "__main__":
    evaluate_dataset()