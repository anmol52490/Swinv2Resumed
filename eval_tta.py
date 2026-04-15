import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from datasets import load_dataset

from model import SwinUperNet

# -----------------------------
# Config
# -----------------------------
CHECKPOINT_PATH = r"D:\swinv2resumed\Swinv2UpernetFoodseg\epochs_200_Tuna_640\models\47.94MIOU_0.13Loss_80.29pixAcc_57.99mAcc_model.pth.tar"
DATASET_NAME = "EduardoPacheco/FoodSeg103"
CACHE_DIR = "../FoodSegWithUnet/data/"
SPLIT_NAME = "validation"

NUM_CLASSES = 104
BASE_SIZE = (640, 640)

# How many original images to process together
IMAGE_BATCH_SIZE = 2   # try 2, 4, 6, 8 depending on VRAM

# Keep the original TTA behavior for accuracy
TTA_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
TTA_FLIPS = [False, True]

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
# Helpers
# -----------------------------
def to_pil(x):
    if isinstance(x, Image.Image):
        return x
    return Image.fromarray(np.array(x))

def preprocess_sample(sample, transform):
    image = to_pil(sample["image"]).convert("RGB")
    mask = to_pil(sample["label"])

    img_tensor = transform(image)  # (3, 640, 640)
    mask = mask.resize(BASE_SIZE, NEAREST)
    mask_np = np.array(mask, dtype=np.int64)

    return img_tensor, mask_np


# -----------------------------
# TTA Engine for a batch of images
# -----------------------------
@torch.inference_mode()
def predict_with_tta_batch(model, batch_tensor, base_size=BASE_SIZE):
    """
    batch_tensor: shape (B, 3, H, W)
    returns: shape (B, H, W)
    """
    B = batch_tensor.shape[0]
    final_logits = torch.zeros(
        (B, NUM_CLASSES, base_size[0], base_size[1]),
        device=batch_tensor.device,
        dtype=batch_tensor.dtype,
    )

    total_passes = 0

    for scale in TTA_SCALES:
        h, w = int(base_size[0] * scale), int(base_size[1] * scale)

        # Scale the whole batch together
        scaled = F.interpolate(batch_tensor, size=(h, w), mode="bilinear", align_corners=False)

        for flip in TTA_FLIPS:
            x = torch.flip(scaled, dims=[3]) if flip else scaled

            logits = model(x)

            if flip:
                logits = torch.flip(logits, dims=[3])

            logits = F.interpolate(logits, size=base_size, mode="bilinear", align_corners=False)

            final_logits += logits
            total_passes += 1

    final_logits /= total_passes
    preds = torch.argmax(final_logits, dim=1)
    return preds


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

    transform = transforms.Compose([
        transforms.Resize(BASE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    total_hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

    print(f"Starting TTA Evaluation on {len(dataset)} validation images...")
    print(f"Batching {IMAGE_BATCH_SIZE} images at a time.")

    for start in tqdm(range(0, len(dataset), IMAGE_BATCH_SIZE)):
        end = min(start + IMAGE_BATCH_SIZE, len(dataset))
        samples = [dataset[i] for i in range(start, end)]

        batch_imgs = []
        batch_masks = []

        for sample in samples:
            img_tensor, mask_np = preprocess_sample(sample, transform)
            batch_imgs.append(img_tensor)
            batch_masks.append(mask_np)

        batch_tensor = torch.stack(batch_imgs, dim=0).to(DEVICE, non_blocking=True)

        try:
            pred_batch = predict_with_tta_batch(model, batch_tensor, base_size=BASE_SIZE)
            pred_batch_np = pred_batch.cpu().numpy().astype(np.int64)

            for i in range(len(samples)):
                total_hist += fast_hist(
                    batch_masks[i].flatten(),
                    pred_batch_np[i].flatten(),
                    NUM_CLASSES
                )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\nCUDA OOM during batched TTA.")
                print("Lower IMAGE_BATCH_SIZE first. If needed, then reduce TTA_SCALES.")
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