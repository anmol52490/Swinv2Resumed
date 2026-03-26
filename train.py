import torch
# Enable TF32 for extreme speedups on Ampere GPUs (A5000)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset
from tqdm import tqdm
import sys
import os
import time
import datetime

from model import SwinUperNet
from utils import get_loaders, check_accuracy, save_checkpoint, MetricLogger, DiceCELoss, LovaszSoftmaxLoss

# --- Hyperparameters ---
LR = 1e-4 # Higher LR because we are training from scratch (Adapters + Decoder)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 24# SwinV2 + UperNet uses heavy VRAM; reduced to 16.
TOTAL_EPOCHS = 200
EVAL_FREQ = 5
LOSS_SWITCH_EPOCH = int(TOTAL_EPOCHS * 0.85)
IMG_HEIGHT = 384
IMG_WIDTH = 384

def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader, leave=False, file=sys.stdout, dynamic_ncols=True)
    total_loss = 0
    batch_losses = []

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)

        
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # scaler.scale(loss).backward()
        loss.backward()
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()

        loss_val = loss.item()
        total_loss += loss_val
        batch_losses.append(loss_val)
        loop.set_postfix(loss=loss_val)

    return total_loss / len(loader), batch_losses

def main():
    script_start_time = time.time()
    
    train_transform = A.Compose([
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=35, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet standards
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    save_dir = "epochs_200_384_manual_arch_class_weights"
    os.makedirs(save_dir, exist_ok=True)

    batch_loss_file = os.path.join(save_dir, "batch_losses_peft.csv")
    if not os.path.isfile(batch_loss_file):
        with open(batch_loss_file, mode='w', newline='') as f:
            f.write("Epoch,Batch_Index,Loss\n")

    # Initialize Model and Compile
    model = SwinUperNet(num_classes=104).to(DEVICE)
    # print("=> Compiling Model with torch.compile...")
    # model = torch.compile(model) # Compiles the execution graph for speed

    # Only pass parameters that require gradients to the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Place this right after initializing your model
    trainable_params1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Total trainable params: {trainable_params1:,}")
    print(f"Total params: {total_params:,}")
    print(f"{(trainable_params1 / total_params) * 100:.2f}% of params are trainable")
    optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=1e-4)
    # scaler = torch.amp.GradScaler('cuda')

    if os.path.exists("class_weights.pt"):
        print("=> Loading smoothed Inverse Frequency Class Weights...")
        class_weights = torch.load("class_weights.pt").to(DEVICE)
    else:
        print("=> WARNING: class_weights.pt not found. Using uniform weights.")
        class_weights = None
    
    dice_ce_loss = DiceCELoss(num_classes=104, weight=class_weights)
    lovasz_loss = LovaszSoftmaxLoss()
    active_loss_fn = dice_ce_loss

    dataset = load_dataset("EduardoPacheco/FoodSeg103", cache_dir="../FoodSegWithUnet/data/")
    train_loader, val_loader = get_loaders(dataset, BATCH_SIZE, train_transform, val_transform)

    logger = MetricLogger(main_file="metrics_peft200.csv", class_file="iou_peft200.csv")
    best_miou = 0.0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)
    
    print("--- Starting Training ---")
    for epoch in range(1, TOTAL_EPOCHS + 1):
        print(f"\nEpoch [{epoch}/{TOTAL_EPOCHS}]")
        
        if epoch == LOSS_SWITCH_EPOCH:
            print("=> Phase 2: Switching to LovaszSoftmaxLoss.")
            active_loss_fn = lovasz_loss

        avg_train_loss, current_batch_losses = train_fn(train_loader, model, optimizer, active_loss_fn)
        print(f"Average Train Loss: {avg_train_loss:.4f}")

        with open(batch_loss_file, mode='a', newline='') as f:
            for b_idx, b_loss in enumerate(current_batch_losses):
                f.write(f"{epoch},{b_idx},{b_loss}\n")

        scheduler.step()

        if epoch % EVAL_FREQ == 0 or epoch >= LOSS_SWITCH_EPOCH:
            print("=> Evaluating...")
            metrics = check_accuracy(val_loader, model, active_loss_fn, device=DEVICE)
            
            logger.log(epoch, avg_train_loss, metrics['val_loss'], metrics['miou'], metrics['pixel_acc'], metrics['per_class_iou'], metrics['mAcc'])

            if metrics['miou'] > best_miou:
                best_miou = metrics['miou']
                checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                save_dir = "epochs_200_384_manual_arch_class_weights"
                model_dir = os.path.join(save_dir, "models")

                os.makedirs(model_dir, exist_ok=True)

                filename = os.path.join(
                    model_dir,
                    f"{best_miou:.2f}MIOU_{avg_train_loss:.2f}Loss_{metrics['pixel_acc']:.2f}pixAcc_{metrics['mAcc']:.2f}mAcc_model.pth.tar"
                    )
                save_checkpoint(checkpoint, filename=filename)
        else:
            logger.log(epoch, avg_train_loss, "N/A", "N/A", "N/A", None, None)


    torch.cuda.synchronize(device=DEVICE)
    script_end_time = time.time()
    formatted_time = str(datetime.timedelta(seconds=int(script_end_time - script_start_time)))
    print("\n" + "="*50)
    print(f"✅ SCRIPT COMPLETE: Total execution time was {formatted_time}")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()