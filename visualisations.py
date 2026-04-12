import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class SegmentationVisualizer:
    def __init__(self, metrics_csv="metrics_peft.csv", batch_csv="batch_losses.csv", iou_csv="iou_peft.csv"):
        self.metrics_csv = metrics_csv
        self.batch_csv = batch_csv
        self.iou_csv = iou_csv
        
        # Output directory for saved plots
        self.out_dir = r"D:\swinv2resumed\Swinv2UpernetFoodseg\epochs_200_640_improvedFPN\visualizations"
        os.makedirs(self.out_dir, exist_ok=True)

    def load_clean_metrics(self):
        df = pd.read_csv(self.metrics_csv)
        # Convert numeric columns, forcing 'N/A' to NaN
        for col in ['Train_Loss', 'Val_Loss', 'Val_mIoU', 'Val_Pixel_Acc', 'mAcc']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def load_clean_iou(self):
        df = pd.read_csv(self.iou_csv)
        # Replace 'N/A' with NaN and drop rows where evaluation was skipped
        df = df.replace('N/A', np.nan)
        df = df.dropna(subset=['Class_0']) # If Class_0 is missing, the whole eval was skipped
        
        # Convert all class columns to float
        class_cols = [col for col in df.columns if col.startswith('Class_')]
        df[class_cols] = df[class_cols].astype(float)
        return df, class_cols

    def plot_macro_convergence(self):
        print("Generating Macro Convergence Plot...")
        df = self.load_clean_metrics()
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['Epoch'], df['Train_Loss'], label='Train Loss', color='blue', alpha=0.6)
        
        val_df = df.dropna(subset=['Val_Loss'])
        plt.plot(val_df['Epoch'], val_df['Val_Loss'], label='Validation Loss', color='red', marker='o', linewidth=2)
        
        plt.title('Macro Convergence: Train vs. Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{self.out_dir}/1_macro_convergence.png", dpi=300)
        plt.close()

    def plot_metric_divergence(self):
        print("Generating Metric Divergence Plot...")
        df = self.load_clean_metrics().dropna(subset=['Val_mIoU'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['Epoch'], df['Val_Pixel_Acc'], label='Pixel Accuracy (%)', color='green', marker='s')
        plt.plot(df['Epoch'], df['Val_mIoU'], label='mIoU (%)', color='purple', marker='^')
        
        if 'mAcc' in df.columns:
            plt.plot(df['Epoch'], df['mAcc'], label='mAcc (%)', color='orange', marker='d')
            
        plt.title('Metric Divergence Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Percentage (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{self.out_dir}/2_metric_divergence.png", dpi=300)
        plt.close()

    def plot_batch_ema(self, span=200):
        print("Generating Batch Loss EMA Plot...")
        if not os.path.exists(self.batch_csv):
            print(f"Warning: {self.batch_csv} not found. Skipping EMA plot.")
            return

        df = pd.read_csv(self.batch_csv)
        df['EMA'] = df['Loss'].ewm(span=span, adjust=False).mean()
        
        # Create a continuous step index across all epochs
        df['Global_Step'] = range(len(df))

        plt.figure(figsize=(12, 6))
        plt.plot(df['Global_Step'], df['Loss'], color='gray', alpha=0.2, label='Raw Batch Loss')
        plt.plot(df['Global_Step'], df['EMA'], color='red', linewidth=2, label=f'EMA (span={span})')
        
        plt.title('Micro-Stability: Batch Loss Exponential Moving Average')
        plt.xlabel('Global Batch Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.out_dir}/3_batch_loss_ema.png", dpi=300)
        plt.close()

    def plot_terminal_performance(self):
        print("Generating Terminal Performance Bar Chart...")
        df, class_cols = self.load_clean_iou()
        if df.empty: return

        # Get the final epoch's data
        final_epoch = df.iloc[-1]
        class_ious = final_epoch[class_cols].sort_values()

        plt.figure(figsize=(10, 20)) # Tall figure for 104 classes
        class_ious.plot(kind='barh', color='teal')
        plt.title(f"Terminal Performance by Class (Epoch {int(final_epoch['Epoch'])})")
        plt.xlabel("IoU (%)")
        plt.ylabel("Class ID")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{self.out_dir}/4_terminal_performance.png", dpi=300)
        plt.close()

    def plot_top_bottom_trajectories(self):
        print("Generating Top/Bottom Class Trajectories...")
        df, class_cols = self.load_clean_iou()
        if df.empty: return

        final_epoch = df.iloc[-1]
        sorted_classes = final_epoch[class_cols].sort_values(ascending=False)
        
        top_5 = sorted_classes.head(5).index.tolist()
        bottom_5 = sorted_classes.tail(5).index.tolist()

        plt.figure(figsize=(12, 6))
        
        # Plot Top 5
        for cls in top_5:
            plt.plot(df['Epoch'], df[cls], linestyle='-', marker='o', label=f'Top: {cls}')
            
        # Plot Bottom 5
        for cls in bottom_5:
            plt.plot(df['Epoch'], df[cls], linestyle='--', marker='x', label=f'Bot: {cls}')

        plt.title("Class Trajectories: Top 5 vs Bottom 5 Performers")
        plt.xlabel("Epoch")
        plt.ylabel("IoU (%)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{self.out_dir}/5_class_trajectories.png", dpi=300)
        plt.close()

    def plot_training_heatmap(self):
        print("Generating Training Heatmap...")
        df, class_cols = self.load_clean_iou()
        if df.empty: return

        # Set Epoch as index and transpose so classes are on the Y axis
        heatmap_data = df.set_index('Epoch')[class_cols].T
        
        plt.figure(figsize=(14, 20))
        sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'IoU (%)'}, vmin=0, vmax=1)
        plt.title("Class Learning Heatmap Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Class ID")
        plt.tight_layout()
        plt.savefig(f"{self.out_dir}/6_training_heatmap.png", dpi=300)
        plt.close()

    def generate_all(self):
        print("--- Starting Visualization Generation ---")
        self.plot_macro_convergence()
        self.plot_metric_divergence()
        self.plot_batch_ema()
        self.plot_terminal_performance()
        self.plot_top_bottom_trajectories()
        self.plot_training_heatmap()
        print(f"--- Complete. All plots saved to ./{self.out_dir}/ ---")

if __name__ == "__main__":
    # Adjust filenames here if your actual CSVs are named differently
    visualizer = SegmentationVisualizer(
        metrics_csv=r"D:\swinv2resumed\Swinv2UpernetFoodseg\epochs_200_640_improvedFPN\metrics_peft200.csv",  
        batch_csv=r"D:\swinv2resumed\Swinv2UpernetFoodseg\epochs_200_640_improvedFPN\batch_losses_peft.csv",
        iou_csv=r"D:\swinv2resumed\Swinv2UpernetFoodseg\epochs_200_640_improvedFPN\iou_peft200.csv"           
    )
    visualizer.generate_all()