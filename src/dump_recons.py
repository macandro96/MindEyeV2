"""Dump recons across seeds"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def plot_all_side_by_side_grid(all_recons_save_tensor, all_ground_truth_save_tensor, save_path: str, images_per_row=10):
    """
    Plots all ground truth and reconstructed RGB images side-by-side in one big grid.
    Images are arranged in rows, each row contains pairs of images (GT and recon) for multiple images.

    Args:
        all_recons_save_tensor: (N, 3, H, W) tensor/array of reconstructed images
        all_ground_truth_save_tensor: (N, 3, H, W) tensor/array of ground-truth images
        images_per_row: Number of image pairs per row (default 10)
    """
    def prepare_image_for_plot(img):
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0
        img = np.clip(img, 0, 1)
        return img

    if hasattr(all_recons_save_tensor, 'cpu'):
        recons = all_recons_save_tensor.cpu().numpy()
    else:
        recons = np.array(all_recons_save_tensor)
    if hasattr(all_ground_truth_save_tensor, 'cpu'):
        gt = all_ground_truth_save_tensor.cpu().numpy()
    else:
        gt = np.array(all_ground_truth_save_tensor)

    num_images = recons.shape[0]
    images_per_row = min(images_per_row, num_images)
    num_rows = int(np.ceil(num_images / images_per_row))

    # Create figure and axes: 2 columns per image (GT + recon), images_per_row pairs per row
    fig, axs = plt.subplots(num_rows, images_per_row * 2, figsize=(2 * images_per_row * 3, num_rows * 3))

    for i in range(num_images):
        row = i // images_per_row
        col_gt = (i % images_per_row) * 2
        col_recon = col_gt + 1

        gt_img = np.transpose(gt[i], (1, 2, 0))
        recons_img = np.transpose(recons[i], (1, 2, 0))

        gt_img = prepare_image_for_plot(gt_img)
        recons_img = prepare_image_for_plot(recons_img)

        if num_rows == 1:
            axs[col_gt].imshow(gt_img)
            axs[col_gt].set_title(f'GT {i}')
            axs[col_gt].axis('off')

            axs[col_recon].imshow(recons_img)
            axs[col_recon].set_title(f'Recon {i}')
            axs[col_recon].axis('off')
        else:
            axs[row, col_gt].imshow(gt_img)
            axs[row, col_gt].set_title(f'GT {i}')
            axs[row, col_gt].axis('off')

            axs[row, col_recon].imshow(recons_img)
            axs[row, col_recon].set_title(f'Recon {i}')
            axs[row, col_recon].axis('off')

    # Hide any empty subplots if total slots > num_images*2
    total_slots = num_rows * images_per_row * 2
    for j in range(num_images * 2, total_slots):
        if num_rows == 1:
            axs[j].axis('off')
        else:
            row = j // (images_per_row * 2)
            col = j % (images_per_row * 2)
            axs[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def recons_per_seed(args: argparse.Namespace):
    for seed in tqdm(range(args.seed_range)):
        seed_dir = os.path.join(args.input_dir, f"{seed}")
        recon_file = [file for file in os.listdir(seed_dir) if file.endswith("_recons.pt")][0]

        all_images = torch.load(args.all_images_path)
        all_recons = torch.load(os.path.join(seed_dir, recon_file))
        if args.indices_path:
            indices = pd.read_csv(args.indices_path)['indices'].tolist()
            all_images = all_images[indices]
            # all_recons = all_recons[indices]
            
            # remap indices to match ordering in all_recons
            indices2idx_ord = {ind: idx for idx, ind in enumerate(sorted(indices))}
            idx2choose = [indices2idx_ord[ind] for ind in indices]
            all_recons = all_recons[idx2choose]
        
        save_path = os.path.join(seed_dir, f"recons_vs_gt.png")
        plot_all_side_by_side_grid(all_recons, all_images, save_path)
        print(f"Saved reconstructions vs ground truth for seed={seed} at {save_path}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="path to input dir containing seed folders")
    parser.add_argument("--seed_range", type=int, default=5, help="number of seeds to collate")
    parser.add_argument("--all_images_path", type=str, help="path to all images tensor file", default="./mindeyev2_data/evals/all_images.pt")
    parser.add_argument("--indices_path", type=str, help="path to indices file", default="./special515_indices.csv")
    args = parser.parse_args()
    
    recons_per_seed(args)