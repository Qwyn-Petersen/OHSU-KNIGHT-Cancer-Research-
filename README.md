# Learning the Language of Invasion

**A deep learning framework for cell–cell interaction and ERK signaling in breast cancer**

Qwynestria Petersen, Vaibhav Murthy, Jeremy Copperman, Alexander Davies
Cancer Early Detection Advanced Research Center (CEDAR), Knight Cancer Institute, Oregon Health & Science University

---

## Overview

What causes breast cancer cells to invade beyond the mammary gland? And which patterns in 4D imaging (3D + time) link a cell's surroundings to its **ERK signaling**, a pathway tied to cancer cell growth, survival and transformation?

Studying cancer in 4D is computationally expensive, complex and time-consuming. This project is one piece of a larger imaging pipeline built to address that. It trains a **variational autoencoder (VAE)** on multi-channel single-cell images to learn a compact **latent space** that captures cell state, and then explores that space for patterns linked to early-stage invasion.

A VAE compresses each image into a small set of numbers (the latent space) and learns to rebuild the image from them. Because it's generative, it can also:

- **Reconstruct** cells from their compressed representation
- **Interpolate** between two cells to show the transition between states
- **Map** cells so that similar behaviors cluster together

## Where this fits in the pipeline

1. **Imaging (SITE):** The Serial Imaging of Tumor and microEnvironment platform captures live 4D interactions between ex-vivo mammary host organoids and tumor cell lines (T4-2, MDA-MB-231).
2. **Segmentation:** Cellpose and ilastik outline cancer cells, host cells and tissue structures. A Cellpose model retrained with manual annotations greatly improved segmentation accuracy.
3. **Tracking:** Cells are matched across frames to follow their movement, shape and ERK activity over time.
4. **VAE (this repo):** Single-cell images and masks are encoded into a latent space to look for invasion-related patterns.

## Data

- Single-cell `.tif` crops from SITE imaging (10,656 images in the run shown in the notebook)
- Each sample has **5 channels**: 3 segmentation masks (cytoplasm, host cell, cancer cell) and 2 images (bright-field, ERK reporter)
- Images are resized to **64 × 64** (input shape `5 × 64 × 64`)

The imaging data belongs to the lab and is **not included** in this repository.

## What the notebook does

All of the code is in **`vae_pipeline_full_1.ipynb`**, organized as step-by-step cells:

| Step | Description |
|---|---|
| **Setup & config** | Imports, random seed (42) and settings: 2D/3D mode, image size, latent size, batch size, epochs, learning rate, data path |
| **Dataset** | `TifDataset` loads 2D or 3D, single- or multi-channel `.tif` files, scales them to [0, 1], resizes them and normalizes each channel |
| **Normalization stats** | Computes exact per-channel mean and standard deviation over the full dataset in one pass (Welford's algorithm) and caches them to `norm_stats.json` |
| **Model** | Convolutional `VAE2D` (and `VAE3D` for volumes) behind a `VAEUnified` wrapper. Layers are built automatically from the first input's size. The 2D model uses 3 strided conv layers (32 → 64 → 128 filters) and a **64-dimensional latent space** (~1.9 M parameters) |
| **Training** | Adam optimizer, MSE reconstruction loss + KL divergence with a β warm-up schedule, and mixed precision on GPU |
| **Visualization** | Side-by-side original vs. reconstructed images per channel, and intensity histograms comparing the two |
| **Checkpointing** | Saves and reloads the model, optimizer state, normalization stats and config (`vae_checkpoint_simple.pth`) |
| **Latent extraction** | Encodes every cell to its mean latent vector and saves the embeddings to `vae_latents.npy` |
| **Clustering benchmark** | Reduces latents to 2D with **PCA, t-SNE and UMAP**, clusters with **K-Means, Agglomerative, GMM, DBSCAN and HDBSCAN**, scores each with the silhouette score, and saves results to `clusters_<method>.csv` |
| **Latent interpolation** | Blends the latent vectors of two cells and decodes the steps in between to check for smooth, realistic transitions |

## Results

- The VAE reconstructs multi-channel single-cell images from a 64-dimensional latent space.
- **Interpolation** between cells produces smooth transitions, which suggests the latent space learned meaningful features rather than noise.
- **t-SNE projections** of the latent space show structure associated with cancer cell invasion behavior.
- **Single-cell trajectories** through the latent space, colored by ERK intensity, link cell behavior to microenvironment cues such as contact with host tissue.
- Overall, VAEs show promise for modeling early-stage invasion by capturing subtle single-cell transitions, but the approach still needs refinement.

## How to run

1. Install the dependencies (a CUDA GPU is recommended):
   ```bash
   pip install torch numpy pandas matplotlib seaborn scikit-learn umap-learn hdbscan tifffile tqdm jupyter
   ```
2. Open `vae_pipeline_full_1.ipynb` and edit the **CONFIG** section in the first code cell:
   - Set `data_path_2D` / `data_path_3D` to your folder of `.tif` images
   - Set `is_3d = True` for 3D volumes
   - Adjust `epochs`, `latent_dim`, `batch_size` and `lr` as needed
3. Run the cells in order. Normalization stats are computed once and then loaded from `norm_stats.json` on later runs.

### Outputs

| File | Contents |
|---|---|
| `norm_stats.json` | Per-channel mean and standard deviation |
| `vae_checkpoint_simple.pth` | Trained model, optimizer state and config |
| `vae_latents.npy` | Latent embedding for every cell |
| `clusters_PCA.csv`, `clusters_t-SNE.csv`, `clusters_UMAP.csv` | 2D coordinates and cluster labels from each method |

## References

- Davies, A. E., et al. (2020). Systems-level properties of EGFR-RAS-ERK signaling amplify local signals to generate dynamic gene expression heterogeneity. *Cell Systems*, 11(2), 85–100.e9. https://doi.org/10.1016/j.cels.2020.07.004
- Murthy, V., et al. (in preparation). Serial Imaging of Tumor and microEnvironment (SITE) platform for live-cell modeling of cancer progression dynamics. Oregon Health & Science University.
