# Satellite Image Super Resolution using Deep Learning

This project focuses on improving the quality and resolution of low-resolution satellite images using Deep Learning techniques with PyTorch.

## Project Overview

Satellite images often suffer from low resolution due to limitations in image acquisition systems. This project applies a Super Resolution model to generate clearer and higher-quality satellite imagery.

## Features

- Deep Learning based Super Resolution (SRMamba-T inspired architecture)
- PyTorch implementation
- Satellite image enhancement (4× upscale)
- Model training and inference
- PSNR / SSIM evaluation metrics
- Visual comparison grids (LR | Bicubic | SR | HR)

## Technologies Used

- Python
- PyTorch
- NumPy
- OpenCV
- Matplotlib
- Kaggle Notebook

## Project Structure

```text
satellite-image-super-resolution/
│
├── satellite_sr_training.ipynb   # Training notebook (run on Kaggle)
├── test_inference.py             # Testing / inference script
├── sat_sr_model.pth              # Pre-trained model weights
├── satellite_sr_results.png      # Sample results
├── README.md
└── requirements.txt
```

## Datasets

This project uses the following Kaggle datasets:
- **DIV2K** — Standard SR benchmark dataset
- **Planets Dataset** — Planet satellite imagery for training
- **4× Satellite Image Super Resolution** — HR/LR satellite image pairs for evaluation

## Model Architecture

SRMamba-T inspired SR network with:
- Residual blocks for local feature extraction
- Channel attention for spectral recalibration
- Spatial attention for satellite spatial structure focus
- PixelShuffle (×2 chained twice) for stable 4× upsampling

## How to Use on Kaggle

### Training
1. Create a new Kaggle notebook with GPU enabled
2. Add the **Planets Dataset** (`nikitarom/planets-dataset`) as input
3. Upload or paste the code from `satellite_sr_training.ipynb`
4. Run all cells — the trained model saves to `/kaggle/working/satellite_sr_model.pth`

### Testing / Inference
1. Create a new Kaggle notebook with GPU enabled
2. Add these datasets as input:
   - **DIV2K** (`eugeneshenderov/div2k-dataset-for-super-resolution`)
   - **4× Satellite SR** (`cristobaltudela/4x-satellite-image-super-resolution`)
3. Import this repository as a Kaggle dataset (or upload `sat_sr_model.pth`)
4. Upload or paste the code from `test_inference.py`
5. The script auto-detects the checkpoint location and runs evaluation

## Results

The model successfully improves image clarity and sharpness for satellite imagery, producing PSNR gains over bicubic interpolation baseline.

## Future Improvements

- Improve PSNR and SSIM scores
- Train on larger datasets
- Deploy as web application
- Add real-time inference

## Author

Vujja Punith Sai

## License

This project is for educational and research purposes.
