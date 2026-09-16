# U-Net: Handwritten Digits to Musical Spectrograms

A cross-modal deep-learning experiment that maps a 28×28 MNIST/EMNIST digit to
a 1008×1008 spectrogram representing a major triad. The project combines a
procedural target-generation pipeline, a PyTorch U-Net, perceptual image losses,
and waveform synthesis.

## Project idea

Digits `0`–`7` stand in for scale degrees. For each input image, the data
pipeline creates a target spectrogram whose vertical structure encodes a
fundamental, major third, and perfect fifth. A U-Net then learns the mapping
while preserving the digit's spatial character through skip connections.

## Technical highlights

- Procedural supervision built with resampling, Gaussian decay and smearing,
  harmonic row selection, and Perlin-noise texture
- Encoder–decoder model with skip connections and a 1008×1008 output
- Huber and multi-scale SSIM objectives for pixel and structural fidelity
- Mixed-precision training and checkpointing in PyTorch
- Spectrogram-to-waveform synthesis for audible qualitative evaluation

## Quick start

```bash
git clone https://github.com/takakhoo/mnist-to-music-unet.git
cd mnist-to-music-unet
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter lab UNet_MNIST_to_Spectrogram.ipynb
```

For a bounded CPU smoke test of the complete pipeline, execute the notebook in
fast-development mode:

```bash
UNET_FAST_DEV_RUN=1 python -m jupyter nbconvert \
  --to notebook \
  --execute UNet_MNIST_to_Spectrogram.ipynb \
  --output /tmp/mnist-to-music-unet-verified.ipynb \
  --ExecutePreprocessor.timeout=900
```

This mode uses two EMNIST samples, a batch size of one, and one training epoch.
It still exercises data preparation, target construction, optimization,
inference, spectrogram rendering, and waveform synthesis. The default notebook
configuration preserves the original ten-epoch research experiment.

The committed checkpoint is an interrupted snapshot from an older model
configuration. The notebook intentionally does not load it during the smoke
test; loading it in the full path requires matching the historical channel
dimensions.

## Verification

The fast-development command above was executed successfully on macOS with
Python 3.13 on September 16, 2026. It produced both predicted spectrogram and
audio artifacts without requiring a GPU. Generated datasets and smoke-test
outputs are intentionally excluded from version control.

## Artifacts

- `UNet_MNIST_to_Spectrogram.ipynb` — complete experiment notebook
- `Paper MNIST to Chord Spectrogram.pdf` — project report
- `Final Presentation.pptx` — presentation deck
- `thenumber*.png` / `thenumber*.npy` — conditioning-stage examples
- `redu_U-Nettttttt_checkpoint_step_156819_INTERRUPTED.pth` — historical
  checkpoint
- `Upscaler/` — Real-ESRGAN exploration and model assets

## Contributors

Taka Khoo, Harry Leiter, and Doruk Ozel developed this project for ENGS 106.

## Scope

This is a research/course prototype. The audio examples provide qualitative
evidence; a stronger follow-up would add held-out metrics, deterministic
training configuration, and a standalone inference script.
