"""Deterministic digit → harmonic magnitude → additive audio building blocks.

Magnitudes are synthesis controls, not an invertible STFT. Frequency coordinates
are explicit Hz; synthesis does not pretend image rows are FFT bins.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.io.wavfile import write
import torch
from torch import nn
from torch.nn import functional as F


def normalize(image):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or not np.isfinite(image).all() or np.any(image < 0):
        raise ValueError("expected a finite nonnegative 2-D magnitude image")
    peak = image.max(initial=0)
    return image / peak if peak > 0 else np.zeros_like(image)


def select_harmonics(image, fundamental, frequencies=None, sigma=.6):
    """Nearest-bin Gaussian lines, including fractional third/fifth frequencies."""
    image = normalize(image)
    frequencies = np.arange(image.shape[0], dtype=float) if frequencies is None else np.asarray(frequencies)
    if frequencies.shape != (image.shape[0],) or not np.isfinite(frequencies).all() or np.any(np.diff(frequencies) <= 0):
        raise ValueError("frequencies must be a strictly increasing finite Hz grid")
    if not np.isfinite(fundamental) or fundamental <= 0 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("fundamental and sigma must be positive and finite")
    bins = np.arange(len(frequencies))
    mask = np.zeros(len(frequencies))
    for ratio in (1, 1.25, 1.5):
        for harmonic in range(1, int(frequencies[-1] / (fundamental * ratio)) + 1):
            center = np.interp(fundamental * ratio * harmonic, frequencies, bins)
            mask += np.exp(-.5 * ((bins-center)/sigma)**2) / harmonic**1.5
    mask[0] = 0  # No DC oscillator.
    return image * mask[:, None]


def target(image, label, shape=(129, 64)):
    if int(label) != label or not 0 <= label <= 9:
        raise ValueError("the standalone digit experiment supports labels 0–9")
    image = normalize(image)
    enlarged = zoom(image, (shape[0]/image.shape[0], shape[1]/image.shape[1]), order=1)
    enlarged = gaussian_filter(enlarged, (3, 2))
    envelope = np.sin(np.linspace(0, np.pi, shape[1]))**2
    frequencies = np.linspace(0, 2048, shape[0])
    result = select_harmonics(enlarged * envelope, 220 * 2**(label/12), frequencies)
    return normalize(result).astype(np.float32)


def synthesize(magnitude, frequencies, duration=1.5, sample_rate=8000, seed=7):
    magnitude = normalize(magnitude)
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.shape != (len(magnitude),) or not np.isfinite(frequencies).all() or np.any(frequencies < 0) or np.any(frequencies >= sample_rate/2):
        raise ValueError("one finite nonnegative sub-Nyquist frequency per row required")
    if not np.isfinite(duration) or not 0 < duration <= 60 or sample_rate < 1000:
        raise ValueError("duration must be in (0,60] seconds and sample rate >= 1000")
    n = int(round(duration * sample_rate))
    t = np.arange(n) / sample_rate
    frames = np.linspace(0, duration, magnitude.shape[1])
    audio = np.zeros(n)
    rng = np.random.default_rng(seed)
    for row, hz in zip(magnitude, frequencies):
        phase = rng.uniform(0, 2*np.pi)
        if hz > 0 and np.any(row > 0):
            envelope = np.interp(t, frames, row)
            audio += envelope * np.sin(2*np.pi*hz*t + phase)
    fade = min(n//2, round(.01*sample_rate))
    if fade:
        audio[:fade] *= np.linspace(0, 1, fade)
        audio[-fade:] *= np.linspace(1, 0, fade)
    peak = np.max(np.abs(audio), initial=0)
    return (audio * (.9/peak) if peak else audio).astype(np.float32)


def write_audio(path, magnitude, frequencies, duration=1.5, sample_rate=8000, seed=7):
    write(path, sample_rate, synthesize(magnitude, frequencies, duration, sample_rate, seed))


class CompactUNet(nn.Module):
    """Small CPU ablation of the notebook's image-only encoder/decoder idea.

    Not checkpoint-compatible with the historical full-resolution model.
    """
    def __init__(self, channels=8, output_shape=(129, 64)):
        super().__init__()
        def block(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.Conv2d(b, b, 3, padding=1), nn.ReLU())
        self.enc1, self.enc2 = block(1, channels), block(channels, 2*channels)
        self.middle = block(2*channels, 4*channels)
        self.dec2, self.dec1 = block(6*channels, 2*channels), block(3*channels, channels)
        self.head = nn.Conv2d(channels, 1, 3, padding=1)
        nn.init.constant_(self.head.bias, -3)  # Sparse magnitudes, not a 0.5-gray image.
        self.output_shape = output_shape

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.middle(F.max_pool2d(b, 2))
        d = self.dec2(torch.cat((F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b), 1))
        e = self.dec1(torch.cat((F.interpolate(d, size=a.shape[-2:], mode="bilinear", align_corners=False), a), 1))
        return torch.sigmoid(self.head(F.interpolate(e, size=self.output_shape, mode="bilinear", align_corners=False)))
