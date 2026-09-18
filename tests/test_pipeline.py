import numpy as np
import pytest
import torch
from digit_audio import normalize, select_harmonics, target, synthesize, CompactUNet

def test_fractional_harmonics_not_dropped():
    # Original floating-point modulo removed thirds/fifths unless exactly integral.
    spectrum = select_harmonics(np.ones((1008, 3)), 111)
    for hz in (111, 138.75, 166.5):
        assert spectrum[round(hz), 1] > .5

def test_silence_is_finite_and_reproducible():
    silence = synthesize(np.zeros((3, 8)), [0, 100, 200])
    assert np.all(silence == 0)
    image = np.eye(8)
    np.testing.assert_array_equal(target(image, 4), target(image, 4))
    audio = synthesize(np.ones((3, 8)), [0, 100, 200])
    np.testing.assert_array_equal(audio, synthesize(np.ones((3, 8)), [0, 100, 200]))
    assert np.max(np.abs(audio)) <= .901

def test_frequency_grid_matches_audio():
    audio = synthesize(np.ones((1, 8)), [440], duration=1)
    frequencies = np.fft.rfftfreq(len(audio), 1/8000)
    assert frequencies[np.argmax(np.abs(np.fft.rfft(audio)))] == 440

def test_validation():
    for label in (-1, .5, 10):
        with pytest.raises(ValueError):
            target(np.eye(8), label)
    with pytest.raises(ValueError):
        normalize(np.array([[np.nan]]))
    with pytest.raises(ValueError):
        synthesize(np.ones((1, 2)), [4000])

def test_model_shape_range_and_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = CompactUNet(channels=4)
    prediction = model(torch.rand(2, 1, 8, 8))
    assert prediction.shape == (2, 1, 129, 64)
    assert prediction.min() >= 0 and prediction.max() <= 1
    prediction.mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
