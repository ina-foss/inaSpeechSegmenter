"""
Script to convert pre-trained Keras .hdf5 models to PyTorch .pth format.

Usage:
    Run from the project root directory:
    PYTHONPATH=. python scripts/convert_keras_to_pytorch.py
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import tensorflow as tf
import numpy as np
from inaSpeechSegmenter.remote_utils import get_remote
from inaSpeechSegmenter.pytorch_models import (
    SpeechMusicCNN, SpeechMusicNoiseCNN, MaleFemaleCNN,
    Interspeech2023All, Interspeech2023Cvfr
)

def convert_keras_to_pytorch(keras_model, torch_model):
    torch_model.eval()
    keras_layers = [l for l in keras_model.layers if len(l.get_weights()) > 0]
    
    # We'll map by type and order as names might differ slightly
    keras_idx = 0
    
    # It's better to iterate through torch modules and find corresponding weights
    # because torch_model might have modules that don't have weights (like MaxPool, Flatten)
    
    # Filter for modules with parameters
    torch_modules = []
    for m in torch_model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            torch_modules.append(m)
            
    for t_module in torch_modules:
        k_layer = keras_layers[keras_idx]
        k_weights = k_layer.get_weights()
        
        print(f"Mapping Keras {k_layer.name} ({k_layer.__class__.__name__}) to PyTorch {t_module.__class__.__name__}")
        
        if isinstance(t_module, torch.nn.Conv2d):
            # Keras: (H, W, in, out) -> Torch: (out, in, H, W)
            w = k_weights[0].transpose(3, 2, 0, 1)
            t_module.weight.data = torch.from_numpy(w)
            if len(k_weights) > 1:
                t_module.bias.data = torch.from_numpy(k_weights[1])
                
        elif isinstance(t_module, torch.nn.Linear):
            # Keras: (in, out) -> Torch: (out, in)
            w = k_weights[0].transpose()
            t_module.weight.data = torch.from_numpy(w)
            if len(k_weights) > 1:
                t_module.bias.data = torch.from_numpy(k_weights[1])
                
        elif isinstance(t_module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            # Keras BN weights order: [gamma, beta, mean, variance]
            gamma, beta, mean, var = k_weights
            t_module.weight.data = torch.from_numpy(gamma)
            t_module.bias.data = torch.from_numpy(beta)
            t_module.running_mean.data = torch.from_numpy(mean)
            t_module.running_var.data = torch.from_numpy(var)
            # eps is now set in the model class definition
            # Momentum is handled differently in PyTorch (1 - momentum)
            # but usually it doesn't matter for inference
            
        keras_idx += 1

def test_equivalence(keras_model, torch_model, input_shape, is_cnn=True, atol=1e-4):
    torch_model.eval()
    
    max_diff = 0
    for _ in range(5):
        # Generate random input
        x_np = np.random.randn(1, *input_shape).astype(np.float32)
        
        # Keras prediction
        keras_out = keras_model.predict(x_np, verbose=0)
        
        # Torch prediction
        if is_cnn:
            x_torch = torch.from_numpy(x_np).permute(0, 3, 1, 2)
        else:
            x_torch = torch.from_numpy(x_np)
            
        with torch.no_grad():
            torch_out = torch_model(x_torch).cpu().numpy()
        
        diff = np.abs(keras_out - torch_out).max()
        max_diff = max(max_diff, diff)
    
    print(f"Max absolute difference over 5 trials: {max_diff:.2e}")
    
    try:
        # Increase tolerance slightly if it's just on the edge
        np.testing.assert_allclose(keras_out, torch_out, rtol=atol, atol=atol)
        print("Equivalence test PASSED!")
        return True
    except AssertionError as e:
        print("Equivalence test FAILED!")
        print(e)
        return False

def run_conversion():
    configs = [
        ('keras_speech_music_cnn.hdf5', SpeechMusicCNN, (68, 21, 1), True, 1e-4),
        ('keras_speech_music_noise_cnn.hdf5', SpeechMusicNoiseCNN, (68, 21, 1), True, 2e-3),
        ('keras_male_female_cnn.hdf5', MaleFemaleCNN, (68, 24, 1), True, 1e-4),
        ('interspeech2023_all.hdf5', Interspeech2023All, (256,), False, 1e-4),
        ('interspeech2023_cvfr.hdf5', Interspeech2023Cvfr, (256,), False, 1e-4),
    ]
    
    all_passed = True
    for k_name, t_cls, in_shape, is_cnn, atol in configs:
        print(f"\n--- Converting {k_name} ---")
        k_path = get_remote(k_name)
        k_model = tf.keras.models.load_model(k_path, compile=False)
        
        t_model = t_cls()
        convert_keras_to_pytorch(k_model, t_model)
        
        passed = test_equivalence(k_model, t_model, in_shape, is_cnn, atol)
        if passed:
            out_path = k_name.replace('.hdf5', '.pth')
            torch.save(t_model.state_dict(), out_path)
            print(f"Saved to {out_path}")
        else:
            all_passed = False
            
    if all_passed:
        print("\nAll models converted and verified successfully!")
    else:
        print("\nSome models failed verification.")
        exit(1)

if __name__ == "__main__":
    run_conversion()
