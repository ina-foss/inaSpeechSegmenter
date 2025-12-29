
import os
import torch
import tensorflow as tf
import numpy as np
import pytest
from inaSpeechSegmenter.remote_utils import get_remote
from inaSpeechSegmenter.pytorch_models import (
    SpeechMusicCNN, SpeechMusicNoiseCNN, MaleFemaleCNN,
    Interspeech2023All, Interspeech2023Cvfr
)

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@pytest.mark.parametrize("keras_name, torch_cls, input_shape, is_cnn, atol", [
    ('keras_speech_music_cnn.hdf5', SpeechMusicCNN, (68, 21, 1), True, 1e-4),
    ('keras_speech_music_noise_cnn.hdf5', SpeechMusicNoiseCNN, (68, 21, 1), True, 5e-3),
    ('keras_male_female_cnn.hdf5', MaleFemaleCNN, (68, 24, 1), True, 1e-4),
    ('interspeech2023_all.hdf5', Interspeech2023All, (256,), False, 1e-4),
    ('interspeech2023_cvfr.hdf5', Interspeech2023Cvfr, (256,), False, 1e-4),
])
def test_keras_to_torch_equivalence(keras_name, torch_cls, input_shape, is_cnn, atol):
    keras_path = get_remote(keras_name)
    keras_model = tf.keras.models.load_model(keras_path, compile=False)
    
    pth_name = keras_name.replace('.hdf5', '.pth')
    torch_model = torch_cls()
    
    if not os.path.exists(pth_name):
        pytest.skip(f"Weight file {pth_name} not found. Run conversion first.")
        
    torch_model.load_state_dict(torch.load(pth_name, weights_only=True))
    torch_model.eval()
    
    for i in range(3):
        x_np = np.random.randn(1, *input_shape).astype(np.float32)
        
        keras_out = keras_model.predict(x_np, verbose=0)
        
        if is_cnn:
            # Keras (B, H, W, C) -> Torch (B, C, H, W)
            x_torch = torch.from_numpy(x_np).permute(0, 3, 1, 2)
        else:
            x_torch = torch.from_numpy(x_np)
            
        with torch.no_grad():
            torch_out = torch_model(x_torch).cpu().numpy()
        
        np.testing.assert_allclose(keras_out, torch_out, rtol=atol, atol=atol, 
                                   err_msg=f"Failed for {keras_name} on trial {i}")

if __name__ == "__main__":
    # If run directly, just execute the conversion logic if needed or run pytest
    import sys
    pytest.main([__file__])
