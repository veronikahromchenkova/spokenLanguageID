import librosa
from torch.utils.data import Dataset
from transformers import Wav2Vec2FeatureExtractor
import numpy as np
import torch


# define Data class from Dataset
class Data(Dataset):
    def __init__(self, path):
        self.path = path
        self.featext = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53", sampling_rate=16000)
    
    def __getitem__(self, index):
        path = self.path[index]
        audio_input, _ = librosa.load(path, sr=16000)
        audio_input = self.featext(audio_input, return_tensors="pt", sampling_rate=16000).input_values
        audio_input = audio_input.reshape(-1)
        if len(audio_input) < 16000*3:
            audio_input_ = torch.Tensor(np.array(list(audio_input) + [0]*(16000*3-len(audio_input))))
        else:
            audio_input_ = audio_input[:16000*3]
        return audio_input_, path

    def __len__(self):
        return len(self.path)




