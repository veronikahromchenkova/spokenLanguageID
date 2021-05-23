import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2Vec2(nn.Module):
    def __init__(self, n_classes):
        super(Wav2Vec2, self).__init__()
        self.Wav2Vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.Wav2Vec2.feature_extractor._freeze_parameters()

        self.fn1 = nn.Linear(149 * 1024, n_classes)

        initrange = 0.00001
        self.fn1.bias.data.zero_()
        self.fn1.weight.data.uniform_(-initrange, initrange)


    def forward(self, audio_input):
        batch = audio_input.shape[0]
        output = self.Wav2Vec2(audio_input).last_hidden_state
        output = self.fn1(output.reshape(batch, -1))
        output = output.reshape(batch, -1)
        return output
