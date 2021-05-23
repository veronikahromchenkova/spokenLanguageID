import sys
import torch
from torch.utils.data import DataLoader
from data import Data
import numpy as np
import os

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)


def test(data_path, model_path):
    langs_prediction = {
        0: "lv",
        1: "en",
        2: "ru",
    }
    dataset = []
    batch = 1
    temp_path = []
    for j in os.listdir(data_path):
        if j.endswith(".wav"):
            temp_path.append(os.path.join(data_path, j))
    dataset.extend(temp_path)
    data = Data(dataset)
    data = DataLoader(data, batch_size=batch, shuffle=True)
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        for _input, path in data:
            _input = _input.float()
            output = model(_input)
            pre = np.array(output.cpu().detach().numpy()).argmax(1)
            predicted_lang = langs_prediction[pre[0]]
            print(path, predicted_lang)


# test
test(sys.argv[1], './model100_e20.pt')
