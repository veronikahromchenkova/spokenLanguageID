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
    langs_true = {
        0: "en",
        1: "lv",
        2: "ru",
    }
    dataset = []
    label = []
    batch = 1
    class_path = os.listdir(data_path)
    for ind, i in enumerate(class_path):
        temp_path = []
        temp_label = []
        for j in os.listdir(os.path.join(data_path, i)):
            if j.endswith(".wav"):
                temp_path.append(os.path.join(data_path, i, j))
                label.append(ind)
        dataset.extend(temp_path)
        label.extend(temp_label)
    data = Data(dataset, label)
    data = DataLoader(data, batch_size=batch, shuffle=True)
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        for _input, label in data:
            _input = _input.float()
            output = model(_input)
            label = label.squeeze()
            pre = np.array(output.cpu().detach().numpy()).argmax(1)
            predicted_lang = langs_prediction[pre[0]]
            true_lang = langs_true[int(label)]
            print(predicted_lang, true_lang)


# test
test(sys.argv[1], './model100_e20.pt')
