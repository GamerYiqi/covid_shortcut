# Import dependencies
import torch

from models import CXRClassifier
from datasets import PadChestH5Dataset
from datasets import GitHubCOVIDDataset
from datasets import ChestXray14H5Dataset
from datasets import BIMCVCOVIDDataset

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from tqdm import tqdm

from utils import PathExplainerTorch
from utils import monotonically_increasing_red

## obviously you'll want to replace this with the path to your own saved model
model_path = '.\checkpoints\dataset2-0615.densenet121.30493.pkl.best_auroc'

classifier = CXRClassifier()
classifier.load_checkpoint(model_path)

print('Load classifier, model is ' + model_path)

# ds = PadChestH5Dataset(fold='train', labels='ChestX-ray14', initialize_h5=True, random_state=30493)
ds = GitHubCOVIDDataset(fold='train', labels='ChestX-ray14', random_state=30493)

label_odds_probs = np.zeros((ds.__len__(), 3))
for k in tqdm(range(ds.__len__())):

    x = ds[k]

    image = x[0].view(1, 3, 224, 224).cuda()
    image.requires_grad = False

    label = x[1]

    odds = np.exp(classifier.model(image)[0][14].cpu().detach().numpy().item())
    probs = odds / (1 + odds)

    label_odds_probs[k, 0] = label[14]
    label_odds_probs[k, 1] = odds
    label_odds_probs[k, 2] = probs

np.savetxt('label_odds_probs.csv', label_odds_probs, delimiter=',', fmt='%f')
