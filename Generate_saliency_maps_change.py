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
import pandas as pd

from tqdm import tqdm

from utils import PathExplainerTorch
from utils import monotonically_increasing_red

# Configs
model_path = '.\checkpoints\dataset2-0615.densenet121.30493.pkl.best_auroc'
data_name = 'BIMCVCOVID'  # 'PadChest', 'GitHubCOVID', 'ChestXray14', 'BIMCVCOVID'
max_number_images = 20

# Process
classifier = CXRClassifier()
classifier.load_checkpoint(model_path)

print('Load classifier, model is ' + model_path)

pet = PathExplainerTorch(classifier.model.cpu())

# We generate the background references using COVID-19 negative images from the PadChest dataset

try:
    background_ds = torch.load('background_ds.pt')
except FileNotFoundError:
    padchest_train_ds = PadChestH5Dataset(fold='train', labels='ChestX-ray14', random_state=30493)
    background_ds = torch.zeros(200, 3, 224, 224)
    for i, x in enumerate(padchest_train_ds):
        background_ds[i, :, :, :] = x[0]
        if i == 199:
            break
    torch.save(background_ds, 'background_ds.pt')

# Pick one image to use its tag
test_ds = PadChestH5Dataset(fold='test', labels='ChestX-ray14', initialize_h5=True, random_state=30493)
y = test_ds[6]
tagged_pic = y[0]

# Pick image sets
if data_name == 'BIMCVCOVID':
    test_ds = BIMCVCOVIDDataset(fold='test', labels='ChestX-ray14', initialize_h5=True, random_state=30493)
elif data_name == 'PadChest':
    test_ds = PadChestH5Dataset(fold='test', labels='ChestX-ray14', initialize_h5=True, random_state=30493)
elif data_name == 'GitHubCOVID':
    test_ds = GitHubCOVIDDataset(fold='test', labels='ChestX-ray14', random_state=30493)
elif data_name == 'ChestXray14':
    test_ds = ChestXray14H5Dataset(fold='test', labels='ChestX-ray14', initialize_h5=True, random_state=30493)

obb_prob_matrix = np.zeros((max_number_images, 7))
n = 0
for k in tqdm(range(test_ds.__len__())):
    obb_prob_matrix[n, 0] = k

    for j in range(3):
        x = test_ds[k]
        org_pic = x[0]
        if j == 1:
            org_pic[:, 6:22, 6:17] = tagged_pic[:, 6:22, 6:17]
        elif j == 2:
            org_pic[:, 106:122, 6:17] = tagged_pic[:, 6:22, 6:17]

        example = org_pic.view(1, 3, 224, 224)
        example.requires_grad = True

        label = x[1]

        odds = np.exp(classifier.model(example)[0][14].detach().numpy().item())
        probs = odds / (1 + odds)
        obb_prob_matrix[n, j*2+1] = odds
        obb_prob_matrix[n, j*2+2] = probs

        output_attribs = pet.attributions(example, background_ds,
                                          output_indices=torch.tensor([14]),
                                          use_expectation=True,
                                          num_samples=50)
        ##
        ## for visualization purposes, we take the mean absolute EG values
        ## (variable named "shaps" here because EG calculates an Aumann-Shapley value)
        ##
        ma_shaps = output_attribs.abs().mean(0).mean(0).detach().numpy()

        sb.set_style("white")
        fig, (showcxr, heatmap) = plt.subplots(ncols=2, figsize=(14, 5))
        hmap = sb.heatmap(ma_shaps,
                          cmap=monotonically_increasing_red(),
                          linewidths=0,
                          zorder=2,
                          vmax=np.percentile(ma_shaps.flatten(), 99.5)  # we clip attributions at 99.5th percentile
                          )  # to fix Coverage (see http://ceur-ws.org/Vol-2327/IUI19WS-ExSS2019-16.pdf)
        cxr = example.detach().numpy().squeeze().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        cxr = std * cxr + mean
        cxr = np.clip(cxr, 0, 1)

        hmap.axis('off')

        showcxr.imshow(cxr)
        showcxr.axis('off')
        fig.suptitle('EG Attributions, True Label: {:d}, Pred: {:0.4f}'.format(label[14], probs), fontsize=16)
        result_tuple = (cxr, ma_shaps, label, odds, probs)
        torch.save(result_tuple, 'Result_cxr_ma_label_probs_testnum_{:d}_change{:d}.pt'.format(k, j))
        plt.savefig('Saliency_maps_testnum_{:d}_change{:d}_ChestXray14Dataset.png'.format(k, j))
        plt.close(fig)

        n += 1
        if n >= max_number_images:
            break

df = pd.DataFrame(obb_prob_matrix,
                   columns=['PicNum', 'Obbs_NoChange', 'Probs_NoChange', 'Obbs_UpleftR', 'Probs_UpleftR', 'Obbs_MiddleleftR', 'Probs_MiddleleftR'])
df.to_csv('obb_prob_matrix.csv', index=False)
