import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
# m1 = np.loadtxt('label_odds_probs_BIMCVCOVIDDataset@ds2_densnet.csv', delimiter=',')
# m2 = np.loadtxt('label_odds_probs_PadChestH5Dataset@ds2_densnet.csv', delimiter=',')

m1 = np.loadtxt('label_odds_probs_ChestXray14H5Dataset@ds2_densnet.csv', delimiter=',')
m2 = np.loadtxt('label_odds_probs_Github@ds2_densnet.csv', delimiter=',')
m = np.vstack((m1, m2))
p = sns.boxenplot(x=m[:, 1], scale='linear')
# sns.scatterplot(x=[0.42819,0.084732,3.3651], y=[0,0,0], sizes=50, color='red') # bmicvcovid
# sns.scatterplot(x=[4.961290331846565,0.18471840237565573,5.88478647804783], y=[0,0,0], sizes=50, color='red') # Github
sns.scatterplot(x=[0.032952,0.042753,2.734058], y=[0,0,0], sizes=50, color='red') # ChestXray
plt.xscale('log')
plt.xlabel('Odds')
plt.show()
