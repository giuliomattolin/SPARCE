import numpy as np
import os
import matplotlib.pyplot as plt
from utils import *
import torch

# set directories
pathToProject = os.getcwd()
dataset = 'motionsense'
fileFormat = '.npy'

num_plots = 20

counterfactuals = np.load(os.path.join(pathToProject, 'counterfactuals', dataset, 'testing_sparce_counterfactuals.npy'))
queries = np.load(os.path.join(pathToProject, 'counterfactuals', dataset, 'testing_sparce_queries.npy'))
targets = np.load(os.path.join(pathToProject, 'counterfactuals', dataset, 'testing_sparce_targets.npy'))

num_samples = counterfactuals.shape[0]
print(f"Plotting {num_plots} samples out of {num_samples}")

# load classifier
classifier = load_model(dataset, 'bidirectional_lstm_classifier')
classifier.eval()

fig, axs = plt.subplots(nrows=4, ncols=num_plots)

for i in range(num_plots):
    # visualise matrices
    axs[0, i].matshow(queries[i], cmap='plasma')
    axs[1, i].matshow(counterfactuals[i], cmap='plasma')
    axs[2, i].matshow(targets[i], cmap='plasma')
    axs[3, i].matshow(counterfactuals[i]-queries[i], cmap='plasma')

    # set as title model prediction
    pred_q = torch.argmax(classifier(torch.unsqueeze(torch.from_numpy(queries[i]), dim=0)))
    pred_c = torch.argmax(classifier(torch.unsqueeze(torch.from_numpy(counterfactuals[i]), dim=0)))
    pred_t = torch.argmax(classifier(torch.unsqueeze(torch.from_numpy(targets[i]), dim=0)))
    axs[0, i].set_title(pred_q.item(), size='small', pad=3)
    axs[1, i].set_title(pred_c.item(), size='small', pad=3)
    axs[2, i].set_title(pred_t.item(), size='small', pad=3)

    # remove ticks
    axs[0, i].set_xticks([]) , axs[0, i].set_yticks([])
    axs[1, i].set_xticks([]) , axs[1, i].set_yticks([])
    axs[2, i].set_xticks([]) , axs[2, i].set_yticks([])
    axs[3, i].set_xticks([]) , axs[3, i].set_yticks([])

for ax, row in zip(axs[:,0], ["Queries", "Counterfactuals", "Targets", "Residuals"]):
    ax.set_ylabel(row, rotation=90, size='small')

fig.savefig(os.path.join(pathToProject, 'figures', dataset, 'testing_sparce.png'))

print("Done!")