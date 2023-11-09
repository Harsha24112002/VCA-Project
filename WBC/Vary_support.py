import math
import sys
sys.path.append('../UniverSeg')
from example_data.wbc import WBCDataset
import itertools
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import einops as E
import torch
from universeg import universeg
from collections import defaultdict
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--device', '-d', type=int, default=0)
parser.add_argument('--n_support', '-ns',type=int, default=8)
parser.add_argument('--n_ensemble', '-ne',type=int, default=5)
parser.add_argument('--n_repeats', '-nr',type=int, default=100)

args = parser.parse_args()

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

model = universeg(pretrained=True)
_ = model.to(device)

def visualize_tensors(tensors, col_wrap=8, col_names=None, title=None):
    M = len(tensors)
    N = len(next(iter(tensors.values())))

    cols = col_wrap
    rows = math.ceil(N/cols) * M

    d = 2.5
    fig, axes = plt.subplots(rows, cols, figsize=(d*cols, d*rows))
    if rows == 1:
      axes = axes.reshape(1, cols)

    for g, (grp, tensors) in enumerate(tensors.items()):
        for k, tensor in enumerate(tensors):
            col = k % cols
            row = g + M*(k//cols)
            x = tensor.detach().cpu().numpy().squeeze()
            ax = axes[row,col]
            if len(x.shape) == 2:
                ax.imshow(x,vmin=0, vmax=1, cmap='gray')
            else:
                ax.imshow(E.rearrange(x,'C H W -> H W C'))
            if col == 0:
                ax.set_ylabel(grp, fontsize=16)
            if col_names is not None and row == 0:
                ax.set_title(col_names[col])

    for i in range(rows):
        for j in range(cols):
            ax = axes[i,j]
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

    if title:
        plt.suptitle(title, fontsize=20)

    plt.tight_layout()


d_support = WBCDataset('JTSC', split='support', label='cytoplasm')
d_test = WBCDataset('JTSC', split='test', label='cytoplasm')

n_support = 8

support_images, support_labels = zip(*itertools.islice(d_support, n_support))
support_images = torch.stack(support_images).to(device)
support_labels = torch.stack(support_labels).to(device)

# Dice metric for measuring volume agreement
def dice_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred = y_pred.long()
    y_true = y_true.long()
    score = 2*(y_pred*y_true).sum() / (y_pred.sum() + y_true.sum())
    return score.item()

# run inference and compute losses for one test image
@torch.no_grad()
def inference(model, image, label, support_images, support_labels):
    image, label = image.to(device), label.to(device)

    # inference
    logits = model(
        image[None],
        support_images[None],
        support_labels[None]
    )[0] # outputs are logits

    soft_pred = torch.sigmoid(logits)
    hard_pred = soft_pred.round().clip(0,1)

    #  score
    score = dice_score(hard_pred, label)

    # return a dictionary of all relevant variables
    return {'Image': image,
            'Soft Prediction': soft_pred,
            'Prediction': hard_pred,
            'Ground Truth': label,
            'score': score}


# helpful function to sample support data
def sample_support(seed,support_size):
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0,len(d_support), size=support_size)
    support_images, support_labels = zip(*[d_support[i] for i in idxs])
    support_images = torch.stack(support_images).to(device)
    support_labels = torch.stack(support_labels).to(device)
    return support_images, support_labels

# get support data
d_support = WBCDataset('JTSC', split='support', label='cytoplasm')
d_test = WBCDataset('JTSC', split='test', label='cytoplasm')



# setup the number of predictions and ensembling
def inference_vary_se(support_size=1, n_ensemble=1,n_predictions=1,seed=1,vis=False):    
    # get various support sets
    scores = []
    # go through the number of experiments
    for i in (range(n_predictions)):

        # go through the number of predictions we will ensemble
        results = defaultdict(list)
        for j in range(n_ensemble):
            # get support set and query
            support_images, support_labels = sample_support(j+seed,support_size)
            image, label = d_test[i]

            # perform inference
            vals = inference(model, image, label, support_images, support_labels)
            for k, v in vals.items():
                results[k].append(v)

        results['Image'].append(image)
        ensemble = torch.mean(torch.stack(results['Soft Prediction']), dim=0)
        results['Soft Prediction'].append(ensemble)
        results['Prediction'].append(ensemble.round())
        results['Ground Truth'].append(label)
        results['score'].append(dice_score(ensemble.round().clip(0,1), label.to(device)))

        scores.append(results['score'][-1])
    return scores

f = open("../results/vary_support.txt", "w")

def vary_support(support_sizes,n_ensemble,num_repeats=100):
    mean_scores = []
    for support_size in tqdm(support_sizes):
        mean_scores_support = []
        for _ in range(num_repeats):
            outputs = inference_vary_se(support_size, n_ensemble,n_predictions=len(d_test),seed=_)
            mean_scores_support.append(np.mean(outputs))
        mean_scores.append(mean_scores_support) 
        print("Support Size:", support_size, "Mean Score:", np.mean(mean_scores_support), "Std:", np.std(mean_scores_support),file=f)
    return mean_scores

support_sizes = [1, 2, 4, 8, 16, 32, 64]
mean_scores = vary_support(support_sizes, n_ensemble=args.n_ensemble, num_repeats=args.n_repeats)

fig, ax = plt.subplots()
boxplot = ax.boxplot(mean_scores,labels=[str(support_size) for support_size in support_sizes],patch_artist=True)

for box in boxplot['boxes']:
    box.set(facecolor='lightcoral')

for median in boxplot['medians']:
    median.set(color='black')

for flier in boxplot['fliers']:
    flier.set(marker='x')

ax.set_xlabel('Support Size')
ax.set_ylabel('Dice Score')
ax.set_title('Varying Support Size')
plt.savefig('../results/vary_support.png')


