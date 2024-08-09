# It is Never Too Late to Mend: Separate Learning for Multimedia Recommendation

The code and datasets of our paper "It is Never Too Late to Mend: Separate Learning for Multimedia Recommendation"

# Overview

Multimedia recommendation, which incorporates various modalities (e.g., images, texts, etc.) into user or item representation to improve recommendation quality, and self-supervised learning carries multimedia recommendation to a plateau of performance, because of its superior performance in aligning different modalities. However, more and more research finds that aligning all modal representations is suboptimal because it damages the attributes unique to each modal. These studies use subtraction and orthogonal constraints in geometric space to learn unique parts. However, our rigorous analysis reveals the flaws in this approach, such as that subtraction does not necessarily yield the desired modal-unique and that orthogonal constraints are ineffective in user and item high-dimensional representation spaces. To make up for the previous weaknesses, we propose Separate Learning (SEA) for multimedia recommendation, which mainly includes mutual information view of modal-unique and -generic Learning. Specifically, we first use GNN to learn the representations of users and items in different modalities and split each modal rep- resentation into generic and unique parts. We employ contrastive log-ratio upper bound to minimize the mutual information between the general and unique parts within the same modality, to distance their representations, thus learning modal-unique features. Then, we design Solosimloss to maximize the lower bound of mutual information, to align the general parts of different modalities, thus learning more high-quality modal-generic features. Finally, extensive experiments on three datasets demonstrate the effectiveness and generalization of our proposed framework. The code is available at SEA and the full training record of the main experiment

# Requirements

The model is implemented using PyTorch. The versions of packages used are shown below.

- numpy==1.18.0
- scikit-learn==0.22.1
- torch==1.6.1

# Data Preparation

Corresponding to MMRec.

# Special Thanks

Special thanks to [MMRec](https://github.com/enoche/MMRec)

# Quick run

```js
python main.py
```

# Open Source Log

We open the main experimental log