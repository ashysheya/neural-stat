# Towards a neural statistician

This repo contains PyTorch implementation of the generative model proposed in [Towards a neural statistician (Edwards and Storkey, ICLR 2017)](https://arxiv.org/pdf/1606.02185.pdf). The implementation contains our replication of all experiments provided in the paper. It also has an extension that allows for generating datasets conditioned on some labels. 

## Synthetic data experiment

Train a model:

```
python train.py --experiment synthetic --lr 1e-3 --num_epochs 50 --context_dim 3
```

Test a model:

```
python test_synthetic.py --model_name path_to_your_model 
```

Test script will save contexts, distribution, means and variances for each sampled dataset as numpy array. 

Our visualisation of these numpy arrays:

![Synthetic experiment](readme_images/synthetic.jpg)

The following image shows 3-D scatter plots of the summary statistics learned. Each point is the mean of the approximate posterior
over the context. Left plot shows points colored by distribution family, center plot colored by the mean and
right plot colored by the variance. The plots have been rotated to illustrative angles.


