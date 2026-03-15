# About

This repo is for me to convince myself that ML and AI is not magic. It's just a bunch of numbers multiplied by each other. I will attempt to implement most models from scratch only with Numpy. Nothing fancy like

```python
from pytorch import machine_learning as ml

ml.learn()
```

I'll also add a write-up of the things that I learned here as a reference for myself

---

# Models

## K-nearest neighbors

Barely an ML model. Literally just look at the k-nearest neighbors (haha) in the training data and average across them.

K is a hyperparameter to be tuned. Use k-fold CV by splitting the data into subsets, use one for testing and the others for training. Try a bunch of different k values to see which one minimizes loss

## Linear Regression

TODO
