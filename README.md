# About

This repo is for me to convince myself that ML and AI is not magic. It's just a bunch of numbers multiplied by each other. I will attempt to implement most models from scratch only with Numpy. Nothing fancy like

```python
from pytorch import machine_learning as ml

ml.learn()
```

Admittedly the code will be slow since it's raw Python loops but who cares. I'll also add a write-up of the things that I learned here as a reference for myself

---

# Models

## K-nearest neighbors

Barely an ML model. Literally just look at the k-nearest neighbors (haha) in the training data and average across them.

K is a hyperparameter to be tuned. Use k-fold CV by splitting the data into subsets, use one for testing and the others for training. Try a bunch of different k values to see which one minimizes loss

## Linear Regression

This is genuinely just finding linear coefficients you could ask a high schooler to do. The main assumption is that the output is a linear combination of the input features so you can just write $\hat{y} = w_0 + w_1x_1 + w_2x_2+...+w_nw_n$. Then you just solve a system of equations to find all the $w_i$.

Conveniently, there is a closed form solution to this problem. Since solving the system of equations is equivalent to minimizing SSE, we can take its gradient and set to zero. Note the dimensions
- $X$ is $n \times (d+1)$
- $W$ is $(d+1)\times 1$
- $y$ is $n\times 1$

We have the loss given by
$$
\begin{align}
O & =|| XW-y ||^{2}_{2}  \\
 & =(XW-y)^T(XW-y) \\
 & =(XW)^TXW - (XW)^Ty-y^TXW+y^Ty \\
 & =W^TX^TXW-W^TX^Ty-y^TXW+y^Ty \\
 & =y^Ty -2W^TX^Ty + W^TX^TXW
\end{align}
$$
Now we find the gradient
$$
\frac{\nabla O}{\nabla W} =-2X^Ty+2X^TXW
$$
setting this to zero we get
$$
X^TXW=X^Ty
$$

The solution differs depending on features ($d$) and data ($n$):
- $n\geq d$: We have sufficient or more data to solve the system of equations. More importantly, $X$ has full column rank i.e. $X^TX$ is invertible. We have $W=(X^TX)^{-1}X^Ty$
- $n<d$: We find the solution with lowest norm which gives us $X^T(XX^T)^{-1}Y$

## Logistic Regression


# Concepts

## Regularization
%% TODO %%