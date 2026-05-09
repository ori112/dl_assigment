# Deep Learning Assignment 1 — Complete Study Guide
### Image Classification: SVM, Softmax, Two-Layer Neural Network on CIFAR-10

---

## TABLE OF CONTENTS

1. [The Big Picture](#1-the-big-picture)
2. [CIFAR-10 Dataset](#2-cifar-10-dataset)
3. [Linear Classifiers — Core Concepts](#3-linear-classifiers--core-concepts)
4. [Multiclass SVM (Hinge Loss)](#4-multiclass-svm-hinge-loss)
5. [Softmax Classifier (Cross-Entropy Loss)](#5-softmax-classifier-cross-entropy-loss)
6. [Numeric Stability — The Log-Sum-Exp Trick](#6-numeric-stability--the-log-sum-exp-trick)
7. [Gradient Descent and SGD Training](#7-gradient-descent-and-sgd-training)
8. [Regularization](#8-regularization)
9. [Hyperparameter Search](#9-hyperparameter-search)
10. [Two-Layer Neural Network](#10-two-layer-neural-network)
11. [Backpropagation — Full Derivation](#11-backpropagation--full-derivation)
12. [Gradient Checking](#12-gradient-checking)
13. [Comparing the Three Classifiers](#13-comparing-the-three-classifiers)
14. [Oral Exam Q&A](#14-oral-exam-qa)

---

## 1. The Big Picture

### What is Image Classification?
Given an image (a grid of pixels), assign it one label from a fixed set of categories.
- **Input:** raw pixel values → a vector
- **Output:** a single label (e.g., "cat", "plane", "truck")

### Why is it hard?
- Viewpoint variation (same cat from different angles)
- Illumination changes
- Deformation (cats curl up)
- Occlusion (cat behind a sofa)
- Background clutter
- Intra-class variation (a chihuahua and a great dane are both "dogs")

### Approach: parametric classifiers
Instead of memorizing every image, learn a function `f(X, W)` that maps pixels → class scores using a weight matrix `W`. Once trained, `W` encodes all knowledge — we discard the training data at test time.

```
Image X  →  score function f(X, W)  →  class scores  →  highest score = prediction
```

---

## 2. CIFAR-10 Dataset

- **60,000** color images total: 50,000 train + 10,000 test
- **10 classes**: plane, car, bird, cat, deer, dog, frog, horse, ship, truck
- Each image: **32 × 32 pixels × 3 color channels (RGB)** = **3,072 numbers** per image

### Preprocessing pipeline (done in `eecs598/data.py`)
1. Load images as float32 tensors scaled to [0, 1]
2. Flatten to shape `(N, 3072)` — each row is one image as a vector
3. Subtract the **mean image** (zero-center the data): this improves gradient flow
4. Split training set → train (40,000) + validation (10,000)
5. Test set: 10,000 (touched only at the very end)

```python
# After preprocessing:
X_train.shape == (40000, 3072)   # N_train x D
X_val.shape   == (10000, 3072)   # N_val x D
X_test.shape  == (10000, 3072)   # N_test x D
y_train.shape == (40000,)        # integer labels in [0, 9]
```

### The bias trick (optional)
Instead of tracking `b` separately, append a column of 1s to X:

```
[x1, x2, ..., xD]  →  [x1, x2, ..., xD, 1]    shape: D+1
W shape: (D, C)    →   W shape: (D+1, C)
```
Then `X @ W` automatically computes `X_pixels @ W_pixels + b`. Cleaner but our code keeps `W` and `b` separate.

---

## 3. Linear Classifiers — Core Concepts

### Score function
```
f(X, W) = X @ W        shape: (N, D) @ (D, C) = (N, C)
```
- `N` = batch size (number of images)
- `D` = 3072 (pixel dimension)
- `C` = 10 (number of classes)
- `scores[i, c]` = raw score (logit) for image `i` belonging to class `c`

The weight matrix `W` has shape `(D, C)`. Each column `W[:, c]` is a "template" for class `c` — dot product with an image measures similarity to that template.

### What makes a good W?
A good `W` assigns high scores to the correct class and low scores to all others. We need a **loss function** to measure "how bad is the current W?" and **gradient descent** to improve it.

---

## 4. Multiclass SVM (Hinge Loss)

### Intuition
"The correct class should score higher than every wrong class by at least a margin Δ = 1."

If image `i` has correct class `y_i`, the SVM loss for that image is:

```
L_i = Σ_{j ≠ y_i}  max(0,  s_j - s_{y_i} + Δ)
```
Where `s_j = scores[i, j]` and `Δ = 1` (the safety margin).

- If `s_j < s_{y_i} - 1` for all wrong classes → no loss (already safe enough)
- If any wrong class `j` gets a score within the margin → that term contributes to the loss

### Full SVM loss (over the batch)
```
L = (1/N) * Σ_i L_i  +  reg * Σ_{k,l} W_{k,l}^2
```
The second term is L2 regularization (explained in Section 8).

### Worked example (single image, 3 classes)
```
scores  = [3.2,  5.1,  -1.7]      (for classes 0, 1, 2)
y_i = 0   (correct class is 0, score = 3.2)

margin for class 1: max(0, 5.1 - 3.2 + 1) = max(0, 2.9) = 2.9  ← violates margin
margin for class 2: max(0, -1.7 - 3.2 + 1) = max(0, -3.9) = 0  ← no loss

L_i = 2.9 + 0 = 2.9
```

### SVM gradient derivation

We need `dL/dW`, i.e., how much changing each weight changes the loss.

**For the wrong class column `j` (when margin > 0):**
```
∂L_i/∂W[:,j] = X[i]          (because s_j = X[i] · W[:,j], so d(s_j)/d(W[:,j]) = X[i])
```

**For the correct class column `y_i`:**
```
∂L_i/∂W[:,y_i] = -1 * (number of classes j where margin > 0) * X[i]
```
Because every violating class j contributes a `-s_{y_i}` term, and `d(s_{y_i})/d(W[:,y_i]) = X[i]`.

**Regularization gradient:**
```
∂/∂W (reg * sum(W^2)) = 2 * reg * W
```
Note: NO 1/2 coefficient in front of reg, so the gradient has the 2× factor.

### Naive implementation (loops)

```python
def svm_loss_naive(W, X, y, reg):
    dW = torch.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = W.t().mv(X[i])              # shape (C,)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1   # delta = 1
            if margin > 0:
                loss += margin
                # Gradient update (the key part):
                dW[:, j] += X[i]             # wrong class gets +X[i]
                dW[:, y[i]] -= X[i]          # correct class gets -X[i]

    loss /= num_train
    loss += reg * torch.sum(W * W)

    dW /= num_train                          # average gradient
    dW += 2 * reg * W                        # regularization gradient

    return loss, dW
```

### Vectorized implementation (no loops)

Key insight: replace the nested loop with matrix operations.

```python
def svm_loss_vectorized(W, X, y, reg):
    N = X.shape[0]
    dW = torch.zeros_like(W)

    # --- LOSS ---
    all_scores = X.mm(W)                                  # (N, C)
    correct_scores = all_scores[torch.arange(N), y]       # (N,)  correct class scores
    correct_scores = correct_scores.unsqueeze(1)          # (N, 1) for broadcasting

    margins = torch.clamp(all_scores - correct_scores + 1, min=0)  # (N, C)
    margins[torch.arange(N), y] = 0.0                    # zero out correct class

    loss = margins.sum() / N + reg * torch.sum(W * W)

    # --- GRADIENT ---
    # Binary mask: 1 where margin > 0
    margin_mask = (margins > 0).float()                   # (N, C)
    # Correct class column: -count of violating classes for that example
    margin_mask[torch.arange(N), y] = -margin_mask.sum(dim=1)

    dW = X.t().mm(margin_mask) / N + 2 * reg * W        # (D, C)

    return loss, dW
```

**Why vectorized?** Instead of a Python loop over N examples (slow), we use matrix multiply which runs in C/CUDA. Expected speedup: 15–120×.

### Intuition for the vectorized gradient
`X.t().mm(margin_mask)` computes:
```
dW[d, c] = Σ_i  X[i, d] * mask[i, c]
```
Which is exactly what the loop computes: for each (d, c) pair, sum X[i, d] over all images i that contributed to the gradient for class c.

---

## 5. Softmax Classifier (Cross-Entropy Loss)

### Intuition
Instead of a margin-based loss, interpret the scores as **unnormalized log-probabilities** and use a probabilistic framework.

### Converting scores to probabilities
Apply the **softmax function**:

```
P(class = c | image i)  =  exp(s_c) / Σ_k exp(s_k)
```

This turns any vector of real numbers into a valid probability distribution (all positive, sum to 1).

### Cross-entropy loss
We want the model to assign high probability to the correct class. The loss is the negative log-probability of the correct class:

```
L_i = -log( P(y_i) )  =  -log( exp(s_{y_i}) / Σ_k exp(s_k) )
       =  -s_{y_i}  +  log( Σ_k exp(s_k) )
```

### Full softmax loss
```
L = (1/N) * Σ_i L_i  +  reg * Σ_{k,l} W_{k,l}^2
```

### Worked example (single image, 3 classes)
```
scores  = [3.2,  5.1,  -1.7]
y_i = 0   (correct class)

exp(3.2)  = 24.53
exp(5.1)  = 164.02
exp(-1.7) = 0.18
sum       = 188.73

P(class 0) = 24.53 / 188.73 = 0.130   ← correct class
P(class 1) = 164.02 / 188.73 = 0.869
P(class 2) = 0.18 / 188.73  = 0.001

L_i = -log(0.130) = 2.04
```
A perfect classifier would get L_i ≈ 0. A random classifier on 10 classes gets L_i ≈ log(10) ≈ 2.3.

### Softmax gradient derivation

Let `p_c = exp(s_c) / Σ_k exp(s_k)` be the softmax probabilities.

**Gradient with respect to scores:**
```
dL_i/ds_c  =  p_c - 1(c == y_i)
```
Where `1(c == y_i)` is 1 for the correct class, 0 otherwise.

In English: subtract 1 from the correct class probability, leave others unchanged.

**Gradient with respect to W (via chain rule):**
Since `s_c = X[i] · W[:, c]`, we have `ds_c/dW[:, c] = X[i]`, so:
```
dL_i/dW[:, c]  =  (p_c - 1(c == y_i)) * X[i]
```

### Naive implementation

```python
def softmax_loss_naive(W, X, y, reg):
    loss = 0.0
    dW = torch.zeros_like(W)
    num_train = X.shape[0]

    for i in range(num_train):
        scores = W.t().mv(X[i])         # (C,)
        # Numeric stability: subtract max (see Section 6)
        scores -= scores.max()
        exp_scores = torch.exp(scores)
        probs = exp_scores / exp_scores.sum()   # softmax probabilities

        # Cross-entropy loss
        loss -= torch.log(probs[y[i]])

        # Gradient: (probs - one_hot) * X[i]
        probs[y[i]] -= 1.0              # subtract 1 from correct class
        dW += X[i].unsqueeze(1) * probs.unsqueeze(0)   # outer product (D, C)

    loss /= num_train
    loss += reg * torch.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W

    return loss, dW
```

### Vectorized implementation

```python
def softmax_loss_vectorized(W, X, y, reg):
    N = X.shape[0]
    loss = 0.0
    dW = torch.zeros_like(W)

    scores = X.mm(W)                                      # (N, C)
    scores -= scores.max(dim=1, keepdim=True).values      # stability
    exp_scores = torch.exp(scores)
    probs = exp_scores / exp_scores.sum(dim=1, keepdim=True)  # (N, C)

    # Loss: -log(correct class probability), averaged
    correct_log_probs = -torch.log(probs[torch.arange(N), y])
    loss = correct_log_probs.sum() / N + reg * torch.sum(W * W)

    # Gradient: subtract 1 from correct class, then X^T @ probs / N
    probs[torch.arange(N), y] -= 1.0
    dW = X.t().mm(probs) / N + 2 * reg * W              # (D, C)

    return loss, dW
```

### SVM vs Softmax: key differences

| Property | SVM (Hinge) | Softmax (Cross-entropy) |
|----------|-------------|------------------------|
| Output | Margin scores | Probabilities |
| Loss when correct | 0 (if margin satisfied) | Always > 0 |
| Cares about magnitude | No (only relative) | Yes (wants confidence) |
| Sanity check loss | ~9.0 (10 wrong classes × 1.0 margin) | ~2.3 = log(10) |
| Typical val accuracy | ~37% | ~37% |

In practice, both give similar accuracy. Softmax is more commonly used in modern deep learning.

---

## 6. Numeric Stability — The Log-Sum-Exp Trick

### The problem
When scores are large (e.g., 1000), `exp(1000)` overflows to `inf` in float32.
When scores are very negative, `exp(-1000)` underflows to 0, and `log(0) = -inf`.

### The trick
Subtracting a constant from all scores before taking exp doesn't change the softmax:
```
exp(s_c - M) / Σ_k exp(s_k - M)  =  exp(s_c) / Σ_k exp(s_k)
```
Because `exp(s_c - M) = exp(s_c) * exp(-M)`, and the `exp(-M)` factor cancels in the fraction.

Standard choice: `M = max(s_k)` over the current example.

```python
scores -= scores.max()          # now max score = 0, others are negative
exp_scores = torch.exp(scores)  # all values in (0, 1]
```

### Why this is critical
Without this trick, any image with slightly high raw scores will produce `nan` loss, and your gradient will also be `nan`. The entire training diverges. **Always subtract the max.**

---

## 7. Gradient Descent and SGD Training

### Why gradient descent?
The loss `L(W)` is a surface in a high-dimensional space. We want to find the `W` that minimizes it. The gradient `∇L(W)` tells us the direction of steepest ascent — so we step in the opposite direction.

```
W_new = W_old - learning_rate * ∇L(W_old)
```

### Full gradient descent vs. SGD
- **Full GD:** compute loss over all N=40,000 training examples every step → very slow
- **SGD (Stochastic):** compute loss over a small random **minibatch** of 200 examples → fast, noisy but usually converges to a good solution

### Minibatch sampling

```python
def sample_batch(X, y, num_train, batch_size):
    random_indices = torch.randint(0, num_train, (batch_size,))
    X_batch = X[random_indices]       # shape (batch_size, D)
    y_batch = y[random_indices]       # shape (batch_size,)
    return X_batch, y_batch
```

### Full training loop

```python
def train_linear_classifier(loss_func, W, X, y, learning_rate=1e-3, reg=1e-5,
                              num_iters=100, batch_size=200, verbose=False):
    num_train, dim = X.shape
    if W is None:
        num_classes = torch.max(y) + 1
        W = 0.000001 * torch.randn(dim, num_classes, device=X.device, dtype=X.dtype)

    loss_history = []
    for it in range(num_iters):
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)
        loss, grad = loss_func(W, X_batch, y_batch, reg)
        loss_history.append(loss.item())

        W -= learning_rate * grad    # SGD update

        if verbose and it % 100 == 0:
            print(f'iteration {it}/{num_iters}: loss {loss:.4f}')

    return W, loss_history
```

### What is `learning_rate`?
- Too large: loss oscillates or diverges (overshooting the minimum)
- Too small: training is very slow
- Typical values for linear classifier: `1e-3` to `1e-2`

### Learning rate decay (used in neural network training)
Multiply `learning_rate *= decay_factor` (e.g., 0.95) after every epoch. This allows faster learning early, then finer adjustments as training progresses.

```
epoch 1: lr = 1e-3
epoch 2: lr = 9.5e-4
epoch 3: lr = 9.025e-4
...
```

---

## 8. Regularization

### Why regularize?
Without regularization, the model can overfit: it memorizes training data but performs poorly on unseen data. Regularization penalizes large weights, forcing the model to learn simpler, more generalizable patterns.

### L2 regularization (weight decay)
```
L_total = L_data + reg * Σ_{k,l} W_{k,l}^2
```
Also written as `reg * ||W||_F^2` (Frobenius norm squared).

- Prefers small, diffuse weights over large, concentrated ones
- When multiple weight vectors explain the data equally well, L2 picks the one with smallest magnitude

### Gradient contribution
```
∂/∂W (reg * sum(W^2)) = 2 * reg * W
```
**Important:** We do NOT use a 1/2 coefficient (the assignment is explicit about this), so the gradient gets the full `2 * reg` factor.

### Effect of reg strength
- High reg (`1e-1`): weights stay small → underfitting, lower accuracy
- Low reg (`1e-6`): weights can grow → potential overfitting
- Typical best values found via validation: `1e-4` to `1e-2`

---

## 9. Hyperparameter Search

### What are hyperparameters?
Values that control the training process but are NOT learned by gradient descent:
- `learning_rate`: step size in SGD
- `reg`: regularization strength
- (for NN) `hidden_size`, `learning_rate_decay`

### Grid search
Try every combination of candidate values, train each model, pick the one with highest **validation accuracy**.

```python
def svm_get_search_params():
    learning_rates = [1e-2, 5e-3, 1e-3]           # 3 candidates
    regularization_strengths = [1e-4, 1e-3, 1e-2, 5e-2]   # 4 candidates
    # Total: 3 × 4 = 12 combinations (< 25 limit)
    return learning_rates, regularization_strengths
```

### test_one_param_set — training and evaluation

```python
def test_one_param_set(cls, data_dict, lr, reg, num_iters=2000):
    cls.train(data_dict['X_train'], data_dict['y_train'],
              learning_rate=lr, reg=reg, num_iters=num_iters)

    train_preds = cls.predict(data_dict['X_train'])
    train_acc = (train_preds == data_dict['y_train']).float().mean().item()

    val_preds = cls.predict(data_dict['X_val'])
    val_acc = (val_preds == data_dict['y_val']).float().mean().item()

    return cls, train_acc, val_acc
```

### predict function
```python
def predict_linear_classifier(W, X):
    y_pred = X.mm(W).argmax(dim=1)
    return y_pred
```
Multiply input by weights → class scores → pick the index of the highest score.

### Validation vs. test set
- **Validation set:** used during development to select hyperparameters
- **Test set:** used ONCE at the end to report final accuracy
- Never tune hyperparameters based on test accuracy — that would leak test information and give an overly optimistic estimate

---

## 10. Two-Layer Neural Network

### Why go beyond linear?
A linear classifier draws a single hyperplane in the feature space. Many real boundaries are curved or non-linear. A neural network can learn non-linear boundaries by composing simple functions.

### Architecture
```
Input X (N, D)
    ↓ W1 (D, H) + b1 (H,)
FC Layer 1: h_raw = X @ W1 + b1        shape: (N, H)
    ↓ ReLU
Hidden Layer: h = max(0, h_raw)         shape: (N, H)   [non-linearity!]
    ↓ W2 (H, C) + b2 (C,)
FC Layer 2: scores = h @ W2 + b2       shape: (N, C)
    ↓
Softmax + Cross-entropy loss
```

Where:
- `D` = 3072 (input dimension)
- `H` = hidden layer size (hyperparameter, e.g. 128)
- `C` = 10 (number of classes)

### ReLU (Rectified Linear Unit)
```
ReLU(x) = max(0, x)
```
- If x > 0: passes the value through unchanged
- If x ≤ 0: outputs 0 (the neuron is "dead" for that input)

```python
# In code (no torch.relu allowed in this assignment):
hidden = (X.mm(W1) + b1).clamp(min=0)
```

**Why ReLU?**
- Computationally cheap (just a threshold)
- Gradient is 0 or 1 (no vanishing gradient for positive values)
- Introduces the non-linearity needed to learn complex functions

### Parameters
```python
params = {
    'W1': shape (D, H)   # D = 3072, H = 128
    'b1': shape (H,)
    'W2': shape (H, C)   # H = 128, C = 10
    'b2': shape (C,)
}
```
Total parameters for H=128: `3072×128 + 128 + 128×10 + 10` = **394,634 parameters**.

### Forward pass

```python
def nn_forward_pass(params, X):
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']

    # First FC layer + ReLU activation
    hidden = (X.mm(W1) + b1).clamp(min=0)   # shape: (N, H)
    # Second FC layer gives class scores
    scores = hidden.mm(W2) + b2              # shape: (N, C)

    return scores, hidden
```

---

## 11. Backpropagation — Full Derivation

Backprop applies the chain rule repeatedly to compute how the loss changes with respect to every parameter. We go layer by layer from the loss back to the inputs.

### Forward pass recap (with notation)
```
h_raw = X @ W1 + b1          # pre-activation, shape (N, H)
h     = max(0, h_raw)         # post-ReLU (hidden), shape (N, H)
s     = h @ W2 + b2           # scores, shape (N, C)
p     = softmax(s)            # probabilities, shape (N, C)
L     = cross_entropy(p, y)  + reg_term
```

### Step 1: Gradient of loss w.r.t. scores
Softmax + cross-entropy has a beautiful combined gradient:
```
ds = p                          # start with softmax probs
ds[i, y[i]] -= 1               # subtract 1 from correct class
ds /= N                         # average over batch
```

In code:
```python
d_scores = probs.clone()
d_scores[torch.arange(N), y] -= 1.0
d_scores /= N          # shape: (N, C)
```

### Step 2: Gradient w.r.t. W2 and b2
Since `scores = h @ W2 + b2`:
```
dL/dW2 = h^T @ d_scores + 2 * reg * W2
dL/db2 = d_scores.sum(axis=0)
```
Chain rule: `dL/dW2 = (dL/ds) * (ds/dW2)`. Since `ds/dW2 = h^T`, we get the matmul.

In code:
```python
grads['W2'] = h1.t().mm(d_scores) + 2 * reg * W2    # (H, C)
grads['b2'] = d_scores.sum(dim=0)                    # (C,)
```

### Step 3: Gradient w.r.t. h (backprop through second FC layer)
```
dL/dh = d_scores @ W2^T     shape: (N, H)
```
In code:
```python
d_hidden = d_scores.mm(W2.t())    # (N, H)
```

### Step 4: Backprop through ReLU
ReLU is `max(0, x)`. Its derivative is:
- 1 if the input was > 0 (gate was open)
- 0 if the input was ≤ 0 (gate was shut — no gradient flows back)

```
dL/dh_raw[i, j] = dL/dh[i, j]  if  h[i, j] > 0
                = 0              if  h[i, j] <= 0
```

In code:
```python
d_hidden[h1 <= 0] = 0.0      # zero out gradient where ReLU was inactive
# (h1 is the post-ReLU hidden, so h1 <= 0 exactly where h_raw <= 0)
```

### Step 5: Gradient w.r.t. W1 and b1
Since `h_raw = X @ W1 + b1`:
```
dL/dW1 = X^T @ d_hidden_after_relu + 2 * reg * W1
dL/db1 = d_hidden_after_relu.sum(axis=0)
```

In code:
```python
grads['W1'] = X.t().mm(d_hidden) + 2 * reg * W1    # (D, H)
grads['b1'] = d_hidden.sum(dim=0)                   # (H,)
```

### Complete backward pass in one place

```python
# Assume we already have: scores, h1 (post-ReLU hidden), probs, N, W1, W2, reg

# 1. Gradient of softmax+cross-entropy w.r.t. scores
d_scores = probs.clone()
d_scores[torch.arange(N), y] -= 1.0
d_scores /= N                                         # (N, C)

# 2. Second FC layer gradients
grads['W2'] = h1.t().mm(d_scores) + 2 * reg * W2    # (H, C)
grads['b2'] = d_scores.sum(dim=0)                    # (C,)

# 3. Backprop into hidden layer
d_hidden = d_scores.mm(W2.t())                       # (N, H)

# 4. Backprop through ReLU
d_hidden[h1 <= 0] = 0.0

# 5. First FC layer gradients
grads['W1'] = X.t().mm(d_hidden) + 2 * reg * W1     # (D, H)
grads['b1'] = d_hidden.sum(dim=0)                    # (H,)
```

### Chain rule — verbal explanation
Think of the network as a pipeline of functions:
```
Loss ← softmax ← FC2 ← ReLU ← FC1 ← X
```
Backprop works right-to-left. At each step we ask: "given how much the output of this layer affects the loss, how much does the input to this layer affect the loss?" The answer is: multiply by the local derivative.

- FC layer: local derivative is the other matrix (transpose)
- ReLU: local derivative is 1 or 0 (the "gate")
- Softmax + cross-entropy: local derivative is `probs - one_hot`

### SGD update for neural network

```python
def nn_train(params, loss_func, pred_func, X, y, X_val, y_val,
             learning_rate, learning_rate_decay, reg, num_iters, batch_size, verbose):
    ...
    for it in range(num_iters):
        X_batch, y_batch = sample_batch(X, y, num_train, batch_size)
        loss, grads = loss_func(params, X_batch, y=y_batch, reg=reg)

        # SGD: update every parameter
        for param_name in params:
            params[param_name] -= learning_rate * grads[param_name]

        # Learning rate decay at the end of each epoch
        if it % iterations_per_epoch == 0:
            learning_rate *= learning_rate_decay
```

### Neural network hyperparameter search

Four dimensions to search (must stay under 256 total combos):
```python
def nn_get_search_params():
    learning_rates = [1e-3, 5e-4, 1e-4]          # 3 values
    hidden_sizes   = [128, 256]                    # 2 values
    regularization_strengths = [1e-4, 1e-3, 1e-2] # 3 values
    learning_rate_decays = [1.0, 0.95]             # 2 values
    # Total: 3 × 2 × 3 × 2 = 36 combinations ✓

    return learning_rates, hidden_sizes, regularization_strengths, learning_rate_decays
```

### find_best_net — grid search

```python
def find_best_net(data_dict, get_param_set_fn):
    best_net = None
    best_stat = None
    best_val_acc = 0.0

    lrs, hidden_sizes, regs, lr_decays = get_param_set_fn()
    input_size = data_dict['X_train'].shape[1]      # 3072
    num_classes = int(data_dict['y_train'].max().item()) + 1  # 10
    device = str(data_dict['X_train'].device)        # 'cuda' or 'cpu'

    for lr in lrs:
        for hs in hidden_sizes:
            for reg in regs:
                for lr_decay in lr_decays:
                    net = TwoLayerNet(input_size, hs, num_classes, device=device)
                    stat = net.train(
                        data_dict['X_train'], data_dict['y_train'],
                        data_dict['X_val'],   data_dict['y_val'],
                        learning_rate=lr, learning_rate_decay=lr_decay,
                        reg=reg, num_iters=1000, batch_size=200, verbose=False
                    )
                    val_preds = net.predict(data_dict['X_val'])
                    val_acc = (val_preds == data_dict['y_val']).float().mean().item()

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_net = net
                        best_stat = stat

    return best_net, best_stat, best_val_acc
```

---

## 12. Gradient Checking

### What is it?
A technique to verify that your analytically-computed gradient is correct by comparing it to a numerically-estimated gradient.

### Numeric gradient (finite differences)
```
df/dx ≈ (f(x + h) - f(x - h)) / (2h)      h = 1e-7
```
This is called the **centered difference** approximation. It is accurate to O(h²).

### Relative error
```
rel_error = |analytical - numeric| / max(|analytical|, |numeric|)
```
Expected tolerances:
- Linear classifier (SVM, Softmax): `rel_error < 1e-5`
- Neural network: `rel_error < 1e-4`

### What if gradient check fails?
Common causes:
1. Wrong sign (forgot to negate somewhere)
2. Missing division by N (batch averaging)
3. Missing regularization gradient term
4. Wrong dimension (e.g., transposed a matrix incorrectly)
5. In ReLU backprop: not zeroing gradient where activation ≤ 0
6. Numeric instability in softmax (forgot log-sum-exp trick)

---

## 13. Comparing the Three Classifiers

### Linear SVM
- **Pros:** Simple, robust, clear geometric interpretation
- **Cons:** Linear boundary only, doesn't produce probabilities
- **Target accuracy:** val ≥ 37%, test ≥ 35%

### Softmax
- **Pros:** Produces calibrated probabilities, smooth loss function (gradient always non-zero)
- **Cons:** Still linear boundary
- **Target accuracy:** val ≥ 37%, test ≥ 36%

### Two-Layer Neural Network
- **Pros:** Non-linear boundary (can model curves), much higher capacity
- **Cons:** More hyperparameters, longer to train, harder to interpret
- **Target accuracy:** val ≥ 50%, test ≥ 50%

### Why does the NN do ~13% better?
The hidden layer with ReLU allows the network to learn non-linear feature combinations. For example, it might learn that "fur + pointy ears + whiskers" → cat, even if each feature alone doesn't distinguish categories well.

### Decision boundary comparison
- Linear classifier: flat hyperplane (like drawing a straight line between classes)
- Two-layer NN: can draw curved, complex boundaries using combinations of linear pieces from the ReLU units

---

## 14. Oral Exam Q&A

**Q: What is the SVM loss and why is the margin Δ=1?**
A: The SVM loss penalizes cases where a wrong class score comes within 1 of the correct class score. Δ=1 is a convention — the actual value doesn't matter much because `W` can scale to absorb any Δ. What matters is the relative gap between correct and wrong class scores.

**Q: What happens to the SVM loss if we scale W by 2?**
A: All scores double, so all margins double, and the loss increases. This is why regularization is important — it prevents the model from just scaling W to reduce the hinge loss without learning anything.

**Q: Why does softmax loss equal log(C) ≈ 2.3 at random initialization (10 classes)?**
A: With small random weights, all class scores are nearly equal, so each class gets probability ≈ 1/10. Then `L = -log(1/10) = log(10) ≈ 2.3`. This is a sanity check: if your loss doesn't start near 2.3, something is wrong with initialization.

**Q: What is the vanishing gradient problem and how does ReLU help?**
A: In deep networks, gradients get multiplied at each layer during backprop. Sigmoid activations saturate (output ≈ 0 or 1), causing gradients to shrink exponentially → weights in early layers barely update. ReLU has gradient exactly 1 for positive inputs, so gradients flow freely through "open" neurons.

**Q: What is overfitting and how does regularization prevent it?**
A: Overfitting means the model memorizes training data (high train accuracy) but doesn't generalize (low val/test accuracy). Regularization penalizes large weights — large weights allow the model to memorize specific training examples via sharp decision boundaries. By penalizing `||W||²`, we force smoother, simpler boundaries.

**Q: What is the purpose of the validation set?**
A: We use validation accuracy to choose hyperparameters (learning_rate, reg, hidden_size). If we used test accuracy for this, we'd be optimizing for the test set, and our final test accuracy would be an overestimate of true generalization. The test set is used exactly once to report the final honest number.

**Q: Why do we subtract the mean from CIFAR images?**
A: Zero-centering the data (subtracting the mean) ensures that the gradient doesn't systematically push in one direction. Also, it puts the data in a range where small random initial weights give usable gradients.

**Q: In the SVM gradient, why does the correct class column get `-count` × X[i]?**
A: For each wrong class j that violates the margin, the loss includes the term `s_j - s_{y_i} + 1`. The `s_{y_i}` appears negatively in each such term. So if `k` wrong classes violate the margin, the total contribution of `W[:,y_i]` to the loss is `-k * X[i]`, giving gradient `-k * X[i]`.

**Q: Why is the vectorized implementation so much faster than the naive one?**
A: Python loops are slow (interpreted). Matrix operations (`mm`, `clamp`, indexing) run in highly optimized C/CUDA code, often using SIMD instructions or GPU parallelism. A 40,000-sample minibatch computed in one matmul is 15–120× faster than 40,000 Python loop iterations.

**Q: What does the hidden layer in a neural network actually compute?**
A: Each hidden neuron computes a weighted sum of inputs (a linear projection) followed by a ReLU. Think of each neuron as a template detector: it fires when the input contains a certain pattern. The second layer then combines these template responses into a final class prediction.

**Q: Explain the chain rule in backpropagation in plain English.**
A: If changing weight W_ij by a small amount ε changes the neuron's output by A, and changing that output by A changes the loss by B, then changing W_ij by ε changes the loss by A×B. Backprop applies this logic backward through every layer, multiplying local derivatives to get the total effect on the loss.

**Q: What is learning rate decay and when is it useful?**
A: Multiply the learning rate by a factor < 1 (e.g., 0.95) after every epoch. This allows large steps at the start (fast rough learning) and finer steps later (careful convergence). Without decay, SGD keeps bouncing around the minimum rather than settling into it.

**Q: What does the SVM loss look like geometrically?**
A: In 2D, it creates a linear boundary between classes. The "support vectors" are the training examples sitting exactly on the margin boundaries. Everything on the correct side of the margin by more than 1 contributes zero loss.

**Q: Why is ReLU preferred over sigmoid or tanh?**
A:
1. ReLU doesn't saturate for positive inputs → no vanishing gradient there
2. Cheaper to compute: just a max(0, x)
3. Empirically trains faster
4. Naturally sparse: about 50% of neurons output 0 on average, which can be beneficial

**Q: What is the cross-entropy loss and what does it measure?**
A: Cross-entropy measures the "surprise" of the model's predictions. If the model assigns probability 0.9 to the correct class, loss = -log(0.9) ≈ 0.1 (low, good). If it assigns 0.01, loss = -log(0.01) ≈ 4.6 (high, bad). It's the negative log-likelihood of the data under the model.

**Q: What is the difference between training loss and validation accuracy?**
A: Training loss measures how well the model fits the training batch (smooth continuous signal). Validation accuracy measures classification correctness on unseen data (what we actually care about). We track both: dropping training loss + rising val accuracy = healthy learning. If training loss drops but val accuracy stays low = overfitting.

---

## Quick Reference: Shapes

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `X` | (N, D) | N images, each D=3072 pixels |
| `W` | (D, C) | D×C weight matrix |
| `scores` | (N, C) | Class scores for each image |
| `y` | (N,) | True class labels (integers 0–9) |
| `probs` | (N, C) | Softmax probabilities |
| `dW` | (D, C) | Gradient of loss w.r.t. W |
| `W1` | (D, H) | First layer weights |
| `b1` | (H,) | First layer biases |
| `W2` | (H, C) | Second layer weights |
| `b2` | (C,) | Second layer biases |
| `h1` | (N, H) | Hidden layer activations (post-ReLU) |

## Quick Reference: Loss Sanity Checks

| Classifier | What to check | Expected value |
|------------|--------------|----------------|
| SVM | loss with small W, no reg | ~9.0 (10 classes − 1 correct = 9 violating terms × 1 margin) |
| Softmax | loss with small W, no reg | ~2.3 = log(10) |
| NN | loss with small W, no reg | ~2.3 = log(10) |
| All | gradient check rel_error | < 1e-5 (linear) / < 1e-4 (NN) |

## Quick Reference: Target Accuracies

| Model | Validation | Test |
|-------|-----------|------|
| SVM | ≥ 37% | ≥ 35% |
| Softmax | ≥ 37% | ≥ 36% |
| Two-layer NN | ≥ 50% | ≥ 50% |

Baseline (random guessing on 10 classes): **10%**

---

*Good luck on the oral exam! The examiners will likely ask you to derive gradients on the whiteboard — focus on the chain rule, the softmax gradient formula, and the ReLU gate concept.*
