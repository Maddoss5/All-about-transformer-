# All About Transformers: Architecture Deep Dive

## Part 1: Self-Attention Mechanism

### The Intuition
The goal is to determine how related each token is to the tokens that came before it. We want to construct a context vector for every word.

1.  **Simple Summation:** Ideally, the context for token $x_i$ is a sum of previous tokens:
    $$\text{Attention}(x_i) = \sum_{j < i} x_j$$
2.  **Weighted Summation:** Not all previous tokens are equally important. We need a weight ($W_{ij}$) to scale the interaction:
    $$\text{Attention}(x_i) = \sum_{j < i} (W_{ij} \cdot x_j)$$

### Calculating Similarity (The Weights)
To find the weight $W_{ij}$, we calculate the **similarity** between the current token ($x_i$) and a previous token ($x_j$) using the dot product, followed by a Softmax to create a probability distribution.

$$W_{ij} = \text{Softmax}(x_i \cdot x_j)$$



---

## Part 2: The Attention Head (Q, K, V)

In the Transformer, we refine the simple dot product by introducing three learnable matrices to project the input into different subspaces.

### 1. The Projections
For input vector $x_i$ (dimension $d$), we generate:
* **Query ($Q$):** The current token seeking information ($x_i \cdot W_q$).
* **Key ($K$):** The previous tokens identifying themselves ($x_j \cdot W_k$).
* **Value ($V$):** The actual information content to extract ($x_j \cdot W_v$).

### 2. The Dimensions
If we have embedding dimension $d$ and we split into heads with dimension $d_k$:
* $W_q, W_k \in \mathbb{R}^{d \times d_k}$
* $W_v \in \mathbb{R}^{d \times d_v}$

$$
\begin{aligned}
Q &= (N \times d) \cdot (d \times d_k) \rightarrow (N \times d_k) \\
K &= (N \times d) \cdot (d \times d_k) \rightarrow (N \times d_k) \\
V &= (N \times d) \cdot (d \times d_v) \rightarrow (N \times d_v)
\end{aligned}
$$

### 3. Scaled Dot-Product Attention
We calculate the dot product of Query and Key. To prevent the Softmax gradients from vanishing when dimensions are large, we scale by $\frac{1}{\sqrt{d_k}}$.

$$\text{Score} = \frac{Q \cdot K^T}{\sqrt{d_k}}$$

$$\text{Attention Output} = \text{Softmax}(\text{Score}) \cdot V$$

> **Why Q/K/V and not just Embeddings?**
> 1.  **Subspaces:** It allows the model to look at the input data from different perspectives (grammar, semantic meaning, tone) simultaneously.
> 2.  **Disambiguation:** It helps resolve ambiguities (e.g., distinguishing whether "He" refers to "Vatsal" or "Madd" based on specific context, not just general word closeness).

### 4. Multi-Head Output
If we have $h$ heads, we concatenate their outputs and project them back to the original dimension $d$ using a matrix $W_o$.
$$\text{Output} = \text{Concat}(head_1, ..., head_h) \cdot W_o \rightarrow (N, d)$$

---

## Part 3: The Transformer Block

The Transformer block combines attention with normalization and feed-forward networks. The architecture below follows the **Pre-Norm** formulation (common in GPT-3/LLaMA).



[Image of Transformer architecture block diagram]


### The Flow
For a token $x_i$:

1.  **First Residual Stream (Attention):**
    * $y_1 = \text{LayerNorm}(x_i)$
    * $y_2 = \text{MultiHeadAttention}(y_1)$
    * $y_3 = x_i + y_2$  *(Residual Connection)*

2.  **Second Residual Stream (FFN):**
    * $y_4 = \text{LayerNorm}(y_3)$
    * $y_5 = \text{FFN}(y_4)$
    * $y_6 = y_3 + y_5$ *(Output of Block)*

### The Components
* **Residual Connections ($x + f(x)$):** These allow gradients to flow through the network without vanishing, effectively creating a "highway" for information to pass from early layers to later layers.
* **Feed Forward Network (FFN):** A simple MLP that processes each token individually. It usually projects the dimension up (e.g., $4d$) and then back down.
    * `Linear(d -> 4d)` -> `Activation (ReLU/GELU)` -> `Linear(4d -> d)`

---

## Part 4: Working with Transformers

### Transformers vs. LSTMs
1.  **Parallelization:** LSTMs are sequential ($t_2$ depends on $t_1$). Transformers process the entire sequence $(N, d)$ at once using matrix multiplication.
2.  **Long-Term Dependencies:** LSTMs suffer from "forgetting" over long sequences. Transformers have direct access to all previous tokens via the attention mechanism (path length is $O(1)$).

### Masking (Preventing "Cheating")
In a Decoder (GPT-style), $x_i$ should not see $x_{i+1}$.
We calculate the dot product scores ($Q \cdot K^T$) and then apply a mask before the Softmax:
* Set indices where $j > i$ to $-\infty$.
* $\text{Softmax}(-\infty) = 0$.

---

## Part 5: Positional Embeddings

Since Attention is permutation invariant (it treats "A B" the same as "B A"), we must inject order information.



We add a **Positional Embedding** vector to the **Token Embedding** vector before entering the first Transformer block.

$$x_{input} = x_{token\_embedding} + x_{position\_embedding}$$

This allows the model to distinguish between "Vatsal studies, Madd does not" vs "Madd studies, Vatsal does not," even if the bag-of-words similarity is identical.

---

## Part 6: Language Modeling Head

To use this for text generation:
1.  Take the output of the final Transformer block ($y_{final}$).
2.  Pass it through an **Unembedding Layer** (Linear projection from $d \rightarrow \text{Vocab Size}$).
3.  Apply Softmax to get probabilities for the next token.
