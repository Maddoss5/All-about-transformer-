# All About Transformers: Deep Dive

## Part 1: The Intuition of Self-Attention

### 1. The Goal
We want to determine how related each token is to the tokens that came before it. We want to capture the "context" for every word.

### 2. Attempt 1: Simple Summation
Ideally, the context for a token ($x_i$) is a sum of the vectors of previous tokens.
$$
\text{Attention}(x_i) = \sum_{j < i} x_j
$$

### 3. Attempt 2: Weighted Summation
However, we shouldn't give equal weight to all previous tokens. Some are more important than others. We need a weight ($W_j$) to scale the importance.
$$
\text{Attention}(x_i) = \sum_{j < i} (W_j \cdot x_j)
$$

### 4. Calculating the Weights (Similarity)
How do we find this weight? We calculate the **similarity** between the current token ($x_i$) and a previous token ($x_j$).
* We use the **dot product** for similarity.
* We pass the result through a **Softmax** function to turn scores into probabilities (affinities).

$$
W_{ij} = \text{Softmax}(x_i \cdot x_j)
$$

$$
\text{Attention} = \sum_{j < i} W_{ij} \cdot x_j
$$

---

## Part 2: The Transformer Mechanism (Attention Head)

In the actual Transformer architecture, we refine this by introducing three learnable matrices: **Query**, **Key**, and **Value**.

### 1. The Three Matrices
Instead of using the raw input $x$, we project it into three different versions:
* **Query ($Q$):** The current token looking for information.
* **Key ($K$):** The previous tokens identifying themselves.
* **Value ($V$):** The actual content/information to be extracted.

If the input $X$ has dimension $(N, d)$:
1.  $W_q$ (Dimension: $d \times d_k$)
2.  $W_k$ (Dimension: $d \times d_k$)
3.  $W_v$ (Dimension: $d \times d_v$)

### 2. The Projections
We multiply the input $x_i$ by these matrices:

$$
\begin{aligned}
Query &= x_i \cdot W_q \rightarrow (1 \times d_k) \\
Key &= x_i \cdot W_k \rightarrow (1 \times d_k) \\
Value &= x_i \cdot W_v \rightarrow (1 \times d_v)
\end{aligned}
$$

*(Note: In implementation, we do this for all $N$ tokens at once using matrix multiplication.)*

### 3. Scaled Dot-Product Attention
To find the similarity, we take the dot product of the Query and the Key ($Q \cdot K^T$).

**The Scaling Problem:**
The dot product can become very large as the dimension ($d_k$) increases, pushing the Softmax into regions with extremely small gradients. To fix this, we normalize by $\sqrt{d_k}$.

$$
\text{Score} = \frac{Q_i \cdot K_j}{\sqrt{d_k}}
$$

### 4. The Final Formula
The attention for token $i$ is the weighted sum of the Values ($V$), weighted by the Score:

$$
\text{Attention}(x_i) = \sum_{j \le i} \text{Softmax}\left(\frac{Q_i \cdot K_j}{\sqrt{d_k}}\right) \cdot V_j
$$

> **Note:** The output dimension of one attention head is $(N, d_v)$.

---

## Part 3: Why Query, Key, and Value?

Why not just use the original Embedding matrix?

### 1. Context vs. Global Meaning
Embeddings capture global closeness (e.g., "King" is close to "Queen"). However, they don't capture context-specific relationships.
* *Example:* "Vatsal studies a lot, Madd does not. Yet **he** gets high scores."
* Standard embeddings might associate "he" generally with males.
* Attention needs to dynamically determine that in *this specific sentence*, "he" refers to **Madd**, not Vatsal.

### 2. Subspaces and Multi-Head Attention
We use multiple heads to capture different types of relationships simultaneously.
* **Head 1:** Might focus on grammar (Subject-Verb).
* **Head 2:** Might focus on coreference (He $\rightarrow$ Madd).

If we define $h$ as the number of heads:
1.  We compute $h$ different attention outputs.
2.  We **concatenate** them.
    $$\text{Concat Output} \rightarrow (N, h \cdot d_v)$$
3.  We use a final linear projection $W_o$ to map this back to the original dimension $d$.

$$
\text{Final Output} = \text{Concat}(head_1, ..., head_h) \cdot W_o \rightarrow (N, d)
$$

This ensures the output of the Multi-Head Attention layer is the same shape as the input, allowing us to stack layers.
