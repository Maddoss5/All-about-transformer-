# All About Transformers

## Self-Attention Mechanism

### The Core Concept
The goal of self-attention is to determine how related each token is to the ones that came before it. We need to capture the contextual dependency between the current token and previous tokens.

### 1. Similarity
To measure this relationship, we often use the **dot product**. This calculates the similarity between the current token vector ($x_i$) and a previous token vector ($x_j$).

If we sum these similarities for previous tokens (where $j < i$):

$$
\text{Similarity} = \sum_{j < i} (x_i \cdot x_j)
$$

> **Note:** The condition $j < i$ implies "Causal" or "Masked" attention (common in GPT-style decoders), ensuring the model cannot see the future.

### 2. Weighted Importance
We should not give equal weight to all previous tokens, as some context is more relevant than others. To solve this, we introduce a learnable weight parameter.

### 3. Final Formulation
By combining the similarity (dot product) with the weights ($W_j$), we derive the attention value:

$$
\text{Attention} = \sum (W_j) \cdot (x_i \cdot x_j)
$$
