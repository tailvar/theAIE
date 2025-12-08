import math
import torch
from torch.testing import assert_close
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (..., T, d_k / d_v) tensors
    mask: optional  boolean or float mask broadcastable to (..., T, T)
        - boolean: True = keep, False = mask out
        - float: added directly to the scores (eg 0 or 1e-9

    returns:
        Y    (..., T, d_v)
        attn (..., T, T)
    """
    # 1. Compute scaled scores: S = Q K^T / sqrt(d_k)
    d_k = Q.shape[-1]
    # (..., T, d_k) @ (..., d_k, T) -> (..., T, T)
    scores = Q @ K.transpose(-2,-1) / (d_k**0.5)

    # 2. Add mask 
    if mask is not None:
        if mask.dtype == torch.bool:
            # boolean mask: False entries get a large negative score
            scores = scores.masked_fill(~mask, -1e9)
        else:
            # float mask: assumed to be additive (eg 0 or -1e9)
            scores = scores + mask

    # 3. Softmax over last dimension to get attention weights
    attn = F.softmax(scores, dim=-1) # (..., T, T)

    # 4. Weighted sum of values: Y = AV
    # (..., T, T) @ (..., T, d_v) -> (..., T, d_v)
    Y = attn @ V
    return Y, attn

def test_scaled_dot_product_attention():
    # construct a tiny example
    Q_np = [
        [1.0,0.0],
        [0.0,1.0],
        [1.0,1.0],
    ]

    K_np = [
        [1.0,0.0],
        [1.0,1.0],
        [0.0,1.0],
    ]

    V_np = [
        [1.0,0.0],
        [0.0,2.0],
        [3.0,1.0],
    ]

    # convert to tensors with batch dim B = 1: shape (1, T, d_k/d_v)
    Q = torch.tensor(Q_np, dtype=torch.float32).unsqueeze(0) # (1, 3, 2)
    K = torch.tensor(K_np, dtype=torch.float32).unsqueeze(0) # (1, 3, 2)
    V = torch.tensor(V_np, dtype=torch.float32).unsqueeze(0) # (1, 3, 2)
    
    # --- expected hand computed results ---
    
    # QK^T (unscaled)
    expected_scores_unscaled=torch.tensor(
        [[[1.0,1.0,0.0],
          [0.0,1.0,1.0],
          [1.0,2.0,1.0]]],dtype=torch.float32)
    
    # scaled scores S = QK^T / sqrt(d_k), d_k = 2
    expected_S = torch.tensor(
        [[[0.70710678, 0.70710678, 0.0],
          [0.0, 0.70710678, 0.70710678],
          [0.70710678, 0.70710678*2,0.70710678]]], dtype=torch.float32)
    
    expected_A = torch.tensor(
        [[[0.40111209, 0.40111209,0.19777581],
          [0.19777581,0.40111209,0.40111209],
          [0.24825508,0.50348984,0.24825508]]],dtype=torch.float32)
    
    # output Y = A V
    expected_Y = torch.tensor(
        [[[0.99443954, 1.0],
          [1.40111209, 1.20333628],
          [0.99302031, 1.25523477]]],dtype=torch.float32,
    )
    
    # --- Compute using the same formulas as the implementation ---
    
    d_k = Q.shape[-1]
    
    tol=1e-6
    
    
    # Unscaled scores QK^T
    scores_unscaled = Q @ K.transpose(-2,-1) # (1, 3, 3)
    assert_close(scores_unscaled, expected_scores_unscaled, atol=tol, rtol=tol)
    
    # Scaled scores S
    S = scores_unscaled / math.sqrt(d_k)
    assert_close(S, expected_S, atol=tol, rtol=tol)
    
    # --- Run through the scaled_dot_product_attention ---
    Y, A = scaled_dot_product_attention(Q, K, V, mask=None)
    
    # check attention weights A
    assert_close(A, expected_A, atol=tol, rtol=tol)
    
    # check final outputs Y
    assert_close(Y, expected_Y, atol=tol, rtol=tol) 

