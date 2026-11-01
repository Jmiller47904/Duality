# DRAI V1 — Dynamic Resonance AI  
### Production-Ready Self-Organizing Memory for Transformer Models

DRAI V1 is the **first stable, safe, production-ready attractor-memory algorithm** for transformer attention.  
It adds **true self-organizing memory** to frozen language models **without degrading baseline performance**, even on small models such as **pythia-70m** and **pythia-410m**.

Unlike RAG or external memory systems, DRAI V1 stores memory **inside attention**, using attractor dynamics to detect, stabilize, and reinforce meaningful patterns over the course of generation.

---

## 🔍 What DRAI V1 Actually Does

DRAI V1 augments transformer attention by:

1. **Observing query patterns** across tokens  
2. **Forming attractors** when patterns recur  
3. **Updating attractors** through controlled exponential averaging  
4. **Decaying unused attractors** and evicting weak ones  
5. **Generating synthetic K/V pairs** representing the model’s learned internal memory  
6. **Injecting memory into attention outputs** using a safe, gated influence mechanism

This creates a lightweight, model-internal “dynamic memory field” that supports stable reasoning over the course of a generation.

DRAI V1 requires **no training**, **no database**, **no infrastructure**, and **no tuning** for small models.

---

## 🚀 Quick Start

```python
from drai import apply_drai_v1
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")
model = apply_drai_v1(model)    # Uses conservative V1 config (safe for small models)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
inputs = tokenizer("The future of memory systems is", return_tensors="pt")

outputs = model.generate(inputs["input_ids"], max_length=80)
print(tokenizer.decode(outputs[0]))
```

Your model now has **stable dynamic memory**.

---

## 🧠 Why V1 Exists

Earlier DRAI phases used more aggressive attractor dynamics that worked on larger models but risked destabilizing smaller ones.  
V1 is a **complete redesign** focused on:

- **Safety**  
- **Deterministic training-free behavior**  
- **Strict gating and burn-in**  
- **Predictable attractor formation**  
- **Correct K/V injection**

V1 finally makes attractor memory practical for real-world language model usage.

---

## 🧩 How It Works (Technical Summary)

### 1. Pattern Detection  
For each query vector \(q_t\):

- Normalize queries and attractors  
- Compute cosine similarity  
- Find best-matching attractor  
- Accept match only if:  
  - similarity > `theta_match`  
  - attractor strength > `strength_min`

Otherwise: marked as **novel**.

---

### 2. Accumulation  
Matched attractors receive:

- EMA-based centroid updates  
- Strength reinforcement proportional to match count  

Strength grows ~6–8 units per token due to self-similarity of generation queries.

---

### 3. Novel Pattern Formation  
Novel queries create attractors **only in free slots**, preventing uncontrolled growth.

Unused attractors decay and are evicted automatically.

---

### 4. Burn-In (Critical for Stability)

DRAI V1 will **not influence the model** until:

- total attractor strength ≥ `burn_in_threshold`, OR  
- number of tokens ≥ `burn_in_tokens` (backup mode)

This prevents:

- early noise amplification  
- self-feedback loops  
- attractors forming from repetition or unstable warm-up states

---

### 5. Memory Injection  
After burn-in, V1 produces **synthetic K/V pairs**:

```
[batch, seq_len, num_drai_heads, head_dim]
```

These represent a **scaled, normalized global memory field** derived from attractors.  

Injection uses:

- **soft gating** with `tanh(total_strength)`  
- **max_influence_scale** to cap effect size  
- **per-layer DRAI heads** (usually 1 for small models)

For small models, V1 uses a **simple additive correction** to the attention output—safe and effective without rewriting the entire attention mechanism.

---

## 📊 Monitoring Attractors

```python
from drai import get_drai_v1_stats

stats = get_drai_v1_stats(model)
print(stats["num_drai_layers"])
print(stats["total_strength"])
print(stats["layers"])
```

This reveals:

- active attractors  
- total and per-layer strength  
- `burn_in_active`, computed using the configured strength or token threshold
- equilibrium dynamics  

---

## ⚙️ Recommended Configurations

### Small Models (<1B)
Use `get_v1_conservative_config()`:

- 16 attractors  
- strict matching (0.8)  
- gentle influence  
- mid-layer injection  
- burn-in threshold ≈ 50  

### Larger Models (>1B)
Use `get_v1_standard_config()`:

- 32 attractors  
- slightly faster updates  
- full-layer injection  
- burn-in threshold ≈ 30  

---

## 📈 Performance Profile

- **Baseline perplexity preserved** (no degradation)  
- **5–7% runtime overhead** (dominated by similarity computation)  
- **~20KB memory footprint** for typical models  
- **Stable behavior up to 50+ tokens generation**

DRAI V1 is **not** a performance improver (for now)—it is a *safe* attractor-memory system.

---

## 🧭 Roadmap

### V1 (Current)
✓ Stable attractor memory  
✓ Correct K/V generation  
✓ Safe gating and burn-in  
✓ Mid-layer injection for small models  

### V2–V3 (Planned)
- True concatenated K/V injection into attention (Plan A full form)  
- Multi-head attractor specialization  
- Learned attractor projections  
- Support for GPT-2, LLaMA, Mistral  

---

## 📄 Citation

```
@software{drai2025,
  title={DRAI V1: Stable Self-Organizing Memory for Transformer Models},
  author={Halcyon AI Research},
  url={https://github.com/HalcyonAIR/Duality},
  year={2025}
}
```

---

## 📝 Summary

DRAI V1 is the **first fully stable, production-ready attractor memory** for generative models.  
It gives any GPT-NeoX model dynamic memory with:

- **no training**  
- **no performance loss**  
- **no infrastructure**  

It is safe, predictable, and easy to adopt—finally making attractor dynamics practical for real use.

---
