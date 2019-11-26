# Biological Scale Neural Networks
### Unlocked Sparsely Gated Mixtures of Experts

<img src="assets/mach.png" width="1000" />


"In reality, the law always contains less than the fact itself, because it does not reproduce the fact as a whole but only in that aspect of it which is important for us, the rest being intentionally or from necessity omitted."

-- Ernst Mach

## Overview

**[1] Sparsely gated mixtures of experts** with **[2] synthetic inputs** and **[3] delayed gradients**.

For a deeper description read the research: (https://www.overleaf.com/read/fvyqcmybsgfj)

---

## Run

```
$ pip install -r requirements.txt
$ python experiments/<experiment_name>/main.py
```
---

## Pull Requests

Use [Yapf](https://github.com/google/yapf) for code formatting
```
$ pip install yapf
$ yapf --style google -r -vv -i .
```

---

## References:

1. Outrageously Large Neural Networks: Sparsely Gated Mixtures of Experts <br/>
https://arxiv.org/abs/1701.06538

1. Decoupled Neural Interfaces using Synthetic Gradients <br/>
https://arxiv.org/pdf/1608.05343.pdf

1. Decoupled Parallel Backpropagation with Convergence Guarantee <br/>
https://arxiv.org/pdf/1804.10574.pdf

