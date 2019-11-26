# Biological Scale Neural Networks
### Unlocked Sparsely Gated Mixtures of Experts

<img src="assets/mach.png" width="1000" />


"In reality, the law always contains less than the fact itself, because it does not reproduce the fact as a whole but only in that aspect of it which is important for us, the rest being intentionally or from necessity omitted."

-- Ernst Mach

## Introduction
This repository contains research into Biological Scale Neural Networks. Asynchrony is achieved by breaking the **forward** and **backward** locking problems inherited from back-propogation. We use two techniques: **[1] synthetic inputs**, and **[2] delayed gradients** to break these locks respectively. 

For a deeper description read the research: (https://www.overleaf.com/read/fvyqcmybsgfj)

---

## Run

```
$ pip install -r requirements.txt
$ python experiments/<experiment_name>/main.py
```
---

## Pull Requests

In the interest of speed, just directly commit to the repo. To make that feasible, try to keep your work as modular as possible. I like to iterate fast by creating another sub project where tests can grow. For instance, in this repo, the sync_kgraph, and async_kgraph are separate independent implementations. Yes this creates code copying and rewrite, but allows fast development.

Also, use [Yapf](https://github.com/google/yapf) for code formatting. You can run the following to format before a commit.
```
$ pip install yapf
$ yapf --style google -r -vv -i .
```

---

## References:

1. Decoupled Neural Interfaces using Synthetic Gradients <br/>
https://arxiv.org/pdf/1608.05343.pdf

1. Decoupled Parallel Backpropagation with Convergence Guarantee <br/>
https://arxiv.org/pdf/1804.10574.pdf

1. Outrageously Large Neural Networks: Sparsely Gated Mixtures of Experts <br/>
https://arxiv.org/abs/1701.06538
