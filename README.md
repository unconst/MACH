<<<<<<< HEAD
# Biological Scale Neural Networks
### Unlocked Sparsely Gated Mixtures of Experts
=======
## Biological Scale Neural Networks [![Build Status](https://travis-ci.com/unconst/MACH.svg?branch=master)](https://travis-ci.com/unconst/MACH)
### Asynchronous Sparsely Gated Mixtures of Experts
>>>>>>> 737816eb9a80b39f051a8ec4a17bfceafa51b117

```
███████╗ ██████╗ ██████╗     █████╗ ██╗
██╔════╝██╔═══██╗██╔══██╗   ██╔══██╗██║
█████╗  ██║   ██║██████╔╝   ███████║██║
██╔══╝  ██║   ██║██╔══██╗   ██╔══██║██║
██║     ╚██████╔╝██║  ██║██╗██║  ██║██║
╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝
```

"In reality, the law always contains less than the fact itself, because it does not reproduce the fact as a whole but only in that aspect of it which is important for us, the rest being intentionally or from necessity omitted."

-- Ernst Mach

## Overview

[Sparsely gated mixtures of experts](https://arxiv.org/abs/1701.06538) with [synthetic inputs](https://arxiv.org/pdf/1608.05343.pdf) and [delayed gradients](https://arxiv.org/pdf/1804.10574.pdf)

For a deeper description read the [research](https://www.overleaf.com/read/fvyqcmybsgfj)

---

## Run

```
$ pip install -r requirements.txt
$ python main.py
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
