
## Biological Scale Neural Networks [![Build Status](https://travis-ci.com/unconst/MACH.svg?branch=master)](https://travis-ci.com/unconst/MACH)
### Sequential Distillation models.

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

## TL;DR
Increasing depth-wise scaling by training sequential distillation models.


## Motivation

Depth is good, but deeper networks increasingly suffer from ’gradient locking’  if  the  network  is  required to update synchronously [1] [2] [4]. This issue can be avoided using depth-wise model parallelism, where sections of the network train independently. This, however, creates delayed  gradients [5] which affect convergence when component depth exceed a certain size.

This research investigates a new class of neural network architecture which is composed of many sequentially connected sub-components each training asynchronously and distilling knowledge from their child.

The resulting model is, by its nature, a distilled versions of itself thus immediately usable in a production environment at reduced computational cost. Finally, the approach is well suited to an internet-wide environment making p2p training a potential avenue for future research.   

For a deeper description read the [research](https://www.overleaf.com/read/fvyqcmybsgfj) or join  proj-mach on slack.

---

## Run

```
$ virtualenv env && source env/bin/activate && pip install -r requirements.txt
$ python main.py
```
---

## Resources

Paper: https://www.overleaf.com/read/fvyqcmybsgfj  </br>
Code: https://www.github.com/unconst/Mach

---
## Pull Requests

Use [Yapf](https://github.com/google/yapf) for code formatting
```
$ pip install yapf
$ yapf --style google -r -vv -i .
```

---

## References:

References
1.	Decoupled Neural Interfaces using Synthetic Gradients. </br>
https://arxiv.org/pdf/1608.05343.pdf

2.	Decoupled Parallel Backpropagation with Convergence Guarantee.  </br>
https://arxiv.org/pdf/1804.10574.pdf

3.	 Outrageously Large Neural Networks: Sparsely Gated Mixtures of Experts.  </br>
https://arxiv.org/abs/1701.06538

4.	AMPNet: Asynchronous Model-Parallel Training for Dynamic Neural Networks.  </br> https://www.microsoft.com/en-us/research/wp-content/uploads/2017/07/1705.09786.pdf

5.	An analysis of delayed gradients problem in asynchronous SGD.  </br> https://pdfs.semanticscholar.org/716b/a3d174006c19220c985acf132ffdfc6fc37b.pdf

6. Improved Knowledge Distillation via Teacher Assistant: Bridging the Gap Between Student and Teacher.  </br>
https://arxiv.org/abs/1902.03393
