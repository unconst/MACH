# MACH

<img src="assets/mach.png" width="1000" />


"In reality, the law always contains less than the fact itself, because it does not reproduce the fact as a whole but only in that aspect of it which is important for us, the rest being intentionally or from necessity omitted."

-- Ernst Mach

## Introduction
This repository contains research into a **fully-asynchronous** learning component loosely called **MACH**. Asynchrony is achieved by breaking the **forward locking** and **backward locking**, two synchrnous limitation of neural networks trained with vanilla back-propogation. We use two techniques: **(1) synthetic inputs**, and **(2) delayed gradients** to break these locks respectively.  

## Motivation

A network of MACHs could hypothetically grow to arbitrary size while maintaining its constant training speed. If effective, these could be realistically scaled to multi-trillion parameter neural networks which consumed entire racks of fast inter-connected accelerators, entire data centres, or across the internet at large.

## Pull Requests

In the interest of speed, just directly commit to the repo. To make that feasible, try committing and pulling often. Keep your work modular as possible, I like to iterate fast by creating another sub project where tests can grow. For instance, in this repo, the sync_kgraph, and async_kgraph are separate implementations, you can run those sub projects independently, and commit to them without worring about overlap with other people.

Also, use [Yapf](https://github.com/google/yapf) for code formatting. You can run the following to format before a commit.
```
$ pip install yapf
$ yapf --style google -r -vv -i .
```

## Projects:

### Test 1
Run the following to test a single teacher and student model on mnist.
i.e F(x) = f0
```
$ python single_mach.py
```

### Test 2
This trains a sequence of teachers and student models. Each component contains a teacher model and a student. The student is trains off of the previous teacher. The teacher uses this distilled student model to pull information from the previous teacher. Since the distilled model is 'local' we don't need to run the entire preceding network during inference.
i.e F(x) = f0 o (f1' ~= f1 o (f2' ~= f2 o (f3' ~= f3)))
```
$ python sequential_mach.py
```

### Test 3
This trains a kgraph where each component is connected to every other model.
f0 = f0 o (f123 ~= (f1 ++ f2 ++ f3))
f1 = f0 o (f023 ~= (f0 ++ f2 ++ f3))
f2 = f0 o (f013 ~= (f0 ++ f1 ++ f3))
f3 = f0 o (f012 ~= (f0 ++ f1 ++ f2))
```
$ python kgraph_mach.py
```

### Test 4
The learning component runs on its own loop during training appending gradient to a queue.
```
$ python sequential_mach.py
```
