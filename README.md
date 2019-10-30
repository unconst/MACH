# MACH

<img src="assets/mach.png" width="1000" />


"In reality, the law always contains less than the fact itself, because it does not reproduce the fact as a whole but only in that aspect of it which is important for us, the rest being intentionally or from necessity omitted."

-- Ernst Mach

## Introduction
This repository contains research into a **fully-asynchronous** learning component called **MACH**. Asynchrony is achieved by breaking the **forward** and **backward**, locking problems for neural networks trained with vanilla back-propogation. We use two techniques: **(1) synthetic inputs**, and **(2) delayed gradients** to break these locks respectively.  

## Motivation

A network composed of fully-asynchronous components could grow to arbitrary size without a decrease in training speed. Such networks could realistically scale across entire racks of accelerators, data centres or internet, and reach the scale of the human cortex in learnable parameters.

## Pull Requests

In the interest of speed, just directly commit to the repo. To make that feasible, try to keep your work as modular as possible. I like to iterate fast by creating another sub project where tests can grow. For instance, in this repo, the sync_kgraph, and async_kgraph are separate independent implementations. Yes this creates code copying and rewrite, but allows fast development.

Also, use [Yapf](https://github.com/google/yapf) for code formatting. You can run the following to format before a commit.
```
$ pip install yapf
$ yapf --style google -r -vv -i .
```

## Projects:

***name***: synchronous single mach  <br/>
***dataset***: mnist <br/>
***torun***: ```python synchronous_single/main.py``` <br/>
description:   <br/>
    Single mach instance with a synthetic model (student) trained over a FFNN on MNIST.<br/>

***name***: synchronous sequence mach<br/>
***dataset***: mnist<br/>
***torun***: ```python synchronous_sequece/main.py```<br/>
***description***:<br/>
  Sequence of MACHs training synchrously. Each contains synthetic model for previous mach which feeds into the parent.<br/>

***name***: synchronous kgraph mach<br/>
***dataset***: mnist<br/>
***torun***: ```python synchronous_kgraph/main.py```<br/>
***description***:<br/>
  Synchronous single TF graph, training kgraph between k machs, each has input from all others.<br/>

***name***: asynchronous sequence mach<br/>
***dataset***: mnist<br/>
***torun***: ```python asynchronous_sequence/main.py```<br/>
***description***:  <br/>
  Asynchronous graph with k nodes aligned in sequence each training by passing messages to children. Messages propagate until a depth is reached before
  a response.<br/>

***name***: asynchronous kgraph mach<br/>
***dataset***: mnist<br/>
***torun***: ```python asynchronous_kgraph/main.py ```<br/>
***description***:  <br/>
  k async mach components arranged into a kgraph. Each uses synthetic inputs and delayed gradients.<br/>
