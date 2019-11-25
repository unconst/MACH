# MACH

<img src="assets/mach.png" width="1000" />


"In reality, the law always contains less than the fact itself, because it does not reproduce the fact as a whole but only in that aspect of it which is important for us, the rest being intentionally or from necessity omitted."

-- Ernst Mach

## Introduction
This repository contains research into **fully-asynchronous** learning components called **MACHs**. Asynchrony is achieved by breaking the **forward** and **backward** locking problems inherited from back-propogation. We use two techniques: **[1] synthetic inputs**, and **[2] delayed gradients** to break these locks respectively. 

## Run

```
$ pip install -r requirements.txt
$ python mach/main.py
```

## Motivation

A network composed of fully-asynchronous components could grow to arbitrary size without a decrease in training speed. Such networks could realistically scale across entire racks of accelerators, data centres or internet, and reach the scale of the human cortex in learnable parameters.

## Pull Requests

In the interest of speed, just directly commit to the repo. To make that feasible, try to keep your work as modular as possible. I like to iterate fast by creating another sub project where tests can grow. For instance, in this repo, the sync_kgraph, and async_kgraph are separate independent implementations. Yes this creates code copying and rewrite, but allows fast development.

Also, use [Yapf](https://github.com/google/yapf) for code formatting. You can run the following to format before a commit.
```
$ pip install yapf
$ yapf --style google -r -vv -i .
```

## To Run

Run a asynchronous kgraph of mach components on MNIST. <br/>
```
$ pip install -r requirements.txt
$ python asynchronous_kgraph/main.py 
``` 

<br/>

## Method Description

Each MACH is a self contained learning component which runs on its own thread, process, or host computer. It trains asynchronously, on its own local dataset, only interacting with its neighbors by sending tensors along the edges. These message are of two types, Spikes and Grads.

#### Spikes
A Spike is a forward execution through the graph it passes a tensor of inputs along a directed edge to a child. The call is recursive and triggers further spike queries on each of its children before a response is returned. The problem is of course, that we cannot receive a response until all downstream nodes have executed and responded. Nodes are 'forward-locked'.

##### Synthetic inputs

We avoid forward-locking by adopting synthetic inputs [1]. A synthetic input is the output of model trained to mimic the outputs from a child. The synthetic models are trained by minimizing the following loss term: synthetic loss = | syn_i_j(x) - d_j(x) |^2. Where syn_i_j(x) is the output of the synthetic model at the $i^t^h$ node learning to mimic the output of its child d_j(x).

#### Grads

A Grad back-propagates the training signal through the graph. It is a message created by a parent and sent to a child passing an input(x) and a gradient term ∂b = ∂a / ∂b,  the error signal calculated by the parent. 

If the child node that receives the Grad has children, then the call is recursive, triggering $n$ Grad calls on each child. These have form (x, ∂a/∂c) where ∂a/∂c is derived by the chain rule: ∂a/∂c = ∂a/∂b ∂b/∂c

Each Grad call is non-blocking and the node can move onto the next training step without waiting for the remaining network to apply the step. This creates delayed gradients [2].

##### Delayed Gradients

Delayed gradients alter the back-propagation algorithm. Error terms propagated by the parent node a, through b and then to c, sent at the t.th step, may be k steps behind, where k the number of steps computed at node b between producing its output and receiving the corresponding gradient. Thus our true chain rule takes form: a^(t+k} / c^(t+k) = a^t / b^t * b^(t+k) / c^(t+k)

As the distance between nodes increase, so will k, potentially harming the ability for the network to converge. This was investigated theoretically in [2], their analysis assumed a Lipschitz-continuous gradient and used this to prove that the learned model will converge to the neighborhood of the critical point for sufficiently small learning rates. The experimental results from that paper confirmed their theoretical analysis and demonstrated that the proposed method achieved significant speedup without loss of accuracy.


## References:

1. Decoupled Neural Interfaces using Synthetic Gradients <br/>
https://arxiv.org/pdf/1608.05343.pdf

1. Decoupled Parallel Backpropagation with Convergence Guarantee <br/>
https://arxiv.org/pdf/1804.10574.pdf

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
