# MACH

This repository contains the research into (MACH) Multiple Asynchronous Component Hierarchies

The research focuses on using distillation to cut the direct-dependence between layers in very large Neural Networks. The idea is simple, instead of training the entire network concurrently, we train sub-sections asynchronously where each is using an approximation of the previous component to learn from the whole network. This allows each section to train asynchronously, only talking to its direct child and direct parent.

Specifically, we decompose our NN into set of components, i.e. a layers or a set of layers, as follows. F(x) = f0 o f1 o ... o fn. We augment the network with a set of trainable distillation functions for each component: f1', f2', ... fn' and during a call from our parent each component returns fi = fi o fi+1', instead of the full network. fi = fi o fi+1 o ... o fn.

To run the experiment on a single component:
'''
$ python main.py
'''
