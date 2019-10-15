# MACH

This repository contains the research into (MACH) Multiple Asynchronous Component Hierarchies

The research focuses on using distillation to cut the direct-dependence between layers in very large Neural Networks. The idea is simple, instead of training the entire network concurrently, we train sub-sections asynchronously where each is using an approximation of the previous component to learn from the whole network. This allows each section to train asynchronously, only talking to its direct child and direct parent.

Specifically, we decompose our NN into set of components, i.e. a layers or a set of layers, as follows. F(x) = f0 o f1 o ... o fn. We augment the network with a set of trainable distillation functions for each component: f1', f2', ... fn' and during a call from our parent each component returns fi = fi o fi+1', instead of the full network. fi = fi o fi+1 o ... o fn.

This trains a single teacher and student model on mnist. The student distills from the teacher.
'''
$ python single_mach.py
'''

This trains a sequence of teachers and student models. Each component contains a teacher model and a student. The student is trains off of the previous teacher. The teacher uses this distilled student model to pull information from the previous teacher. Since the distilled model is 'local' we don't need to run the entire preceding network during inference.
'''
$ python sequential_mach.py
'''

This trains a kgraph where each component is connected to every other model.
'''
$ python kgraph_mach.py
'''

The learning component runs on its own loop during training appending gradient to a queue.
'''
$ python sequential_mach.py
'''
