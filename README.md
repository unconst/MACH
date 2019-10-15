# MACH

This repository contains the research into what are loosely called MACHs. The attempt is building a learning component which can be trained in coordination with others, while not being dependent on them during inference. The research focuses on using distillation to cut this dependence. The idea is simple, we train each component to distill information from its neighbors rather than be directly connected. This allows each section to train asynchronously, during training it speaks to its neighbors as it trains its distilled network, during inference it uses the distilled model.

More formally, if a standard Feed forward NN were defined as a sequence of compositions F(x) = f0 o f1 o ... o fn. We would augment each composition with a set of distilled approximations f1', f2', ... fn'. Then during inference each component returns fi = fi o (fi+1' ~= fi+1), instead of the full network. fi = fi o fi+1 o ... o fn.

Run the following to test a single teacher and student model on mnist.
i.e F(x) = f0'
'''
$ python single_mach.py
'''

This trains a sequence of teachers and student models. Each component contains a teacher model and a student. The student is trains off of the previous teacher. The teacher uses this distilled student model to pull information from the previous teacher. Since the distilled model is 'local' we don't need to run the entire preceding network during inference.
i.e F(x) = f0 o (f1' ~= f1 o (f2' ~= f2 o (f3' ~= f3)))
'''
$ python sequential_mach.py
'''

This trains a kgraph where each component is connected to every other model.
f0 = f0 o (f123 ~= (f1 ++ f2 ++ f3))
f1 = f0 o (f023 ~= (f0 ++ f2 ++ f3))
f2 = f0 o (f013 ~= (f0 ++ f1 ++ f3))
f3 = f0 o (f012 ~= (f0 ++ f1 ++ f2))
'''
$ python kgraph_mach.py
'''

The learning component runs on its own loop during training appending gradient to a queue.
'''
$ python sequential_mach.py
'''
