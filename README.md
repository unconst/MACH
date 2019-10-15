# MACH

<img src="assets/mach.png" width="1000" />


"In reality, the law always contains less than the fact itself, because it does not reproduce the fact as a whole but only in that aspect of it which is important for us, the rest being intentionally or from necessity omitted." - Ernst Mach

## Introduction
This repository contains the research into what are loosely called MACHs. The attempt is building a learning component which can be trained in coordination with others, while not being dependent on them during inference or training. The research focuses on using distillation to cut this dependence. The idea is simple, we train each component to distill information from its neighbors rather than be directly connected. This allows each section to train asynchronously, speaks only to its direct neighbors as it trains, while they talk to their neighbors etc etc.

More formally, a standard Feed forward NN can be defined as a sequence of compositions F(x) = f0 o f1 o ... o fn. We augment each composition with a set of distilled approximations f1', f2', ... fn' which train to approximate the previous. Then during inference each component returns fi = fi o (fi+1' ~= fi+1), instead of the full network. fi = fi o fi+1 o ... o fn.

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
