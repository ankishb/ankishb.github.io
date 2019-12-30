---
layout: post
title: "Lottery Ticket Hypothesis"
date: 2019-12-30
tag: deep-learning
---


Generally, in deep learning, we build a deep model and then prune some weight of model (which don't effect the accuracy). Using this process, we can get a smaller size network (75% - 90% in size) at same accuracy. But what happens, if we train the subnetwork again? Will the accuracy improves? Can we reduce the size of network further? AI researchers have found that their accuracy decreases if we retrain the pruned network. So they cann't be pruned further. But researchers from **MIT** challenge this assumption and proposed a [smart method](https://arxiv.org/abs/1803.03635) to prune the model upto **0.3%** with samewhat similar accuracy.

They experiment with the initialization of the network and found that if we reinitialize the pruned network with the new random value, it performance decreases. But if they are re-initialized with the same weight value that are used at the start of training, we can get same or even more (sometime) on the trained model. They call this subnetwork as a winning ticket in big deep network.

#### Algorithms:
1. `Randomly initialize` a neural network `f(x; θ)` (call it `θ_0`)
2. `Train` the network for `k` iteration, so the parameter becomes `θ_k`
3. `Prune` `p%` of the parameter `θ_k` (create a mask `m` for that)
4. `Reset` the remaining parameters to their value in `θ_0`, creating the winning ticket `f(x; m*θ_0)`
5. `Repeat` step `2-5`, till the `accuracy` change are in `threshold` level.


Their experiment result are unbelieve. Iterative pruning make training much faster with better generalization. They proves that we can achieve same test accuracy with only **10% - 20%** of the original model. And this technique can be applied on any neural network structure.


In the following image, we can see that the model performance is **0.3%** more than original model with only **1.2%** of the original model.
<img src="{{ '/assets/images/hypothesis-lottery-winning.png' | relative_url }}" width="700" height="220" align="center" />

Let's talk about if this concept is related to our learning mechanism. I believe that our brain has a similar pruning mechanism. When we read a topic, it make some connection with weak and strong synaptic weights. But when we go through that topic again and again, some of the weak synaptic weight's connection breaks and other becomes more strong. **It also create some new connections based on a relation between the current topic and our prior experience**. Except this facts, this proposed technique behave the same. Following this, we can raise question like, how to choose hyper-parameter `k`, to train a model after pruning step? What would be the best pruning percentage value `p`? Author has done a detail analysis of these questions, please check out [paper](https://arxiv.org/abs/1803.03635).
