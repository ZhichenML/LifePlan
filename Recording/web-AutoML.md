## Web interface learner

### Introduction
Natural language instructed tasks such as [visual question answering (VQA)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8237355), need [reasoning ability](https://papers.nips.cc/paper/8203-learning-to-reason-with-third-order-tensor-products.pdf). Previous works employ [language parsers](https://arxiv.org/pdf/1511.02799.pdf) to ensemble a toolbox of modules, or learn the modular layout by [imitation learning and policy gradient](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8237355). In those VQA works, the related objects can often be idenfied from question sentense and linked to objects in the image by specific modular. Therefore, the actions involve only "seeing". For web interface learner, the actions include mouse clicks and simple keyboard operations. It needs more complex reinforcement learning techniques to understand the task meaning. Challenges include sparse rewards and generalization across different tasks.

Instruction following is ensencially sequence to sequence language translation from words to actions.

Challenges:
- Single wrong step may lead to unforseen behaviour
- Explicit step-by-step instruction following already works
- Sequence to sequence in principle
- Heavy computational cost

Contribution:
- A general model over all tasks
- Automatic modular network design

### Ideas

- Use DARTs To Search The Module Composition
- Training a Neural Network to Imitate DARTs using Imitation Learning

  - Faster Testing Time , rather than training the _architecture_ everytime using Gradient Ascend, training a Master Neural Network that mimic the search.

  - The "Plan of the Action" can also be used to guide by

    - Human Demonstration

    - DARTs

  - Faster Training Time ? Not Relies on the Policy Gradient with Very Sparse Reward

    - Use of Gradient Guided Search

  - Might be Able to Generalize Well.

- Problems

    - Very Sparse reward

    - What happen if the plan gone wrong ?

      - Have some kind of re-plan because we can't "baked" the plan completely.

    - Might need easier task to test the idea -> the Web might be too hard to debug/learn.

      - Consider the BabyAI where you can explicitly build a graph

      - Consider VQA graph building using DARTs

    - Speed ?


### To do list

- Code of MAML
- Code of workflow web interface
- Architecture search for neural modular network
- Marginalized MAML

### Related project
- [Stanford NLP · GitHub](https://github.com/stanfordnlp)
- [MiniWoB++](https://github.com/stanfordnlp/miniwob-plusplus)
- [Mini World of Bits · Issue #55 · aikorea/awesome-rl · GitHub](https://github.com/aikorea/awesome-rl/issues/55)
- [GitHub - aikorea/awesome-rl: Reinforcement learning resources curated](https://github.com/aikorea/awesome-rl)

### Paper list

**Main Task** -- https://stanfordnlp.github.io/miniwob-plusplus/

#### Related Tasks
- [*Work flow guided exploration*](https://arxiv.org/pdf/1802.08802.pdf)
- [*World of Bits: An Open-Domain Platform for Web-Based Agents*](http://proceedings.mlr.press/v70/shi17a/shi17a.pdf)

#### Module Network
- [*Neural Module Networks*](https://arxiv.org/abs/1511.02799)
- [Learning to Compose Neural Networks for Question Answering](https://arxiv.org/abs/1601.01705)
- [Modeling Relationships in Referential Expressions with Compositional Modular Networks](https://arxiv.org/abs/1611.09978)
- [*Learning to Reason: End-to-End Module Networks for Visual Question Answering*](https://arxiv.org/abs/1704.05526)

#### Neuro-Architecture Search

- [*DARTS: Differentiable Architecture Search(Differentiable)*](https://arxiv.org/abs/1806.09055)

#### Hierarchical RLs

- [*Hierarchical Reinforcement Learning for Zero-shotDeep Variational Information Bottleneck]
Generalization with Subtask Dependencies*](https://arxiv.org/pdf/1807.07665.pdf)

- [*Zero-Shot Task Generalization with Multi-Task Deep Reinforcement Learning*](https://arxiv.org/pdf/1706.05064.pdf)

- [*Learning Goal Embeddings via Self-Play for
Hierarchical Reinforcement Learning*](https://arxiv.org/pdf/1811.09083.pdf)

#### Representation Learning 

- [Deep Variational Information Bottleneck](https://arxiv.org/abs/1612.00410)
- [Machine Theory of Mind](https://arxiv.org/abs/1802.07740)
- [Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow](https://arxiv.org/pdf/1810.00821.pdf)
