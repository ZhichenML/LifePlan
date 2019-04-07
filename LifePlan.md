# Life Plan

## Research List
- [ ] [Sainbayar Sukhbaatar](https://cims.nyu.edu/~sainbar/)
- [ ] [CS 294-112 Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/)


- [ ] <details><summary>[Learning Goal Embeddings via Self-Play for Hierarchical Reinforcement Learning](https://arxiv.org/pdf/1811.09083.pdf) </summary> 
  it says goal space embedding helps learning with sparse reward and enhance exploration. The goal space is learned by self-play and pretraining.
  </details>

- [ ] [Causal Reasoning from Meta-reinforcement Learning](https://arxiv.org/pdf/1901.08162.pdf)
- [ ] [Analyzing and Improving Representations with the Soft Nearest Neighbor Loss](https://arxiv.org/pdf/1902.01889.pdf)
- [ ] [Hybrid Models with Deep and Invertible Features](https://arxiv.org/pdf/1902.02767.pdf)
- [ ] [A Lyapunov-based Approach to Safe Reinforcement Learning](https://arxiv.org/pdf/1805.07708.pdf)
- [ ] [Toward high-performance, memory-efficient, and fast reinforcement learning—Lessons from decision neuroscience]()




## Web interface learner

subtask embedding + attention for relation = subtask graph

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

#### Relational Module Network/ Compositionality
- [x] [*Neural Module Networks*](https://arxiv.org/abs/1511.02799)
- [x] [Learning to Compose Neural Networks for Question Answering](https://arxiv.org/abs/1601.01705)
- [x] [Modeling Relationships in Referential Expressions with Compositional Modular Networks](https://arxiv.org/abs/1611.09978)
- [x] [*Learning to Reason: End-to-End Module Networks for Visual Question Answering*](https://arxiv.org/abs/1704.05526)
- [ ] [Modular Multitask Reinforcement Learning with Policy Sketches](https://arxiv.org/abs/1611.01796)
- [ ] [*Neural Task Programming: Learning to Generalize Across Hierarchical Tasks*](https://arxiv.org/abs/1710.01813)
- [x] [*Neural Task Graphs: Generalizing to Unseen Tasks from a Single Video Demonstration*](https://arxiv.org/abs/1807.03480)
  Our solution to scaling visual imitation to complex tasks is to explicitly incorporate compositionality in both the task and the policy representation.
  use imitation and propagation for graph transition
  neural graph generation and execution

- [x] [Deep reinforcement learning with relational inductive biases \| OpenReview](https://openreview.net/forum?id=HkxaFoC9KQ)
  CNN extract entities + use attention for transition
- [x] [*Relational recurrent neural networks*](https://papers.nips.cc/paper/7960-relational-recurrent-neural-networks.pdf)
  transformer applies to memory slots
- [ ] [*Neural Relational Inference for Interacting Systems*](https://arxiv.org/pdf/1802.04687.pdf)
- [ ] [*VAIN: Attentional Multi-agent Predictive Modeling*](https://arxiv.org/abs/1706.06122)
- [ ] [IQ of Neural Networks](https://arxiv.org/abs/1710.01692)
- [ ] [*Neural Message Passing for Quantum Chemistry*](https://arxiv.org/abs/1704.01212)
- [ ] [[1612.00222] Interaction Networks for Learning about Objects, Relations and Physics](https://arxiv.org/abs/1612.00222)
- [ ] [[1702.05068] Discovering objects and their relations from entangled scene representations](https://arxiv.org/abs/1702.05068)
- [ ] [[1711.07971] Non-local Neural Networks](https://arxiv.org/abs/1711.07971)
- [ ] [[1806.02919] Non-Local Recurrent Network for Image Restoration](https://arxiv.org/abs/1806.02919)
- [ ] [[1805.09354] Working Memory Networks: Augmenting Memory Networks with a Relational Reasoning Module](https://arxiv.org/abs/1805.09354)
- [ ] [Galileo: Perceiving Physical Object Properties by Integrating a Physics Engine with Deep Learning](https://papers.nips.cc/paper/5780-galileo-perceiving-physical-object-properties-by-integrating-a-physics-engine-with-deep-learning)
- [ ] [[1706.01433] Visual Interaction Networks](https://arxiv.org/abs/1706.01433)
- [ ] [Deep Sets](https://papers.nips.cc/paper/6931-deep-sets)
- [ ] [Relational Neural Expectation Maximization: Unsupervised Discovery of Objects and their Interactions](https://arxiv.org/pdf/1802.10353.pdf)

- [x] [Learning Goal Embeddings via Self-Play for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1811.09083)
- [ ] [Graph Attention Networks \| OpenReview](https://openreview.net/forum?id=rJXMpikCZ)
- [ ] [Neural Relational Inference for Interacting Systems](https://arxiv.org/pdf/1802.04687.pdf)

#### memory machine
- [ ] [[1410.5401] Neural Turing Machines](https://arxiv.org/abs/1410.5401)
- [ ] [Hybrid computing using a neural network with dynamic external memoryw](https://www.nature.com/articles/nature20101.pdf)
- [ ] [Meta-Learning with Memory-Augmented Neural Networks](http://proceedings.mlr.press/v48/santoro16.pdf)
- [ ] [End-To-End Memory Networks](https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)
- [ ] [[1706.06383] Programmable Agents](https://arxiv.org/abs/1706.06383)


#### Graph neural network 
- [ ] [Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261)
- [ ] [Non-local Neural Networks](https://arxiv.org/abs/1711.07971)
- [ ] [A simple neural network module for relational reasoning](http://papers.nips.cc/paper/7082-a-simple-neural-network-module-for-relational-reasoning)
  CNN extract entities + entities pair and query tuple for output prediceiton
- [ ] [[1511.05493] Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493)
- [ ] 




- [ ] [*Autonomously Constructing Hierarchical Task Networks for Planning and Human-Robot Collaboration*](http://bradhayes.info/papers/icra16.pdf)
- [ ] [Learning graphical state transitions](https://openreview.net/pdf?id=HJ0NvFzxl)
- [ ] [Discovering objects and their relations from entangled scene representations](https://arxiv.org/abs/1702.05068)
- [ ] [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493)

#### Attention
- [x] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [ ] [Machine Theory of Mind](https://arxiv.org/abs/1802.07740)
- [ ] [[1703.07326] One-Shot Imitation Learning](https://arxiv.org/abs/1703.07326)
- [ ] [Can Active Memory Replace Attention?](https://papers.nips.cc/paper/6295-can-active-memory-replace-attention.pdf)
- [ ] [VAIN: Attentional Multi-agent Predictive Modeling](https://papers.nips.cc/paper/6863-vain-attentional-multi-agent-predictive-modeling.pdf)
- [ ] [[1606.01933] A Decomposable Attention Model for Natural Language Inference](https://arxiv.org/abs/1606.01933)



#### Neuro-Architecture Search

- [ ] [*DARTS: Differentiable Architecture Search(Differentiable)*](https://arxiv.org/abs/1806.09055)
- [ ] [Efficient Neural Architecture Search via Parameter Sharing](http://proceedings.mlr.press/v80/pham18a/pham18a.pdf)
- [ ] [Searching for Efficient Multi-Scale Architectures for Dense Image Prediction](https://papers.nips.cc/paper/8087-searching-for-efficient-multi-scale-architectures-for-dense-image-prediction.pdf)
- [ ] [Progressive Neural Architecture Search](https://arxiv.org/pdf/1712.00559.pdf)
- [Differentiable Neural Network Architecture Search \| OpenReview](https://openreview.net/forum?id=BJ-MRKkwG)
- [Learning Transferable Architectures for Scalable Image Recognition](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zoph_Learning_Transferable_Architectures_CVPR_2018_paper.pdf)
- [Neural architecture search with reinforcement learning](https://arxiv.org/pdf/1611.01578.pdf)
- [efficient architecture search by network transformation](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16755/16568)
- [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/pdf/1807.11626.pdf)
- [Designing Neural Network Architectures using Reinforcement Learning](https://arxiv.org/pdf/1611.02167.pdf)
- [Practical Block-wise Neural Network Architecture Generation](https://arxiv.org/pdf/1708.05552.pdf)
- [ ] [Auto-DeepLab:Hierarchical Neural Architecture Search for Semantic Image Segmentation](https://arxiv.org/pdf/1901.02985.pdf)

#### Hierarchical RLs

- [ ] [*Hierarchical Reinforcement Learning for Zero-shot
  Generalization with Subtask Dependencies*](https://arxiv.org/pdf/1807.07665.pdf)

- [ ] [*Zero-Shot Task Generalization with Multi-Task Deep Reinforcement Learning*](https://arxiv.org/pdf/1706.05064.pdf)

- [ ] [*Learning Goal Embeddings via Self-Play for Hierarchical Reinforcement Learning*](https://arxiv.org/pdf/1811.09083.pdf)
- [ ] [[1703.01161] FeUdal Networks for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1703.01161)
- [x] [[1901.08492] Feudal Multi-Agent Hierarchies for Cooperative Reinforcement Learning](https://arxiv.org/abs/1901.08492)
  the manager is not trained to output reward
- [ ] [Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation](https://papers.nips.cc/paper/6233-hierarchical-deep-reinforcement-learning-integrating-temporal-abstraction-and-intrinsic-motivation)
- [ ] [[1609.05140] The Option-Critic Architecture](https://arxiv.org/abs/1609.05140)
- [ ] [[1704.03012] Stochastic Neural Networks for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1704.03012)





#### Optimization
- [ ] [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980.pdf)

#### imitation learning
- [ ] [Reinforced Cross-Modal Matching and Self-Supervised Imitation Learning for Vision-Language Navigation](https://arxiv.org/abs/1811.10092) 
- [ ] [One-Shot Imitation Learning](https://arxiv.org/abs/1703.07326)
- [ ] [(http://proceedings.mlr.press/v78/finn17a/finn17a.pdf)](http://proceedings.mlr.press/v78/finn17a/finn17a.pdf)
- [ ] [ Imitation from Observation: Learning to Imitate Behaviors from Raw Video via Context Translation](https://arxiv.org/abs/1707.03374)
- [ ] [[1803.01840] TACO: Learning Task Decomposition via Temporal Alignment for Control](https://arxiv.org/abs/1803.01840)
- [x] [Compositional Imitation Learning: Explaining and executing one task at a time](https://arxiv.org/pdf/1812.01483.pdf)

#### Few-shot learning
- [ ] [Matching Networks for One Shot Learning](https://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf)
- [ ] [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)

#### Representation Learning
- [x] [Learning Disentangled Joint Continuous and Discrete Representations](https://arxiv.org/pdf/1804.00104.pdf)
  use gumble softmax distribution for categorical variable, unsupervised learning for digit classification by looking at only discrete lantent distributions
- [x] [Deep Variational Information Bottleneck \| OpenReview](https://openreview.net/forum?id=HyxQzBceg)
  ignore the input information and output the target information with stochastic network (sample multiple latent variable for average output)
- [ ] [DARLA: Improving Zero-Shot Transfer in Reinforcement Learning](https://arxiv.org/pdf/1707.08475.pdf)
- [ ] [Regularizing neural networks by penalizing confident output predictions](https://openreview.net/pdf?id=HyhbYrGYe)

- [ ] [SCAN: Learning Hierarchical Compositional Visual Concepts](https://arxiv.org/abs/1707.03389)
- [ ] [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657)
- [ ] [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework \| OpenReview](https://openreview.net/forum?id=Sy2fzU9gl)
- [ ] [Understanding disentangling in $\beta$-VAE](https://arxiv.org/abs/1804.03599)
- [ ] [PixelGAN Autoencoders](https://papers.nips.cc/paper/6793-pixelgan-autoencoders.pdf)
- [ ] [Disentangling by Factorising](https://arxiv.org/abs/1802.05983)
- [ ] [The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables \| OpenReview](https://openreview.net/forum?id=S1jE5L5gl)
- [ ] [Categorical Reparameterization with Gumbel-Softmax \| OpenReview](https://openreview.net/forum?id=rkE3y85ee)
- [ ] [Isolating Sources of Disentanglement in Variational Autoencoders \| OpenReview](https://openreview.net/forum?id=BJdMRoCIf)

#### Reinforcement Learning
- [ ] [Unsupervised Predictive Memory in a Goal-Directed Agent](https://arxiv.org/pdf/1803.10760.pdf)
- [ ] [Variational Information Maximisation for Intrinsically Motivated Reinforcement Learning](https://arxiv.org/abs/1509.08731)
- [ ] [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](http://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf)
- [ ] [[1511.07401] MazeBase: A Sandbox for Learning from Games](https://arxiv.org/abs/1511.07401)
  environment with ordered goals and if-then styple goals
- [ ] [Imagination-Augmented Agents for Deep Reinforcement Learning](https://papers.nips.cc/paper/7152-imagination-augmented-agents-for-deep-reinforcement-learning.pdf)
- [x] [Value Iteration Networks](https://arxiv.org/pdf/1602.02867.pdf)
  planing turns out to be CNN, combined with model-free value iteration of reactive policy, to improve generalization and efficiency
  *learn to plan*
  does it abandon V when updating Q

#### Deep Learning
- [ ] [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [ ] [Recurrent Ladder Networks](https://papers.nips.cc/paper/7182-recurrent-ladder-networks.pdf)

#### Multiagent learning
- [ ] [Neural MMO: A Massively Multiagent Game Environment for Training and Evaluating Intelligent Agents](https://s3-us-west-2.amazonaws.com/openai-assets/neural-mmo/neural-mmo-arxiv.pdf)
- [ ] [Relational Forward Models for Multi-Agent Learning](https://openreview.net/pdf?id=rJlEojAqFm)
- [ ] [[1808.04355] Large-Scale Study of Curiosity-Driven Learning](https://arxiv.org/abs/1808.04355)


#### discrete variables
- [ ] [[1611.00712] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables](https://arxiv.org/abs/1611.00712)
- [ ] [[1611.01144] Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144)

### researcher
- [Xin (Eric) Wang](http://www.cs.ucsb.edu/~xwang/)
- [Computer Laboratory: Petar Veličković](https://www.cl.cam.ac.uk/~pv273/)
- [Yan (Rocky) Duan](http://rockyduan.com/)



### progress

our contribution is 
1. discreate goal embedding + information bottleneck principle 
2. understand relationship between goals in order to train end-2-end and zero shot

### Implementation list
#### platforms
- [Spinning up openai](https://spinningup.openai.com/en/latest/)
- [RLlib: Scalable Reinforcement Learning](https://ray.readthedocs.io/en/latest/rllib.html)
- [Pycolab DeepMind](https://github.com/deepmind/pycolab)
- [Graphnet DeepMind](https://github.com/deepmind/graph_nets)

#### VAE
- [Learning Disentangled Joint Continuous and Discrete Representations](https://arxiv.org/pdf/1804.00104.pdf
- <https://openreview.net/forum?id=BJdMRoCIf>)


#### tensorflow
- [ ] [MIT Deep Learning](https://deeplearning.mit.edu/)

#### memory augmented neural networks
- [ ] [[1410.5401] Neural Turing Machines](https://arxiv.org/abs/1410.5401)
- [ ] [Hybrid computing using a neural network with dynamic external memoryw](https://www.nature.com/articles/nature20101.pdf)
- [ ] [Meta-Learning with Memory-Augmented Neural Networks](http://proceedings.mlr.press/v48/santoro16.pdf)
- [ ] [End-To-End Memory Networks](https://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)


#### Grounded neural network architecture
- [x] [Grounded Language Learning: Where Robotics and NLP Meet](https://www.ijcai.org/proceedings/2018/0810.pdf)
- [x] [Grounded Language Learning in a Simulated 3D World](https://arxiv.org/pdf/1706.06551.pdf)
- [x] [Grounded Recurrent Neural Networks](https://arxiv.org/pdf/1705.08557.pdf)
- [x] [Towards Grounding Conceptual Spaces in Neural Representations](http://ceur-ws.org/Vol-2003/NeSy17_paper1.pdf)
- [ ] [Neural Architecture Optimization](https://papers.nips.cc/paper/8007-neural-architecture-optimization.pdf)
- [ ] [[1811.01567v1] You Only Search Once: Single Shot Neural Architecture Search via Direct Sparse Optimization](https://arxiv.org/abs/1811.01567v1)
- [ ] [DOM-Q-NET: Grounded RL on Structured Language \| OpenReview](https://openreview.net/forum?id=HJgd1nAqFX)
- [ ] [Evolutionary Stochastic Gradient Descent for Optimization of Deep Neural Networks](https://papers.nips.cc/paper/7844-evolutionary-stochastic-gradient-descent-for-optimization-of-deep-neural-networks.pdf)
- [ ] [A Neural Network for Decision Making in Real-Time Heuristic Search](https://aaai.org/ocs/index.php/SOCS/SOCS18/paper/viewFile/17976/17117)
- [ ] [MONAS: Multi-Objective Neural Architecture Search](https://arxiv.org/pdf/1806.10332.pdf)
- [ ] [Progressive Neural Architecture Search](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.pdf)
- [ ] [GitHub - markdtw/awesome-architecture-search: A curated list of awesome architecture search resources](https://github.com/markdtw/awesome-architecture-search)
- [ ] [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/pdf/1812.00332.pdf)
- [ ] [[1803.06744] Fast Neural Architecture Construction using EnvelopeNets](https://arxiv.org/abs/1803.06744)
- [ ] [[1812.09584] Bayesian Meta-network Architecture Learning](https://arxiv.org/abs/1812.09584)
- [ ] [Constructing Deep Neural Networks by Bayesian
  Network Structure Learning](https://papers.nips.cc/paper/7568-constructing-deep-neural-networks-by-bayesian-network-structure-learning.pdf)
- [ ] [Bayesian Model-Agnostic Meta-Learning](https://papers.nips.cc/paper/7963-bayesian-model-agnostic-meta-learning.pdf)
- [ ] [AlphaX: eXploring Neural Architectures with Deep Neural Networks and Monte Carlo Tree Search](https://arxiv.org/pdf/1805.07440.pdf)
- [ ] [Three-Head Neural Network Architecture for Monte Carlo Tree Search](https://www.ijcai.org/proceedings/2018/0523.pdf)
- [ ] [Learning to Search with MCTSnets](https://arxiv.org/pdf/1802.04697v1.pdf)
- [ ] [Thinking Fast and Slow with Deep Learning and Tree Search](https://papers.nips.cc/paper/7120-thinking-fast-and-slow-with-deep-learning-and-tree-search.pdf) imitation learning by deep models and tree search algorithms bootstrap mutally. tree search (t) generates dataset (s,a) based s of imitation policy's (t) s. Then the trained imitation policy (t+1) is used to bias the search preference of tree search algorithm (t+1). In this way, the fast IL learner and slow planning learner finally generates fast learner.
- [ ] [Learning to Search with MCTSnets](https://arxiv.org/pdf/1802.04697v1.pdf) a neural network trained with embedding network, simulation policy, backup network, and readout network for output decision. So composed network is used to learn the behavior of the MCTS rollouts.
- [ ] [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/pdf/1712.01815.pdf)
- [ ] [Mastering the game of Go without Human Knowledge \| DeepMind](https://deepmind.com/research/publications/mastering-game-go-without-human-knowledge/)
- [ ] [Mastering the game of Go with deep neural networks and tree search](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
- [ ] [On Monte Carlo Tree Search and Reinforcement Learning](https://pdfs.semanticscholar.org/3d78/317f8aaccaeb7851507f5256fdbc5d7a6b91.pdf)
- [ ] [AlphaGoZero · Depth First Learning](https://www.depthfirstlearning.com/2018/AlphaGoZero)

two modal:language channel plus 3d simulation environment. language prediction and temporal AutoEncoder helps agent learn the composition.


automl mds+evolution by playing interaction, grounded neural network structure, darts
autoRL 

relational reasoning by compositional structure with differentiable optimization

grounded means a mapping fron language to its physical countpart, means give (something abstract) a firm theoretical or practical basis, means the ability to find a reference for the output, means interpretability. ---> interpretable autoML


The search space covers cnn and rnn, after evolution, the network may converge to typical cnn or rnn, or may converge to a novel network structure. It can be tested on a rnn or cnn required task. The difference between grounded NAS (non-predefined cnn or rnn structure) and autoML (predefined cnn or rnn structure) is the search space.

how to connect? markov modeling units?