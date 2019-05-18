### AutoRL

1. [Optimal Brain Damage, by Yann LeCun et al. in NIPS 1990](https://papers.nips.cc/paper/250-optimal-brain-damage.pdf)

2. [Second order derivatives for network pruning: Optimal Brain Surgeon, by Babak Hassibi and David G. Stork in NIPS 1993](https://papers.nips.cc/paper/647-second-order-derivatives-for-network-pruning-optimal-brain-surgeon.pdf) *Network weights pruning*

3. [The symbol grounding problem, by Stevan Harnad in Physica D 1990](http://www.cs.ox.ac.uk/activities/ieg/e-library/sources/harnad90_sgproblem.pdf): symbolism and connectionism for cognition. Symbols has intrinsic meanings, so it need grounding, despite it ability in composition to higher order symbols. symbolism (behaviourism) and connectionism (cognitism). Symbol grounding means finding the intrinsic meaning of symbols from our first language or connecting to the world in the right way [[1]](http://www.cs.ox.ac.uk/activities/ieg/e-library/sources/harnad90_sgproblem.pdf). However, the dynamic patterns in the activations of connections in a multi-layerd nodes are hardly symbols. 

   Symbols. invariant feature that discriminate and identify them. Compository, symbols can be combined and recombined rulefully to form higher-order grounded symbols.  One formal test: is it a symbol system? One behavioral test: can it discriminate, identify, and describe what the symbol refer?

   connectionism. Ground learning, but not composible.

   

   

   ------

   ​									NeurEvolution

   ------

4. [Designing neural networks using genetic algorithms, by Geoffrey F. Miller et al. in ICGA 1989](https://static1.squarespace.com/static/58e2a71bf7e0ab3ba886cea3/t/5909113c1b631b40f8137956/1493766462349/1989+neural+networks.pdf): one of the earliest automatic architecture designing method

5. [Evolving Neural Networks through Augmenting Topologies, by Kenneth O. Stanley et al. in Evolutionary Computation 2002](https://dl.acm.org/citation.cfm?id=638554)

6. [Large-Scale Evolution of Image Classifiers, by Esteban Real and Sherry Moore et al. in ICML 2017](https://arxiv.org/abs/1703.01041)

7. [NEMO: Neuro-Evolution with Multiobjective Optimization of Deep Neural Network for Speed and Accuracy, by Ye-Hoon Kim et al. in ICML 2017](https://pdfs.semanticscholar.org/0a9c/6947a0b6f79526e537cb83925ef60df674e8.pdf)

   ------

   ​								Bayesian optimisation

   -----

8. [Random search for hyperparameter optimization, by James Bergstra and Yoshua Bengio in JMLR 2012](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf): Bayesian optimisation

9. [Speeding up automatic hyperparameter optimization of deep neural networks by extrapolation of learning curves by Tobias Domhan et al. in IJCAI 2015](http://ml.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf)

10. [Towards Automatically-Tuned Neural Networks, by Hector Mendoza and Aaron Klein et al. in Workshop on Automatic Machine Learning 2016](https://ml.informatik.uni-freiburg.de/papers/16-AUTOML-AutoNet.pdf)

11. [Learning Curve Prediction with Bayesian Neural Networks, by Aaron Klein et al. in ICLR 2017](https://openreview.net/forum?id=S11KBYclx)

12. [BayesNAS: A Bayesian Approach for Neural Architecture Search, by Hongpeng Zhou et al. in ICML 2019](https://arxiv.org/abs/1905.04919)

13. [Neural Architecture Search with Bayesian Optimisation and Optimal Transport, by Eric P. Xing et al. in NIPS 2018](http://papers.nips.cc/paper/7472-neural-architecture-search-with-bayesian-optimisation-and-optimal-transport): it proposes a distance metric for neural networks based on optimal transportation. With this distance metric, a kernel is defined to be used in bayesian optimizaiton.

    ------

14. [*Network In Network*, by Min Lin & Shuicheng Yan et al. in ICLR, 2014](https://openreview.net/forum?id=ylE6yojDR5yqX) <!--"includes micro multi-layer
    perceptrons into the filters of convolutional layers to extract more complicated features."[DenseNet]--> <u>it implements nonlinear feature mapping compared with linear feature mapping of CNNs. The first layer is CNN feature mapping, the second layer performs nonlinear feature combination, and the third layer performs 1x1 convolution (spatial pooling). The success of NIN is due to MLP structure for convolution and pooling, and the global average pooling, which has the same number of channel as the number of class. This dedicated structure is responsible for the performance improvement.</u>

    > "includes micro multi-layer perceptrons into the filters of convolutional layers to extract more complicated features." [8]

15. [Deeply-Supervised Nets, by Chen-Yu Lee et al. in AISTATS, 2015](http://proceedings.mlr.press/v38/lee15a.pdf) <u>Add discriminative objectives for the intermediatiate layers.</u>

16. [Learning both weights and connections for efficient neural network, by Song Han et al. in NIPS 2015](https://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network) *Network Pruning by multi-step training*

17. [Training Very Deep Networks, by Rupesh Kumar Srivastava et al. in NIPS, 2015](https://papers.nips.cc/paper/5850-training-very-deep-networks.pdf): *Highway network*. The input is linearly combined with feature mapping of a plain neural network by a tranformation gate.

    $y=H(x,W_{h})$

    $y=H(x,W_{h})*T(x,W_{T})+x*(1-T(x,W_{T}))$

    The *transform* gate controls the information flow.

18. [A Flexible Approach to Automated RNN Architecture Generation, by Martin Schrimpf et al. in arXiv 2016](https://arxiv.org/pdf/1712.07316.pdf)

      

    ------

    ​								Network Transformation

    ------

19. [Net2Net: Accelerating Learning via Knowledge Transfer, by Tianqi Chen, Iran Goodfellow and Jonathon Shlens in ICLR 2015](https://arxiv.org/abs/1511.05641): Network function-preserving

20. [Learning both Weights and Connections for Efficient Neural Network, by Song Han et al. in NIPS 2015](https://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network)

21. [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149):*Network Pruning*

      

      

    ------

    

22. [Going Deeper with Convolutions, by Christian Szegedy et al. in CVPR, 2015](https://arxiv.org/abs/1409.4842): *Wider Networks*<!--Wider GoogLeNet with an inception module which concatenates feature-maps produced by filters of different sizes--> 

23. [Rethinking the Inception Architecture for Computer Vision, by Christian Szegedy et al. in CVPR, 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf): *Inception*

    ------

24. [Semi-supervised Learning with Ladder Networks](http://papers.nips.cc/paper/5947-semi-supervised-learning-with-ladder-ne): *Ladder Networks*

25. [Deconstructing the Ladder Network Architecture](http://proceedings.mlr.press/v48/pezeshki16.pdf)

    ------

26. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, by Sergey Ioffe and Christian Szegedy in ICML 2015](http://proceedings.mlr.press/v37/ioffe15.pdf): *Batch Normalization*

      

27. [***Deep Residual Learning for Image Recognition***, by Kaiming He et al. in CVPR, 2016 ](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf): <u>*ResNet*, H(x) = F(x) + x, the rest of the network is refered to approximate F(x) = H(x) - x, by adding input x to network blocks. Every block only learns the residual $F(x)=H(x)-x​$</u>

28. [Identity Mappings in Deep Residual Networks，by Kaiming He et al. in ECCV 2016](https://arxiv.org/abs/1603.05027)

29. [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, by Andrew G. Howard et al. in ArXiv 2017](https://arxiv.org/abs/1704.04861) *hand-tuning low computational cost*

30. [Deep Networks with Stochastic Depth, by Gao Huang et al. in ECCV, 2016](https://arxiv.org/abs/1603.09382): *Stochastic Depth Network*

31. [Identity Mappings in Deep Residual Networks, by Kaiming He et al. in ECCV, 2016](https://arxiv.org/abs/1603.05027): *Pre-activation ResNet*

32. [Densely Connected Convolutional Networks, by Gao Huang et al. in CVPR, 2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf): *DenseNet*. Concate all the proceeding activations to the next layer, forming $L(L+1)/2$ connections. It has implicit deep supervision and help avoid vanishing gradients.

33. [***Exploring Randomly Wired Neural Networks for Image Recognition***, by Saining Xie et al. in Arxiv, 2019](https://arxiv.org/pdf/1904.01569.pdf): <u>*RandomWired Network*, Thestrategy to generate the network, or network generator stated in the paper, introduce prior bias to the generated network and limit the network search space to a subspce. To circumvent the prior bias, this paper tries to explot random graphs as network generator and transform the graph into neural networks. The node operation is predifined and universal. It is similar to our idea of letting the network emerge itself, rather than pre-define any network types. </u> 

    ------

    

34. [*Neural architecture search with reinforcement learning* by Barret Zoph et al. in ICLR 2017](https://openreview.net/pdf?id=r1Ue8Hcxg): *RL-NAS*. This paper proposes to compose network structure by using a recurrent neural network as the controller, which outputs a sequence of hyper parameters. The controller network makes use of auto-regressive nature of hyperparameter prediction conditioned on the previous search. Its parameter is trained by policy gradient to maxmize the validation accuracy.  Its performance is comparable with the best Dense-net performance on CIFAR-10 image classification dataset, and excels on Penn Treebank language modeling compared with variational LSTM. It also shows significant improvement with a RL trained controller compared with random generated networks.

35. [Designing neural network architectures using reinforcement learning, by Bowen Baker et al. in ICLR 2017](https://openreview.net/pdf?id=S1c2cvqee)

36. [Learning Transferable Architectures for Scalable Image Recognition, by Barret Zoph et al. in CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zoph_Learning_Transferable_Architectures_CVPR_2018_paper.pdf)：block search space. The paper proposes to search for effictive architectures on small proxy dataset and then transfer the learned architecture to large dataset. The transferability is achieved via block-wise search space, which makes the architecture agnostic to the size of intput data and depth of the whole network. The search space is different from the whole network space, and thus has more ability to generalise to other problems.

    ------

    ​											RL

    ------

37. [Practical Block-wise Neural Network Architecture Generation, by Zhao Zhong et al. in CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhong_Practical_Block-Wise_Neural_CVPR_2018_paper.pdf) *transferable blocks* 

38. [Accelerating Neural Architecture Search using Performance Prediction, by Bowen Baker et al. in ICLR 2018](https://openreview.net/pdf?id=BJypUGZ0Z)

39. [BlockQNN: Efficient Block-wise Neural Network Architecture Generation, by Zhao Zhong et al. in arXiv 2018](https://arxiv.org/abs/1808.05584)

40. [Searching for activation functions, by Prajit Ramachandran and Barret Zoph et al. in ArXiv 2017](https://arxiv.org/abs/1710.05941)

41. [Deeparchitect: Automatically designing and training deep architectures, by Renato Negrinho and Geoff Gordon in ICLR 2017](https://openreview.net/pdf?id=rkTBjG-AZ): *Monte Carlo Tree Search*

42. [Simple and efficient architecture search for Convolutional Neural Networks, by Thomas Elsken et al. in ICLR 2017](https://openreview.net/forum?id=SySaJ0xCZ):*Network Morphisms*

43. [Categorical Reparameterization with Gumbel-Softmax, by Eric Jang et al. in ICLR 2017](https://openreview.net/forum?id=rkE3y85ee)

44. [Neural Optimizer Search with Reinforcement Learning, by Irwan Bello and Barret Zoph et al. in ICML 2017](http://proceedings.mlr.press/v70/bello17a/bello17a.pdf)

45. [Mixed Precision Quantization of ConvNets via Differentiable Neural Architecture Search](https://openreview.net/forum?id=BJGVX3CqYm)

46. [N2n learning: Network to network compression via policy gradient reinforcement learning, by Anubhav Ashok et al. in ICLR 2018](https://openreview.net/pdf?id=B1hcZZ-AW)

47. [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design by Ningning Ma and Jian Sun et al. in ECCV 2018](https://arxiv.org/pdf/1807.11164.pdf): *CNN design*

48. [Searching for Efficient Multi-Scale Architectures for Dense Image Prediction, by Liang-Chieh Chen and Barret Zoph et al. in NIPS 2018](http://papers.nips.cc/paper/8087-searching-for-efficient-multi-scale-architectures-for-dense-image-prediction.pdf)

49. [Progressive Neural Architecture Search, by Chenxi Liu and Barret Zoph et al. in ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.pdf): micro

50. [Efficient Architecture Search by Network Transformation, by Han Cai et al. in AAAI 2018](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16755/16568): reuse network, combined with net2wider and net2deeper transformations, trained by policy gradient.

51. [Hierarchical Representations for Efficient Architecture Search, by Hanxiao Liu et al. in ICLR 2018](https://openreview.net/forum?id=BJQRKzbA-)*Neuro-evolution & transferable block*

    ------

52. [SMASH: One-Shot Model Architecture Search through HyperNetworks, by Andrew Brock et al. in NIPS 2017](http://metalearning.ml/2017/papers/metalearn17_brock.pdf):*hyper-net generates network parameters*, net-level

53. [Faster discovery of neural architectures by searching for paths in a large model, by Hieu Pham et al. in ICLR 2018](https://openreview.net/forum?id=ByQZjx-0-&noteId=rJWmCYxyM) * Efficient Neural Architecture Search*

54. [Population Based Training of Neural Networks, by Max Jaderberg et al. in Arxiv 2017](https://arxiv.org/abs/1711.09846): *parameter sharing*, it proposes a population based hyperparameter optimization algorithm in spirit of genetic algorithms. The algorithm maintains a population of hyperparameters, parameters, and evaluation. It eposodically replaces poor performed ones by expoiting good individual's hyperparameter and parameter, before applying exploring operator to refine the hyperparameter.

55. [Neural Architecture Optimization by Tieyan Liu et al. in NIPS 2018](https://papers.nips.cc/paper/8007-neural-architecture-optimization.pdf): *SGD-NAS*, it encodes the neura architecture into a continuous space and improves it by predicting the performace on dev set, and after that decodes the resulting continuous representation into a neural architecture. The training is to adjust the encoder LSTM and decoder LSTM to performance prediction and architecture reconstruction. When inference for better architecture, the encoding is adjusted based on the gradients of performance predictor.

56. [MorphNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks, by Ariel Gordon et al. in CVPR 2018](https://arxiv.org/abs/1711.06798): *Efficient Flop aware optimizaiton*, two-step training: resource constrained training to prone the network, retraining without resource constraint

57. [Learning Time/Memory-Efficient Deep Architectures with Budgeted Super Networks, by Tom Veniat et al. in CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Veniat_Learning_TimeMemory-Efficient_Deep_CVPR_2018_paper.pdf):*net-level*

58. [*DARTS: Differentiable Architecture Search* by Hanxiao Liu et al. in ICLR 2019](https://openreview.net/forum?id=S1eYHoC5FX): it initialize an overparameterised network by putting a set of candidate structures together and uses their weighted output as prediction. The bilevlel training alternates between adjusting network weights on the training dataset and adjusting the candidate weights on the validation dataset. Since a lot of candicates are trained together, it needs much memory. Also, the computation graph shares 

    ------

59. [Efficient Neural Architecture Search via Parameter Sharing, by Barret Zoph et al. in ICML 2018](http://proceedings.mlr.press/v80/pham18a/pham18a.pdf)*Weight sharing*, it exploits weight sharing among candidate network to facilitate efficiency. The insight is implemented by restricting the search space to a subset of computational graph, and the weights within a repeatable operation nodes are shared across generated networks. The block weights and RL trained RNN controller weights alternatively.

60. [*Evaluating the Search Phase of Neural Architecture Search*, by Christian Sciuto et al. in ArXiv 2019](https://arxiv.org/pdf/1902.08142.pdf): *Random search NAS*, it empirically demonstrates that weight sharing can negatively impact the architecture search phase, that ``the weights of the best architecture are biased by their use in other candidates'', and inturn bias the ranking of the best architecture.. Random sampled architectures are shown to has a competitive performance with DARTS, NAO, and ENAS.

61. [Random Search and Reproducibility for Neural Architecture Search, by Liam Li et al. in ArXiv 2019](https://arxiv.org/abs/1902.07638)

62. [Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation by Chenxi Liu and Li Fei-Fei et al. in ArXiv 2019](https://arxiv.org/abs/1901.02985v2)

63. [NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection by Golnaz Ghiasi et al. in ArXiv 2019](https://arxiv.org/abs/1904.07392)

64. [Single Path One-Shot Neural Architecture Search with Uniform Sampling by Zichao Guo et al. in ArXiv 2019](https://arxiv.org/abs/1904.00420): *model and structure decouple*

65. [IRLAS: Inverse Reinforcement Learning for Architecture Search, by Minghao Guo et al. in arXiv 2018](https://arxiv.org/pdf/1812.05285.pdf): it proposes to leverage the existing well-performed mannally designed network topology in neural architecture search by inverse reinforcement learning. It is featured by extracting patterns from mannually designed network, and fit a linear projection as the reward. The optimization objective is a combination of accuracy and topology guidance.

66. [Path-Level Network Transformation for Efficient Architecture Search, by Han Cai, Song Han et al. in ICML 2018](http://proceedings.mlr.press/v80/cai18a/cai18a.pdf): it extends layer-wise network transformation to more advanced multi-branch network structure. It introduces branch split and emerge operations, which expoits REINFORCE algorithm to sample and train.

67. [Searching for A Robust Neural Architecture in Four GPU Hours, by Xuanyi Dong and Yi Yang in CVPR 2019](https://xuanyidong.com/publication/cvpr-2019-gradient-based-diff-sampler/)

68. [Understanding and Simplifying One-Shot Architecture Search, by Gabriel Bender et al. in ICML 2018](http://proceedings.mlr.press/v80/bender18a/bender18a.pdf): it combines a set of candidate architectures using weight sharing and trained by regularisation.

69. [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware, by Han Cai et al. in ICLR 2019](https://openreview.net/pdf?id=HylVB3AqYm)：like one-shot architecture search, it also employs an ``cumbersome'' over-parameterised network that are composed of a set of potential architectures, but uses binary connect to determine which one to use in evaluation as well as path-level pruning. The network is trained by gradient, instead of a meta-controller.

    $E_{((s,a)_{t=0}^{T})_{\tau=1}^{N}}[R(s_t, a_t)] = E_{\tau=1}^{N}[\sum_{t=0}^{T} R(s_t, a_t) \pi_{\theta}(a_t|s_t)] ​$

    $\nabla_{\theta}E_{\tau=1}^{N}[\sum_{t=0}^{T}R(s_t, a_t) \pi_{\theta}(a_t|s_t)]\\=E_{\tau=1}^{N}[\sum_{t=0}^{T}R(s_t, a_t) \nabla_{\theta}\pi_{\theta}(a_t|s_t)] \\= E_{\pi_{\theta}, \tau=1}^{N}[\sum_{t=0}^{T}R(s_t, a_t) \nabla_{\theta}log\pi_{\theta}(a_t|s_t)]\\=E_{\pi_{\theta}, \tau=1}^{N}[\sum_{t=0}^{T}\sum_{t'=t}^{T} \gamma^{t'-t}r (s_{t'}, a_{t'}) \nabla_{\theta}log\pi_{\theta}(a_t|s_t)]​$

    When $T=1​$, the REINFORCE algorithm becomes variational optimisation.

    $E_{{x}_{\tau=1}^{N}}[R(x_{\tau})]=\sum_{\tau} R(x_{\tau})\pi_{\theta}(x_{\tau})​$

    $\nabla_{\theta}E_{{x}_{\tau=1}^{N}}[R(x_{\tau})]=E_{{x}_{\tau=1}^{N}}[R(x_{\tau})\nabla_{\theta}log\pi_{\theta}(x_{\tau})]$

70. [Binaryconnect: Training deep neural networks with binary weights during propagations, by Matthieu Courbariaux, Yoshua Bengio and Jean-Pierre David in NIPS 2015](https://papers.nips.cc/paper/5647-binaryconnect-training-deep-neural-networks-with-binary-weights-during-propagations) *BinaryConnect*, it binarize the weights of the neural network in both forward and backward propagation, then update the weights with gradients before binarize them again. The ingredients are noise cancellation by averaing out in SGD and regularization provided by noisy weights.

71. [Graph HyperNetworks for Neural Architecture Search, by Chris Zhang et al. in ICLR 2019](https://arxiv.org/pdf/1810.05749.pdf)

72. [Efficient Multi-objective Neural Architecture Search via Lamarckian Evolution, by Thomas Elsken et al in ICLR 2019](https://arxiv.org/abs/1804.09081): *Network Morphisms operator (network transformation)*, it exploits MOEA for Neural architecture search, it generates new solutions by network morphisms, which initializes newly generated architectures with weights from similar, already trained networks to prevent training all networks from scratch. It samples newly generated architectures according to the evluation difficulty of different objectives, so as to reduce the number of networks needed to be trained. Some common priors of NAS, trivial initial network, Marco architecture a-priori, function preserving network transformation, weights generation/transfering, performance prediction, 

73. [Regularized Evolution for Image Classifier Architecture Search, by Esteban Rea et al. in AAAI 2019](https://arxiv.org/pdf/1802.01548.pdf)

74. [Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search, by Xin Li et al. in CVPR 2019](https://arxiv.org/pdf/1903.03777.pdf): it propose to exploit random search with pruning to find useful trade-off between accuracy and latency. It assumes the latency and accuracy of a network keeps partial order and prunes networks that are less likely to yield better accuracy or lower latency. it's not clear how to cut the prune set from the candidate set and how it works in its early stage.

75. [SNAS: stochastic neural architecture search, by Sirui Xie et al. in ICLR 2019](https://openreview.net/pdf?id=rylqooRqK7): It proves its equivalence to MDP architecture samping in ENAS by unrolling, and proves better unbiased analytical results than DARTS due to the uncombined operations in training. It shows the resource contraints like FLOPs, parameter size, MAC are complementary to distinguish skips, pooling, and none. It uses Gumble softmax for each posible connection in a cell, with efficient unbiased simultaneous training of weights and architectures, and explicit credit assignment.

76. [FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search, by BIchen Wu et al. in CVPR 2019](https://arxiv.org/abs/1812.03443):*Differentiable Gumble Softmax*, it directly search in a layered manner instead of cell structure. Like DARTS, it assigns sampling probabiliteis to a set of operations based on Gumbel Softmax, which is differentiable. The optimization alternates between optimising network parameters and architecture sampling parameters. It considers the lantency which is non-differentiable objective by selecting with Gumble softmax probability and a lookup table recording the lantency of each operation.

77. [Reinforced Evolutionary Neural Architecture Search, by Yukang Chen et al. in CVPR 2019](https://arxiv.org/abs/1808.00193)：*parameter inherit in EA*, Within a fixed framework for generating a network by stacking cells, it utilizes a reinforced learning controller, which receives the LSTM encoding of the two inputs, two operations of a block, and outputs a sampling probability among the inputs and operations. Then decision for mutating the inputs or operation is taken. There are 2B mutation actions for a cell with B blocks.

78. [Searching for A Robust Neural Architecture in Four GPU Hours, by Xuanyi Dong et al. in CVPR 2018](<https://xuanyidong.com/publication/cvpr-2019-gradient-based-diff-sampler/>): *cell, differentiable NAS

    

    ------

     Neural Network for RL

    ------

    

79. [A reinforcement learning algorithm for neural networks with incremental learning ability, by Naoto Shiraga et al. in ICONIP 2002](<http://www2.kobe-u.ac.jp/~ozawasei/pub/iconip02a.pdf>)

80. [Value Iteration Networks, by Aviv Tamar et al. in NIPS 2016](<https://papers.nips.cc/paper/6046-value-iteration-networks.pdf>)

81. [How to Discount Deep Reinforcement Learning: Towards New Dynamic Strategies, by Vincent François-Lavet et al. in arXiv 2016](<https://arxiv.org/pdf/1512.02011.pdf>): *discount fator hyperparameter*

82. [**Dueling Network Architectures for Deep Reinforcement Learning**, by Ziyu Wang et al. in ICML 2017](<http://proceedings.mlr.press/v48/wangf16.pdf>): *dueling architecture*, it separates the state value estimation and action advantage estimation, while sharing a common feature learning module.

83. [Thinking Fast and Slow with Deep Learning and Tree Search, by Thomas Anthony et al. in NIPS 2017](<https://papers.nips.cc/paper/7120-thinking-fast-and-slow-with-deep-learning-and-tree-search>)

84. [Learning to Search with MCTSnets, by David Silver et al. in ICML 2018](<https://arxiv.org/pdf/1802.04697.pdf>) *Monte Carlo Tree Search network*

85. [AlphaX: eXploring Neural Architectures with Deep Neural Networks and Monte Carlo Tree Search, by Linnan Wang et al. in arXiv 2018](<https://arxiv.org/pdf/1805.07440.pdf>)

86. [Reinforcement Learning for Improving Agent Design, by David Ha ](https://arxiv.org/abs/1810.03779)

87. [Joint optimization of robot design and motion parameters using the implicit function theorem]()

88. [Computational co-optimization of design parameters and motion trajectories for robotic systems]()



Non reactive policy

implicit dynamics, planning, memory