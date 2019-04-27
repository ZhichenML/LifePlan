### AutoRL

1. [The symbol grounding problem, by Stevan Harnad in Physica D 1990](http://www.cs.ox.ac.uk/activities/ieg/e-library/sources/harnad90_sgproblem.pdf): symbolism and connectionism for cognition. Symbols has intrinsic meanings, so it need grounding, despite it ability in composition to higher order symbols. symbolism (behaviourism) and connectionism (cognitism). Symbol grounding means finding the intrinsic meaning of symbols from our first language or connecting to the world in the right way [[1]](http://www.cs.ox.ac.uk/activities/ieg/e-library/sources/harnad90_sgproblem.pdf). However, the dynamic patterns in the activations of connections in a multi-layerd nodes are hardly symbols. 

   Symbols. invariant feature that discriminate and identify them. Compository, symbols can be combined and recombined rulefully to form higher-order grounded symbols.  One formal test: is it a symbol system? One behavioral test: can it discriminate, identify, and describe what the symbol refer?

   connectionism. Ground learning, but not composible.

   ------

   ​									NeurEvolution

   ------

2. [Designing neural networks using genetic algorithms, by Geoffrey F. Miller et al. in ICGA 1989](https://static1.squarespace.com/static/58e2a71bf7e0ab3ba886cea3/t/5909113c1b631b40f8137956/1493766462349/1989+neural+networks.pdf): one of the earliest automatic architecture designing method

3. [Evolving Neural Networks through Augmenting Topologies, by Kenneth O. Stanley et al. in Evolutionary Computation 2002](https://dl.acm.org/citation.cfm?id=638554)

4. [Large-Scale Evolution of Image Classifiers, by Esteban Real and Sherry Moore et al. in ICML 2017](https://arxiv.org/abs/1703.01041)

   ------

   ​								Bayesian optimisation

   -----

5. [Random search for hyperparameter optimization, by James Bergstra and Yoshua Bengio in JMLR 2012](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf): Bayesian optimisation

6. [Speeding up automatic hyperparameter optimization of deep neural networks by extrapolation of learning curves by Tobias Domhan et al. in IJCAI 2015](http://ml.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf)

7. [Towards Automatically-Tuned Neural Networks, by Hector Mendoza and Aaron Klein et al. in Workshop on Automatic Machine Learning 2016](https://ml.informatik.uni-freiburg.de/papers/16-AUTOML-AutoNet.pdf)

8. [Learning Curve Prediction with Bayesian Neural Networks, by Aaron Klein et al. in ICLR 2017](https://openreview.net/forum?id=S11KBYclx)

   ------

9. [*Network In Network*, by Min Lin & Shuicheng Yan et al. in ICLR, 2014](https://openreview.net/forum?id=ylE6yojDR5yqX) <!--"includes micro multi-layer
   perceptrons into the filters of convolutional layers to extract more complicated features."[DenseNet]--> <u>it implements nonlinear feature mapping compared with linear feature mapping of CNNs. The first layer is CNN feature mapping, the second layer performs nonlinear feature combination, and the third layer performs 1x1 convolution (spatial pooling). The success of NIN is due to MLP structure for convolution and pooling, and the global average pooling, which has the same number of channel as the number of class. This dedicated structure is responsible for the performance improvement.</u>

   > "includes micro multi-layer perceptrons into the filters of convolutional layers to extract more complicated features." [8]

10. [Deeply-Supervised Nets, by Chen-Yu Lee et al. in AISTATS, 2015](http://proceedings.mlr.press/v38/lee15a.pdf) <u>Add discriminative objectives for the intermediatiate layers.</u>

11. [Training Very Deep Networks, by Rupesh Kumar Srivastava et al. in NIPS, 2015](https://papers.nips.cc/paper/5850-training-very-deep-networks.pdf): *Highway network*. The input is linearly combined with feature mapping of a plain neural network by a tranformation gate.

    $y=H(x,W_{h})$

    $y=H(x,W_{h})*T(x,W_{T})+x*(1-T(x,W_{T}))$

    The *transform* gate controls the information flow.

    ------

    ​								Network Transformation

    ------

12. [Net2Net: Accelerating Learning via Knowledge Transfer, by Tianqi Chen, Iran Goodfellow and Jonathon Shlens in ICLR 2015](https://arxiv.org/abs/1511.05641): Network function-preserving

13. [Learning both Weights and Connections for Efficient Neural Network, by Song Han et al. in NIPS 2015](https://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network)

    

    

    ------

    

14. [Going Deeper with Convolutions, by Christian Szegedy et al. in CVPR, 2015](https://arxiv.org/abs/1409.4842): *Wider Networks*<!--Wider GoogLeNet with an inception module which concatenates feature-maps produced by filters of different sizes--> 

15. [Rethinking the Inception Architecture for Computer Vision, by Christian Szegedy et al. in CVPR, 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf): *Inception*

    ------

16. [Semi-supervised Learning with Ladder Networks](http://papers.nips.cc/paper/5947-semi-supervised-learning-with-ladder-ne): *Ladder Networks*

17. [Deconstructing the Ladder Network Architecture](http://proceedings.mlr.press/v48/pezeshki16.pdf)

    ------

18. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, by Sergey Ioffe and Christian Szegedy in ICML 2015](http://proceedings.mlr.press/v37/ioffe15.pdf): *Batch Normalization*

    

19. [***Deep Residual Learning for Image Recognition***, by Kaiming He et al. in CVPR, 2016 ](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf): <u>*ResNet*, H(x) = F(x) + x, the rest of the network is refered to approximate F(x) = H(x) - x, by adding input x to network blocks. Every block only learns the residual $F(x)=H(x)-x​$</u>

20. [Identity Mappings in Deep Residual Networks，by Kaiming He et al. in ECCV 2016](https://arxiv.org/abs/1603.05027)

21. [Deep Networks with Stochastic Depth, by Gao Huang et al. in ECCV, 2016](https://arxiv.org/abs/1603.09382): *Stochastic Depth Network*

22. [Identity Mappings in Deep Residual Networks, by Kaiming He et al. in ECCV, 2016](https://arxiv.org/abs/1603.05027): *Pre-activation ResNet*

23. [Densely Connected Convolutional Networks, by Gao Huang et al. in CVPR, 2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf): *DenseNet*. Concate all the proceeding activations to the next layer, forming $L(L+1)/2$ connections. It has implicit deep supervision and help avoid vanishing gradients.

24. [***Exploring Randomly Wired Neural Networks for Image Recognition***, by Saining Xie et al. in Arxiv, 2019](https://arxiv.org/pdf/1904.01569.pdf): <u>*RandomWired Network*, Thestrategy to generate the network, or network generator stated in the paper, introduce prior bias to the generated network and limit the network search space to a subspce. To circumvent the prior bias, this paper tries to explot random graphs as network generator and transform the graph into neural networks. The node operation is predifined and universal. It is similar to our idea of letting the network emerge itself, rather than pre-define any network types. </u> 

    ------

    

25. [*Neural architecture search with reinforcement learning* by Barret Zoph et al. in ICLR 2017](https://openreview.net/pdf?id=r1Ue8Hcxg): *RL-NAS*. This paper proposes to compose network structure by using a recurrent neural network as the controller, which outputs a sequence of hyper parameters. The controller network makes use of auto-regressive nature of hyperparameter prediction conditioned on the previous search. Its parameter is trained by policy gradient to maxmize the validation accuracy.  Its performance is comparable with the best Dense-net performance on CIFAR-10 image classification dataset, and excels on Penn Treebank language modeling compared with variational LSTM. It also shows significant improvement with a RL trained controller compared with random generated networks.

26. [Designing neural network architectures using reinforcement learning, by Bowen Baker et al. in ICLR 2017](https://openreview.net/pdf?id=S1c2cvqee)

27. [Learning Transferable Architectures for Scalable Image Recognition, by Barret Zoph et al. in CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zoph_Learning_Transferable_Architectures_CVPR_2018_paper.pdf)：block search space. The paper proposes to search for effictive architectures on small proxy dataset and then transfer the learned architecture to large dataset. The transferability is achieved via block-wise search space, which makes the architecture agnostic to the size of intput data and depth of the whole network. The search space is different from the whole network space, and thus has more ability to generalise to other problems.

    ------

    ​											RL

    ------

28. [Practical Block-wise Neural Network Architecture Generation, by Zhao Zhong et al. in CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhong_Practical_Block-Wise_Neural_CVPR_2018_paper.pdf)

    

29. [Searching for activation functions, by Prajit Ramachandran and Barret Zoph et al. in ArXiv 2017](https://arxiv.org/abs/1710.05941)

30. [Deeparchitect: Automatically designing and training deep architectures, by Renato Negrinho and Geoff Gordon in ICLR 2017](https://openreview.net/pdf?id=rkTBjG-AZ): *Monte Carlo Tree Search*

31. [Simple and efficient architecture search for Convolutional Neural Networks, by Thomas Elsken et al. in ICLR 2017](https://openreview.net/forum?id=SySaJ0xCZ)

32. [Neural Optimizer Search with Reinforcement Learning, by Irwan Bello and Barret Zoph et al. in ICML 2017](http://proceedings.mlr.press/v70/bello17a/bello17a.pdf)

33. [N2n learning: Network to network compression via policy gradient reinforcement learning, by Anubhav Ashok et al. in ICLR 2018](https://openreview.net/pdf?id=B1hcZZ-AW)

34. [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design by Ningning Ma and Jian Sun et al. in ECCV 2018](https://arxiv.org/pdf/1807.11164.pdf): *CNN design*

35. [Efficient Neural Architecture Search via Parameter Sharing, by Barret Zoph et al. in ICML 2018](http://proceedings.mlr.press/v80/pham18a/pham18a.pdf)

36. [Searching for Efficient Multi-Scale Architectures for Dense Image Prediction, by Liang-Chieh Chen and Barret Zoph et al. in NIPS 2018](http://papers.nips.cc/paper/8087-searching-for-efficient-multi-scale-architectures-for-dense-image-prediction.pdf)

37. [Progressive Neural Architecture Search, by Chenxi Liu and Barret Zoph et al. in ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.pdf)

38. [Regularized Evolution for Image Classifier Architecture Search by Esteban Real et al. in ArXiv 2018](https://arxiv.org/abs/1802.01548)

39. [Efficient Architecture Search by Network Transformation, by Han Cai et al. in AAAI 2018](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16755/16568): reuse network, combined with net2wider and net2deeper transformations, trained by policy gradient.

40. [Hierarchical Representations for Efficient Architecture Search, by Hanxiao Liu et al. in ICLR 2018](https://openreview.net/forum?id=BJQRKzbA-)*Neuro-evolution*

    ------

41. [Neural Architecture Optimization by Tieyan Liu et al. in NIPS 2018](https://papers.nips.cc/paper/8007-neural-architecture-optimization.pdf): *SGD-NAS*

42. [*DARTS: Differentiable Architecture Search* by Hanxiao Liu et al. in ICLR 2019](https://openreview.net/forum?id=S1eYHoC5FX)

    ------

43. [*Evaluating the Search Phase of Neural Architecture Search*, by Christian Sciuto et al. in ArXiv 2019](https://arxiv.org/pdf/1902.08142.pdf): *Random search NAS*

44. [Random Search and Reproducibility for Neural Architecture Search, by Liam Li et al. in ArXiv 2019](https://arxiv.org/abs/1902.07638)

45. [Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation by Chenxi Liu and Li Fei-Fei et al. in ArXiv 2019](https://arxiv.org/abs/1901.02985v2)

46. [NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection by Golnaz Ghiasi et al. in ArXiv 2019](https://arxiv.org/abs/1904.07392)

47. [Single Path One-Shot Neural Architecture Search with Uniform Sampling by Zichao Guo et al. in ArXiv 2019](https://arxiv.org/abs/1904.00420): *model and structure decouple*

