### AutoRL

1. [*Network In Network*, by Min Lin & Shuicheng Yan et al. in ICLR, 2014](https://openreview.net/forum?id=ylE6yojDR5yqX) <!--"includes micro multi-layer
   perceptrons into the filters of convolutional layers to extract more complicated features."[DenseNet]--> <u>it implements nonlinear feature mapping compared with linear feature mapping of CNNs. The first layer is CNN feature mapping, the second layer performs nonlinear feature combination, and the third layer performs 1x1 convolution (spatial pooling). The success of NIN is due to MLP structure for convolution and pooling, and the global average pooling, which has the same number of channel as the number of class. This dedicated structure is responsible for the performance improvement.</u>

   > "includes micro multi-layer perceptrons into the filters of convolutional layers to extract more complicated features." [8]

2. [Deeply-Supervised Nets, by Chen-Yu Lee et al. in AISTATS, 2015](http://proceedings.mlr.press/v38/lee15a.pdf) <u>Add discriminative objectives for the intermediatiate layers.</u>

3. [Training Very Deep Networks, by Rupesh Kumar Srivastava et al. in NIPS, 2015](https://papers.nips.cc/paper/5850-training-very-deep-networks.pdf): *Highway network*. The input is linearly combined with feature mapping of a plain neural network by a tranformation gate.

   $y=H(x,W_{h})$

   $y=H(x,W_{h})*T(x,W_{T})+x*(1-T(x,W_{T}))$

   The *transform* gate controls the information flow.

   ------

4. [Going Deeper with Convolutions, by Christian Szegedy et al. in CVPR, 2015](https://arxiv.org/abs/1409.4842): *Wider Networks*<!--Wider GoogLeNet with an inception module which concatenates feature-maps produced by filters of different sizes--> 

5. [Rethinking the Inception Architecture for Computer Vision, by Christian Szegedy et al. in CVPR, 2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf): *Inception*

   ------

6. [Semi-supervised Learning with Ladder Networks](http://papers.nips.cc/paper/5947-semi-supervised-learning-with-ladder-ne): *Ladder Networks*

7. [Deconstructing the Ladder Network Architecture](http://proceedings.mlr.press/v48/pezeshki16.pdf)

   ---

8. [***Deep Residual Learning for Image Recognition***, by Kaiming He et al. in CVPR, 2016 ](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf): <u>*ResNet*, H(x) = F(x) + x, the rest of the network is refered to approximate F(x) = H(x) - x, by adding input x to network blocks. Every block only learns the residual $F(x)=H(x)-x$</u>

9. [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)

10. [Deep Networks with Stochastic Depth, by Gao Huang et al. in ECCV, 2016](https://arxiv.org/abs/1603.09382): *Stochastic Depth Network*

11. [Identity Mappings in Deep Residual Networks, by Kaiming He et al. in ECCV, 2016](https://arxiv.org/abs/1603.05027): *Pre-activation ResNet*

12. [Densely Connected Convolutional Networks, by Gao Huang et al. in CVPR, 2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf): *DenseNet*. Concate all the proceeding activations to the next layer, forming $L(L+1)/2$ connections. It has implicit deep supervision and help avoid vanishing gradients.

13. [***Exploring Randomly Wired Neural Networks for Image Recognition***, by Saining Xie et al. in Arxiv, 2019](https://arxiv.org/pdf/1904.01569.pdf): <u>*RandomWired Network*, Thestrategy to generate the network, or network generator stated in the paper, introduce prior bias to the generated network and limit the network search space to a subspce. To circumvent the prior bias, this paper tries to explot random graphs as network generator and transform the graph into neural networks. The node operation is predifined and universal. It is similar to our idea of letting the network emerge itself, rather than pre-define any network types. </u> 

    ------

    

14. [*Neural architecture search with reinforcement learning* by Barret Zoph et al. in ICLR 2017](https://openreview.net/pdf?id=r1Ue8Hcxg): *RL-NAS*

15. [Learning Transferable Architectures for Scalable Image Recognition, by Barret Zoph et al. in CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zoph_Learning_Transferable_Architectures_CVPR_2018_paper.pdf)

    ------

    

16. [Searching for activation functions, by Prajit Ramachandran and Barret Zoph et al. in ArXiv 2017](https://arxiv.org/abs/1710.05941)

17. [Neural Optimizer Search with Reinforcement Learning, by Irwan Bello and Barret Zoph et al. in ICML 2017](http://proceedings.mlr.press/v70/bello17a/bello17a.pdf)

18. [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design by Ningning Ma and Jian Sun et al. in ECCV 2018](https://arxiv.org/pdf/1807.11164.pdf): *CNN design*

19. [Efficient Neural Architecture Search via Parameter Sharing, by Barret Zoph et al. in ICML 2018](http://proceedings.mlr.press/v80/pham18a/pham18a.pdf)

20. [Searching for Efficient Multi-Scale Architectures for Dense Image Prediction, by Liang-Chieh Chen and Barret Zoph et al. in NIPS 2018](http://papers.nips.cc/paper/8087-searching-for-efficient-multi-scale-architectures-for-dense-image-prediction.pdf)

21. [Progressive Neural Architecture Search, by Chenxi Liu and Barret Zoph et al. in ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.pdf)

22. [Regularized Evolution for Image Classifier Architecture Search by Esteban Real et al. in ArXiv 2018](https://arxiv.org/abs/1802.01548)

    ------

23. [Neural Architecture Optimization by Tieyan Liu et al. in NIPS 2018](https://papers.nips.cc/paper/8007-neural-architecture-optimization.pdf): *SGD-NAS*

24. [*DARTS: Differentiable Architecture Search* by Hanxiao Liu et al. in ICLR 2019](https://openreview.net/forum?id=S1eYHoC5FX)

    ------

25. [*Evaluating the Search Phase of Neural Architecture Search*, by Christian Sciuto et al. in ArXiv 2019](https://arxiv.org/pdf/1902.08142.pdf): *Random search NAS*

26. [Random Search and Reproducibility for Neural Architecture Search, by Liam Li et al. in ArXiv 2019](https://arxiv.org/abs/1902.07638)

