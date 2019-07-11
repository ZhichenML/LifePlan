Generalization of RL

The success of RL algorithms usually needs a perfect environment simulator and millions of agent-environment interaction. Recent empirical analysis shows the RL agent has a high tendency to overfit. That is, the trained agent memories the MDPs in the training set but only achieves sub-optimal performance with minor perturbation to the envionrment, such as different distribution of initial state, incremental state-action space, noisy rewards, unexperienced transition dynamics, or slightly changing of goals. To apply RL algorithms to industry applications, a robust and sample efficient framework is needed. 

To improve robustness, we need to consider:

1. Multi-task/multi-goal learning, this may involve factorization of value functions
2. representation learning for states and actions, subspace learning
3. Noise perturbed learning, related work: robust control, marginalized corrupted features
4. logical reasoning, related work: relational RL, Bayesian Prior
5. Exploration
6. POMDP where the hidden environment encodes some basic rule, and can be used to accelarate exploration
7. Deep explore, where current exploration may fail because of myopia
8. generate problem while solving one, go deep





Topics: Hierarchical RL, RL reasoning, state abstraction



1. sample efficiency prevents the scalability of RL algorihtms to practical scenarios, like learning to rank, recommendation, anomaly detection in interactive systems. This especially become problematic in terms of infinite state-action space or incremental state-action space, where approximation is usually employed. 

   'Relational Reinforcement Learning (RRL) is a technique that enables Reinforcement Learning (RL) agents to generalize from their experience, allowing them to learn over large or potentially
   infinite state spaces, to learn context sensitive behaviors, and to learn to solve variable goals and to
   transfer knowledge between similar situations.'

   However, previous **RRL methods lacks a probabilistic view**, which is particular important for exploration and results in requiring a lot of samples. This will also enable model-based RL to learn from large environment.



2. Non-stationary concept drift, pursuing goals that are changing from episode to epsode, incremental state action space 

3. **Efficient Bayesian exploration**

   **lack a regularization technique,** theoritical analysis of generalization and overfitting

    1. Bayesian prior
    2. relational RL, inductive bias, graph networks
    3. Perturbation 
    4. State-action abstraction

4. **cascade learning**, enable the agent learning from abstract to concrete. Human education usually encourages learn abstract definitions before apply them into practice. 

   related work: curriculum learning, pre-training, transfer learning, meta leraning, state locality, sparse reward (initially)

   Examples: starcraftII game retraining if changed races or maps, modulars within the environment and basic rule, wins.

   

inductive bias for exploration





Exporation-conscious memory machine  and surrogate RL

1. [Exploration Conscious Reinforcement Learning Revisited, by Lior Shani et al. in ICML 2019](<http://proceedings.mlr.press/v97/shani19a/shani19a.pdf>)
2. [Sample-Optimal Parametric Q-Learning Using Linearly Additive Features, by Lin Yang et al. in ICML 2019](<http://proceedings.mlr.press/v97/yang19b.html>)



### RL reasoning

1. [Computationally Efficient Relational Reinforcement Learning, by Mitchell Keith Bloch in PHD thesis 2018](<https://deepblue.lib.umich.edu/bitstream/handle/2027.42/145859/bazald_1.pdf?sequence=1&isAllowed=y>): efficient RRL implementations and evaluation metrics
2. [[Learning Action-State Representation Forests for Implicitly Relational Worlds, by Thomas J. Palmer in PHD thesis 2015](<https://shareok.org/bitstream/handle/11244/14592/2015_Palmer_Thomas_Dissertation.pdf?sequence=1&isAllowed=y>)
3. [Exploiting Similarity of Structure in Hierarchical Reinforcement Learning with QBOND, by Sean Harris](<http://unsworks.unsw.edu.au/fapi/datastream/unsworks:51036/SOURCE2?view=true>)
4. [Toward Practical Reinforcement Learning Algorithms: Classification Based Policy Iteration and Model-Based Learning, by Ávila Pires, Bernardo in phd thesis 2016](<http://web.b.ebscohost.com/ehost/detail/detail?vid=0&sid=b365e64a-8f5a-4f88-9724-5bf08cca3fec%40sessionmgr101&bdata=JnNpdGU9ZWhvc3QtbGl2ZQ%3d%3d#AN=154B9ED169B018D8&db=ddu>)
5. [Deep Reinforcement Learning in Natural Language Scenarios, by He, Ji in phd thesis 2017](<https://digital.lib.washington.edu/researchworks/bitstream/handle/1773/40551/He_washington_0250E_17701.pdf?sequence=1&isAllowed=y>)
6. [Efficient Deep Reinforcement Learning via Planning, Generalization, and Improved Exploration, by Oh, Junhyuk in phd thesis 2018](<http://web.b.ebscohost.com/ehost/detail/detail?vid=0&sid=50c3221c-5108-4c6b-abe6-b4e8219f7b73%40sessionmgr102&bdata=JnNpdGU9ZWhvc3QtbGl2ZQ%3d%3d#AN=3CCCD1A366CBB4A0&db=ddu>)
7. [A novel benchmark methodology and data repository for real-life reinforcement learning, by Ali Nouri et al. in Multidisciplinary Symposium on Reinforcement Learning 2009**](<https://pdfs.semanticscholar.org/c3e5/8e1b1846061e1edcbfb25e03fa86171c9f92.pdf>)
8. [Kernel Temporal Differences for Reinforcement Learning with Applications to Brain Machine Interfaces, by Bae, Jihye in phd thesis 2013](<http://web.b.ebscohost.com/ehost/detail/detail?vid=0&sid=bc8a9f25-c460-41bb-8c87-c80e274ed11b%40pdc-v-sessmgr05&bdata=JnNpdGU9ZWhvc3QtbGl2ZQ%3d%3d#AN=EB76CEF1A83041B0&db=ddu>)
9. [Regularization in reinforcement learning, by Farahmand, Amir-massoud in phd thesis 2011](<http://web.b.ebscohost.com/ehost/detail/detail?vid=0&sid=ea859445-559c-416d-9bb1-dec745519165%40pdc-v-sessmgr03&bdata=JnNpdGU9ZWhvc3QtbGl2ZQ%3d%3d#AN=552E988DE5563174&db=ddu>)
10. [On Priors for **Bayesian** Neural Networks, by Nalisnick, Eric Thomas in phd thesis 2018](<http://web.b.ebscohost.com/ehost/detail/detail?vid=1&sid=046028f0-5065-422b-a376-1f143ba2af6d%40pdc-v-sessmgr02&bdata=JnNpdGU9ZWhvc3QtbGl2ZQ%3d%3d#AN=84B7F7201A956650&db=ddu>)
11. [Efficient Reinforcement Learning with Bayesian Optimization, by Ganjali, Danyan in phd thesis 2016](<http://web.b.ebscohost.com/ehost/detail/detail?vid=0&sid=752311e0-b62c-4219-ad4c-c60663ab979f%40pdc-v-sessmgr01&bdata=JnNpdGU9ZWhvc3QtbGl2ZQ%3d%3d#AN=AB858BD17A5FDFDC&db=ddu>)
12. [Deep reinforcement learning with relational inductive biases, by Vinicius Zambaldi et al. in ICLR 2019](<https://arxiv.org/abs/1806.01830>)
13. [Relational inductive biases, deep learning, and graph networks, by Peter W. Battaglia in arXiv 2018](<https://arxiv.org/pdf/1806.01261.pdf>)
14. [Relational Reinforcement Learning, by SASO D ˇ ZEROSKI et al. in Machine Learning 2001](<https://link.springer.com/content/pdf/10.1023/a:1007694015589.pdf>)
15. [**Relational Reinforcement Learning An Overvie**, by Prasad Tadepall et al. in ICML 2004](<https://pdfs.semanticscholar.org/e514/e7e57a6c7b912a062e0a8756dc3c060bbe0c.pdf>)
16. [Protecting against evaluation overfitting in empirical reinforcement learning, by Shimon Whiteson et al. in Adaptive Dynamic Programming And Reinforcement Learning 2011**](<https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5967363>): a "multiple environments" paradigm is proposed to separate the training and testing stages
17. [Exploration in Relational Domains for Model-based Reinforcement Learning, by Tobias Lang et al. in JMLR 2012](<http://www.jmlr.org/papers/volume13/lang12a/lang12a.pdf>)
18. [Understanding deep learning requires rethinking generalization, by Chiyuan Zhang et al. in ICLR 2017****](<https://openreview.net/forum?id=Sy8gdB9xx>)
19. [Can deep reinforcement learning solve erdos-selfridge-spencer games? by Maithra Raghu et al. in arXiv 2017***](<https://arxiv.org/abs/1711.02301>)
20. [Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents, by Marlos C. Machado in JAIR 2018](<https://arxiv.org/abs/1709.06009>)
21. [Contextual decision processes with low Bellman rank are PAC-learnable, by Nan Jiang in ICML 2017](<http://proceedings.mlr.press/v70/jiang17c/jiang17c.pdf>)
22. [Near-optimal regret bounds for reinforcement learning, by Thomas Jaksch et al. in JMLR 2010](<http://www.jmlr.org/papers/volume11/jaksch10a/jaksch10a.pdf>)
23. [Minimax regret bounds for reinforcement learning, by Mohammad Gheshlaghi Azar et al. in ICML 2017](<https://arxiv.org/abs/1703.05449>)
24. [Unifying PAC and Regret: Uniform PAC Bounds for Episodic Reinforcement Learning, by Christoph Dann et al. in NIPS 2017](<https://papers.nips.cc/paper/7154-unifying-pac-and-regret-uniform-pac-bounds-for-episodic-reinforcement-learning.pdf>)
25. [Deep reinforcement learning that matters, by Peter Henderson et al. in arXiv 2017](<https://arxiv.org/abs/1709.06560>): reproducibility
26. [VAIN: Attentional Multi-agent Predictive Modeling, by Yedid Hoshen in arXiv 2017](<https://arxiv.org/abs/1706.06122>)
27. [Machine Theory of Mind, by Neil C. Rabinowitz et al. in arXiv 2018](<https://arxiv.org/pdf/1802.07740.pdf>)
28. [Bayesian Transfer Reinforcement Learning with Prior Knowledge Rules, by Michalis K. Titsias et al. in arXiv 2018](<https://arxiv.org/pdf/1810.00468.pdf>): it considers rules extracted from a memroy storage as prior to learn policy. However, it does not generalize to more complex environments.
29. [A Study on Overfitting in Deep Reinforcement Learning, by Chiyuan Zhang et al. in arXiv 2018****](<https://arxiv.org/pdf/1804.06893.pdf>): A3C shows overfitting to reward noise, while random initialization and perturbed action choice does not prevent (as regularizer in training) or detect it (when evaluation on the training set and compare to test performance). Large training samples largely help with overfiting.
30. [NeurIPS 2019 Competition: The MineRL Competition on Sample Efficient Reinforcement Learning using Human Priors, by William H. Guss et al. in NIPS 2019****](<https://arxiv.org/pdf/1904.10079.pdf>)
31. [Efficient Reinforcement Learning with a Mind-Game for Full-Length StarCraft II, by Ruo-Ze Liu et al. in arXiv 2019****](<https://arxiv.org/pdf/1903.00715.pdf>): mannually design a mind game, which is a subset of original game. The policy is learned on the mind game first and fine tune on the original game.
32. [Diagnosing Bottlenecks in Deep Q-learning Algorithms, by Justin Fu et al. in ICML 2019***](<http://proceedings.mlr.press/v97/fu19a.html>)
33. [Neural Relational Inference for Interacting Systems, by Thomas Kipf et al. in ICML 2019](<https://arxiv.org/pdf/1802.04687.pdf>)
34. 







1. [Bayesian Reinforcement Learning: A Survey, by Mohammad Ghavamzadeh et al. in arXiv 2016](<https://arxiv.org/pdf/1609.04436.pdf>)
2. [Variational Bayesian Reinforcement Learning with Regret Bounds, by Brendan O'Donoghue et al. in arXiv 2019](<https://arxiv.org/abs/1807.09647>)
3. [Bayesian Hierarchical Reinforcement Learning, Feng Cao et al. in NIPS 2012]()
4. [An Efficient Approach to Model-Based Hierarchical Reinforcement Learning](<https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14771>)
5. [Exploring Hierarchy-Aware Inverse Reinforcement Learning, by Chris Cundy et al. in arXiv 2018](<https://arxiv.org/abs/1807.05037>)
6. [Monte Carlo Bayesian Hierarchical Reinforcement Learning, by Ngo Anh Vien in AAMAS 2014](<http://delivery.acm.org/10.1145/2620000/2616057/p1551-ngo.pdf?ip=144.82.8.41&id=2616057&acc=ACTIVE%20SERVICE&key=BF07A2EE685417C5%2ED93309013A15C57B%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1562411123_426cef36aadb0b2f2f02691836c28557>)
7. [DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning, by Wenhan Xiong et al. in ACL2017](<https://www.aclweb.org/anthology/D17-1060>)
8. [Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning, by Rajarshi Das et al. in ICLR 2018](<https://openreview.net/forum?id=Syg-YfWCW>)]



### 5. Diversity and regularization in multi-agent RL

Problem statement: exploration heavily influence the performance of RL algorithms. To encourage exploration, researchers have proposed distance based diversity in action space, MMD distance based diversity in trajectory space,  entropy regularisation.

option aware exploration, relational 

related work: 

 1. Noise: 

     	1. action space: epsilon-greedy, Boltzmann, max-entropy
     	2. Parameter space
     	3. Q ensemble

 2. exporation bonus: 

     	1. state visitation counts,
     	2. curiosity 
     	3. variational information maximisation

 3. Metalearning: RL2, MAESN

 4. Transfer + bonus

 5. Behavior diversity

 6. Hindsight experience replay

 7. opponent modeling in multi-agent RL

 8. Fairness in multi-agent

 9. Multi-goal as regularization

 10. Relational RL

     

1. [Deep exploration via randomized value functions, by Ian Osband in phd thesis 2016](<https://stacks.stanford.edu/file/druid:rp457qc7612/iosband_thesis-augmented.pdf>)
2. [Randomized Prior Functions for Deep Reinforcement Learning, by  Ian Osband in NIPS 2018](<https://sites.google.com/view/randomized-prior-nips-2018/>)
3. [Diversity-Driven Exploration Strategy for Deep Reinforcement Learning, by Zhang-Wei Hong et al. in NIPS 2018](<https://papers.nips.cc/paper/8249-diversity-driven-exploration-strategy-for-deep-reinforcement-learning.pdf>): it adds a distance term that measures the expected KL-distance or MSE distance of the current policy and previous policies to encourage behavior diversity. Case studies include DQN, DDPG, and A2C.
4. [Learning-Driven Exploration for Reinforcement Learning, by Muhammad Usama et al. in arXiv 2019](<https://arxiv.org/abs/1906.06890>): use entropy defined on Q to encourage exploration.
5. [Diversity-Inducing Policy Gradient: Using Maximum Mean Discrepancy to Find a Set of Diverse Policies, by Muhammad A Masood et al. in IJCAI 2019](<https://arxiv.org/pdf/1906.00088.pdf>): use MMD of policy trajectories to encourage diversity.
6. [Modeling Others using Oneself in Multi-Agent Reinforcement Learning, by Roberta Raileanu et al. in arXiv 2018](<https://research.fb.com/wp-content/uploads/2018/07/Modeling-Others-using-Oneself-in-Multi-Agent-Reinforcement-Learning.pdf>): opponent modeling
7. [Simultaneously Learning and Advising in Multiagent Reinforcement Learning, by Felipe Leno da Silva et al. in AAMAS 2017](<http://www.ifaamas.org/Proceedings/aamas2017/pdfs/p1100.pdf>)
8. [Learning latent state representation for speeding up exploration, by Giulia Vezzani et al. in arXiv 2019](<https://arxiv.org/abs/1905.12621>)
9. [Control Regularization for Reduced Variance Reinforcement Learning, by Richard Cheng et al. in ICML 2019](<https://arxiv.org/abs/1905.05380>)
10. [Regularized Policy Gradients: Direct Variance Reduction in Policy Gradient Estimation, by Tingting Zhao et al. in JMLR 2015](<http://proceedings.mlr.press/v45/Zhao15b.pdf>)
11. [SOLAR: Deep Structured Representations for Model-Based Reinforcement Learning, by Marvin Zhang et al. in ICML 2019]()
12. [Social Influence as Intrinsic Motivation for Multi-Agent Deep Reinforcement Learning, by Natasha Jaques et al. in ICML 2019](<https://arxiv.org/pdf/1810.08647.pdf>)
13. [Way Off-Policy Batch Deep Reinforcement Learning of Implicit Human Preferences in Dialog, by Natasha Jaques et al. in arXiv 2019](<https://arxiv.org/pdf/1907.00456.pdf>)
14. [The Value Function Polytope in Reinforcement Learning, by Robert Dadashi in ICML 2019](<https://arxiv.org/pdf/1901.11524.pdf>): geometic of value function.
15. [Information-Theoretic Considerations in Batch Reinforcement Learning, by Nan Jiang et al. in ICML 2019](<https://arxiv.org/abs/1905.00360>): new perspective of exploration
16. [Action Robust Reinforcement Learning and Applications in Continuous Control, by Chen Tessler et al. in ICML 2019](<https://arxiv.org/abs/1901.09184>): robust action and transferiability
17. [Discovering Options for Exploration by Minimizing Cover Time, by Yuu Jinnai et al. in ICML 2019](<https://arxiv.org/abs/1903.00606>): options and diversity
18. [A Theory of Regularized Markov Decision Processes, by Matthieu Geist et al. in ICML 2019](<https://arxiv.org/abs/1901.11275>)
19. [Learning Action Representations for Reinforcement Learning, by Yash Chandak et al. in ICML 2019](<https://arxiv.org/abs/1902.00183>)
20. [Learning from a Learner, by Alexis D. Jacq in ICML 2019](<http://proceedings.mlr.press/v97/jacq19a/jacq19a.pdf>)
21. [DeepMDP: Learning Continuous Latent Space Models for Representation Learning, by Carles Gelada et al. in ICML 2019](<https://arxiv.org/abs/1906.02736>)
22. [Learning to Collaborate in Markov Decision Processes, by Goran Radanovic et al. in ICML 2019](<https://arxiv.org/abs/1901.08029>)
23. [Task-Agnostic Dynamics Priors for Deep Reinforcement Learning, by Yilun Du et al. in ICML 2019](<https://arxiv.org/abs/1905.04819>)
24. [Distributional Reinforcement Learning for Efficient Exploration, by Borislav Mavrin et al. in ICML 2019](<https://arxiv.org/abs/1905.06125>)
25. [CURIOUS: Intrinsically Motivated Modular Multi-Goal Reinforcement Learning, by Cédric Colas et al. in ICML 2019](<https://arxiv.org/abs/1810.06284>)
26. [Multi-Agent Adversarial Inverse Reinforcement Learning, by Lantao Yu et al. in ICML 2019](<http://proceedings.mlr.press/v97/yu19e.html>)
27. [TibGM: A Transferable and Information-Based Graphical Model Approach for Reinforcement Learning, by Tameem Adel  et al. in ICML 2019](<http://proceedings.mlr.press/v97/adel19a.html>)
28. [α-Rank: Multi-Agent Evaluation by Evolution, by Shayegan Omidshafei et al. in Nature 2019](<https://www.nature.com/articles/s41598-019-45619-9.pdf>)
29. [Human Replay Spontaneously Reorganizes Experience, by Yunzhe Liu et al. in Cell 2019](<https://www.cell.com/action/showPdf?pii=S0092-8674%2819%2930640-3>)
30. [Unsupervised Learning of Object Keypoints for Perception and Control, by Tejas Kulkarni et al. in arXiv 2019](<https://arxiv.org/pdf/1906.11883.pdf>): learn keypoints representations for frame by prediction. used to improve explore efficiency
31. [Go-Explore: a New Approach for Hard-Exploration Problems, by Jeff Clune et al. in ICML 2019](<http://www.evolvingai.org/files/1901.10995.pdf>): to solve hard explore problems due to shallow exploration and unable to regenerate previous states, it proposes to determinstically explore and store the action sequence in the archive, then an imitation learner is used to learn robust policy based it. it shows great success in Montezuma’s Revenge and Pitfall.







### 6. Multi-objective RL

### 7. Multi-goal RL

1. [Maximum Entropy-Regularized Multi-Goal Reinforcement Learning, by Rui Zhao et al. in ICML 2019](<http://proceedings.mlr.press/v97/zhao19d/zhao19d.pdf>)

### 8. [Hierarchical RL](<https://www.reddit.com/r/reinforcementlearning/comments/a7jorm/resources_to_study_hierarchical_reinforcement/>)

1. [Hierarchical Reinforcement Learning with Hindsight, by Andrew Levy et al. in arXiv 2018](<https://arxiv.org/abs/1805.08180>)
2. [Automatic Discovery and Transfer of MAXQ Hierarchies, by Thomas Dietterich et al. in ICML 2008](<http://web.engr.oregonstate.edu/~tgd/publications/ml2008-maxq-discovery.pdf>)
3. [Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition, by Thomas Dietterich in JAIR 2000](<https://www.jair.org/index.php/jair/article/view/10266/24463>)
4. [Bayesian Hierarchical Reinforcement Learning, by Feng Cao in NIPS 2012](<https://pdfs.semanticscholar.org/7008/fb8b85bcfcf772435e984dde31e42021da66.pdf>)
5. [Model-Based Bayesian RL](<http://cbl.eng.cam.ac.uk/pub/Intranet/MLG/ReadingGroup/Bayesian_Reinforcement_Learning_Neil.pdf>)

### 9. Prior over MDPs

### 10. disentangled dark environment

The environment interacts with the agent by returning rewards and next state is model-free RL. Model-based RL learns the state transition dynamics so that the agent can plan against it. The hidden environment is defined as the properties of the environment that is not shown to the agent in model-free and model-based environment. Examples include the armour, health etc. in AtariII game, the condition of car, the amount of gas in order-dispatching. These information is is generally agnostic to the agent but may facilitate efficient exloration when giving agent. 

