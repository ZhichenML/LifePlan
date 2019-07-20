[TOC]

------



### Project:  Temporal Logic Constrained MARL

#### Problem definition： 

Slides is [here](https://docs.google.com/presentation/d/16UZpW0PSG4tcUDsLJZgUty8KBgszZabfljHEt3JhaTI/edit?usp=sharing)

Multi-agent model-free RL usually samples data from interactions or memory buffer to improve the policy. Due to uncontrollable data sampling, this regime is generally inefficient. We ask if we could synthesis the logic from the game rule depicted by [Game Description Language](<http://games.stanford.edu/games/gdl.html>) and use that strategic logic to guide the update of the policy.

<u>Inputs</u>: multi agent game rules —>[extraction]—>Alternating temporal logic —>[Convert]--> Linear temporal logic

<u>Outputs</u>: policy

<u>Training</u>: Linear temporal logic, learn the policy

<u>Test</u>: normal RL test on newly generated environment 

#### Related work:

##### 1) [Linear Temporal Logic](<http://www.cds.caltech.edu/~murray/courses/afrl-sp12/L3_ltl-24Apr12.pdf>) Model-free

1. [Limit-deterministic B¨uchi automata for linear temporal logic, by Salomon Sickert et al. in CAV 2016](<https://link.springer.com/content/pdf/10.1007%2F978-3-319-41540-6_17.pdf>)
2. [Probably Approximately Correct MDP Learning and Control With Temporal Logic Constraints, by Jie Fu et al. in arXiv 2014](<http://www.roboticsproceedings.org/rss10/p39.pdf>)
3. [**Reinforcement learning with temporal logic rewards**, by Xiao Li et al. in arXiv 2016](<https://arxiv.org/abs/1612.03471>): temporal logic rewards
4. [Logically-correct reinforcement learning, by Mohammadhosein Hasanbeig et al . in arXiv 2018](<https://arxiv.org/abs/1801.08099>)
5. [A Learning Based Approach to Control Synthesis of Markov Decision Processes for Linear Temporal Logic Specifications, by Dorsa Sadigh et al. in arXiv 2014](<http://iliad.stanford.edu/pdfs/publications/sadigh2014learning.pdf>): formulate automata weighted MDP to solve MDP while satisfying the linear temporal logic contraint
6. [Reduced Variance Deep Reinforcement Learning with Temporal Logic Specifications, by Qitong Gao in ICCPS 2019](<http://delivery.acm.org/10.1145/3320000/3311053/p237-gao.pdf?ip=144.82.8.41&id=3311053&acc=ACTIVE%20SERVICE&key=BF07A2EE685417C5%2ED93309013A15C57B%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1563479818_9c65dedfd6f7b0dfc8cf390258154443>)
7. [DL2: Training and Querying Neural Networks with Logic, by Marc Fischer et al. in ICML 2019](<http://proceedings.mlr.press/v97/fischer19a/fischer19a.pdf>)

##### 2) [Linear Temporal Logic](<http://web.iitd.ac.in/~sumeet/slide3.pdf>) Model-based: we don't need a perfect model for planning, instead we only have a set of rules that provide guidance or warnnings to the agent when exploring

1. [Motion planning and control from temporal logic specifications with probabilistic satisfaction guarantees, by M. Lahijanian et al. in ICRA 2010](<https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5509686>)
2. [Optimal path planning for surveillance with temporal-logic constraints, by Stephen L. Smith et al. in IJRR 2011](<https://pdfs.semanticscholar.org/1fbb/cec5ffaf45af9317c5bddf8f5cf6a365d14f.pdf>)
3. [Model-based reinforcement learning in continuous environments using real-time constrained optimization, by Olov Andersson et al. in AAAI 2015](<https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9850/9878>)
4. [***Using Advice in Model-Based Reinforcement Learning***](<http://www.cs.toronto.edu/~toryn/papers/CAI-2018.pdf>): it uses nondeterministic finite state machine to express logical rules/linear temporal logic. Based on that, an advice guidance and warning indicator function $h$ is calculated to direct the  action choice in model-based RL. Such that the agent takes the action that avoids warning states and reach the goal state with minimum steps.

##### 3) [Safe RL](<https://arxiv.org/pdf/1606.06565.pdf>)

1. [Verification and repair of control policies for safe reinforcement learning](<https://link.springer.com/content/pdf/10.1007%2Fs10489-017-0999-8.pdf>): post-learning by repairing the policy with contraints
2. [**Safe Model-based Reinforcement Learning with Stability Guarantees**, by Felix Berkenkamp et al. in NIPS 2017](<https://papers.nips.cc/paper/6692-safe-model-based-reinforcement-learning-with-stability-guarantees.pdf>): safe region expansion accelates exporation
3. [Constrained Policy Optimization, by Pieter Abbeel et al. in ICML 2017](<https://arxiv.org/abs/1705.10528>)
4. [Safe reinforcement learning via shielding, by Mohammed Alshiekh et al. in AAAI 2018](<https://arxiv.org/abs/1708.08611>)
5. 

##### 4) Learning First-order Logic 

1. [Reinforcement Learning with Markov Logic Networks, by Weiwei Wang et al. in MICAI 2008](<https://link.springer.com/content/pdf/10.1007%2F978-3-540-88636-5_22.pdf>): model free Q(lambda) with first order logic [Markov Logic Network](<https://homes.cs.washington.edu/~pedrod/papers/mlj05.pdf>) in the Q network.
2. [Transfer in Reinforcement Learning via Markov Logic Networks, by Lisa Torrey et al. in AAAI 2008](<https://aaai.org/Papers/Workshops/2008/WS-08-13/WS08-13-009.pdf>)
3. [Solving Relational and First-Order Logical Markov Decision Processes: A Survey, by Martijn van Otterlo in 2012](<http://martijnvanotterlo.nl/vanOtterlo-relational-reinforcement-survey-2012.pdf>)

##### 5) Bayesian RL

1. [Bayesian Transfer Reinforcement Learning with Prior Knowledge Rules, by Michalis K. Titsias et al. in arXiv 2018](<https://arxiv.org/pdf/1810.00468.pdf>): it considers rules extracted from a memroy storage as prior to learn policy. However, it does not generalize to more complex environments.

##### 6) RL with non-technical human input: we don't need to mannually specify what we want the agent to do, instead the agent is given guidence from a rule learner.

1. [**Deep reinforcement learning from human preferences**, by Paul Christiano et al. in NIPS 2017](<https://arxiv.org/abs/1706.03741>)
2. [Interactive Learning from Policy-Dependent Human Feedback, by Michael L. Littman et al. in ICML 2017](<http://proceedings.mlr.press/v70/macglashan17a/macglashan17a.pdf>)
3. [Improving Reinforcement Learning with Human Input, by Matthew E. Taylor in IJCAI 2018 workshop](<https://www.ijcai.org/proceedings/2018/0817.pdf>)

##### 7) Rule Transfer

1. [Cross-Domain Transfer for Reinforcement Learning, by Matthew E. Taylor et al. in ICML 2007](<https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/ICML07-taylor.pdf>)

##### 8) [General Game](<https://graphics.stanford.edu/~mdfisher/Data/GeneralGameLearning.pdf>): omit the representations and learn generalizable rules from many games



Use logic rules: LTL specified RL, compositional neural network

learn logic rules: learn from sets of binary variables to find key observations as atoms 

##### 10) Infer Logical Rules

1. [Extraction of logical rules from training data using backpropagation networks, in 1996](<https://fizyka.umk.pl/publications/kmk/96lrules.pdf>): extract rules from neural networks
2. [Learning Information Extraction Rules: An Inductive Logic Programming approach in ECAI 2002](<http://www.aiai.ed.ac.uk/~stuart/Papers/ecai02-paper.pdf>)
3. [GDL Meets ATL: A Logic for Game Description and Strategic Reasoning, by Guifei Jiang et al. in PRICAI 2014](<https://link.springer.com/content/pdf/10.1007%2F978-3-319-13560-1_58.pdf>)
4. [Strategy Logics and the Game Description Language, by WIEBE VAN DER HOEK et al. in 2007](<http://www.cs.ox.ac.uk/people/michael.wooldridge/pubs/lori2007a.pdf>): it shows transition from [GDL](<http://games.stanford.edu/games/gdl.html>) to ATL
5. [Interpretable Apprenticeship Learning with Temporal Logic Specifications, by Daniel Kasenberg et al. in ](https://hrilab.tufts.edu/publications/kasenbergscheutz17cdc.pdf)
6. [Bayesian Inference of Temporal Task Specifications from Demonstrations, by Ankit Shah et al. NIPS 2018](https://papers.nips.cc/paper/7637-bayesian-inference-of-temporal-task-specifications-from-demonstrations.pdf)
7. [Teaching Multiple Tasks to an RL Agent using LTL, by Rodrigo Toro Icarte in AAMAS 2018](http://www.cs.toronto.edu/~rntoro/docs/LPOPL.pdf)
8. [Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning, by Rodrigo Toro Icarte et al. in ICML 2018](http://proceedings.mlr.press/v80/icarte18a/icarte18a.pdf): high-level task-decomposition for [single agent](http://www.dis.uniroma1.it/~kr18actions/slides/TorynQwyllynKlassen.pdf)

##### 11) Temporal Logical For Multi-agent Learning

1. [**A Multiagent Semantics for the Game Description Language**, by Stephan Schiffel et al. in ICAART 2009](<http://www.cse.unsw.edu.au/~mit/Papers/ICAART09.pdf>): from GDL to multi-agent compatible GDL, implementation unknown
2. [**GDL Meets ATL: A Logic for Game Description and Strategic Reasoning**, by Guifei Jiang et al. in PRICAI 2014](<http://publications.ut-capitole.fr/29976/1/assistant_15442039_397188782_0.pdf>): from GDL extract ATL, ATL to LTL unknown
3. [**Enforcing Signal Temporal Logic Specifications in Multi-Agent Adversarial Environments: A Deep Q-Learning Approach**, by Devaprakash Muniraj et al. in CDC 2018](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8618746): from control objective, linear temporal logic, to the reward of RL agent, with two-player zero-sum game

**Related Work in Multi-agent System Control**

1. [~~A Quantified Epistemic Logic for Reasoning about Multi-Agent Systems, by F. Belardinelli et al. in AAMAS 2007~~](https://www.ibisc.univ-evry.fr/~belardinelli/Documents/AAMAS07.pdf)
2. [~~A Temporal Logic for Stochastic Multi-Agent Systems, by Wojciech Jamroga in PRIMA 2008~~](https://link.springer.com/content/pdf/10.1007%2F978-3-540-89674-6_27.pdf): 
   - all or groups of agent cooperation
3. [~~Relentful Strategic Reasoning in Alternating-Time Temporal Logic, by Fabio Mogavero et al. in MMV 2010~~](http://people.na.infn.it/~Murano/pubblicazioni/GAMES10b.pdf)
4. [Multi-agent plan reconfiguration under local LTL specifications, by Meng Guo in Journal of Robotics Research 2015](https://journals.sagepub.com/doi/pdf/10.1177/0278364914546174)
5. [Census Signal Temporal Logic Inference for Multi-Agent Group Behavior Analysis, by Zhe Xu et al. in arXiv 2016](https://arxiv.org/pdf/1610.05612v1.pdf)
6. [Task and Motion Coordination for Heterogeneous Multi-agent Systems with Loosely-coupled Local Tasks, by Meng Guo and Dimos V. Dimarogonas in IEEE Transactions on Automation Science and Engineering (T-ASE) 2017](https://ieeexplore.ieee.org/document/7778995)
7. [Multi-Agent Non-Linear Temporal Logic with Embodied Agent describing Uncertainty, by Vladimir Rybakov in](https://e-space.mmu.ac.uk/600862/2/K-AMASTA2014a.pdf)
8. [Multi-agent persistent surveillance under temporal logic constraints, by Kevin Leahy in PHD thesis 2017](https://open.bu.edu/handle/2144/20842)
9. [Control Synthesis for Multi-Agent Systems under Metric Interval Temporal Logic Specifications, by Andersson, S., Nikou et al. in IFAC 2017](http://kth.diva-portal.org/smash/get/diva2:1152552/FULLTEXT01.pdf)
10. [On the Timed Temporal Logic Planning of Coupled Multi-Agent Systems, by Alexandros Nikou et al. in arXiv 2017](https://arxiv.org/pdf/1709.06888.pdf)
11. [Provably-Correct Coordination of Large Collections of Agents with Counting Temporal Logic Constraints, by Yunus Emre Sahin et al. in ICCPS 2017](http://web.eecs.umich.edu/~necmiye/pubs/SahinNO_iccps17.pdf)
12. [Motion and Cooperative Transportation Planning for Multi-Agent Systems under Temporal Logic Formulas, by Christos K. Verginis et al. in arXiv 2018](<https://arxiv.org/abs/1803.01579>)
13. [Concurrent Multi-Agent Systems with Temporal Logic Objectives: Game Theoretic Analysis and Planning through Negotiation, by Jie Fu et al. in 2014](http://research.me.udel.edu/~btanner/Papers/masgame_6-27.pdf)
14. [Human-in-the-Loop Control Synthesis for Multi-Agent Systems under Hard and Soft Metric Interval Temporal Logic Specifications, by Sofie Ahlberg et al. in CASE 2019](https://people.kth.se/~sofa/CASE19.pdf)





---------------

------------

### Project:  GDL program synthesis from game playing demonstrations

#### Problem Definition

<u>Input</u>: Demonstrations

<u>Output</u>: synthesised GDL program

<u>Training</u>: supervised multi-task, from demonstrations of a set of games, predict the underlying GDL program

<u>Test</u>: new demonstration, predict the underlying GDL program

#### Related work

##### 1. Program Synthesis:

1. [Neural Program Meta-Induction, by Jacob Devlin et al. in NIPS 2017](<https://papers.nips.cc/paper/6803-neural-program-meta-induction.pdf>)
2. [Leveraging grammar and reinforcement learning for neural program synthesis, by Rudy Bunel et al. in ICLR 2018](<https://openreview.net/forum?id=H1Xw62kRZ>)
3. [Neural Program Synthesis from Diverse Demonstration Videos, by Shao-Hua Sun et al. in ICML 2018](<http://proceedings.mlr.press/v80/sun18a/sun18a.pdf>): it uses a encoder-decoder structure to generate synthesis programs from video demonstrations. demonstrations from a identical program learn together by multi-task learning. Source code is required for supervised training.
4. [Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis, by Rudy Bunel et al. in ICLR 2018](<https://openreview.net/pdf?id=H1Xw62kRZ>)



##### 2. Neural Inductive Logic 

1. [Neural logic machines, by Honghua Dong et al. in ICLR 2019](<https://openreview.net/pdf?id=B1xY-hRctX>)

##### 3. Off-policy exploration

##### 4. Model-based RL

##### 5. Rule based RL

1. [Rule-based Reinforcement Learning augmented by External Knowledge, by Nicolas Bougie et al. in AEGAP 2018](<http://cadia.ru.is/workshops/aegap2018/papers/AEGAP_2018_Bougie_et_al.pdf>)：it utilizes a conjunction of properseptions from external knowledge to represent state and Q table initialisation and shows better convergency than DQN and DDPG. However, since it is rule extraction is based on mannually labeled transitions, it is not straightforward to generalize to environments with complex dynamics.
2. [Gradient-Based Relational Reinforcement Learning of Temporally Extended Policies, by Charles Gretton in AAAI 2007](<https://aaai.org/Papers/ICAPS/2007/ICAPS07-022.pdf>):trains the parameterised rule-based policy using policy gradient

##### 6. Inverse SAT

1. [The Inverse Satisfiability Problem, by Dimitris Kavvadias et al. in 1996](<https://link.springer.com/content/pdf/10.1007/3-540-61332-3_158.pdf>)
2. [A Dichotomy Theorem for the Inverse Satisfiability Problem, in 2017](<https://www.math.tu-dresden.de/~lagerqvi/fsttcs2017.pdf>)

##### 7. Interpretable Reinforcement Learning

1. [Programmatically Interpretable Reinforcement Learning, by Abhinav Verma et al. in ICML 2018](<https://arxiv.org/pdf/1804.02477.pdf>): use a pragramic language to imitate the DRL policy.
2. [Interpretable and Pedagogical Examples, by Smitha Milli et al. in ](<https://arxiv.org/pdf/1711.00694.pdf>): interpret through teaching
3. [Interpretable Multi-Objective Reinforcement Learning through Policy Orchestration, by Ritesh Noothigattu et al. in arXiv 2018](<http://www.cs.cmu.edu/~rnoothig/papers/multiobjective_rl.pdf>)
4. [SDRL: Interpretable and Data-efficient Deep Reinforcement Learning Leveraging Symbolic Planning, by Daoming Lyu in arXiv 2018](<https://arxiv.org/abs/1811.00090>)
5. [Interpretable Reinforcement Learning with Ensemble Methods, by Alexander Brown et al. in arXiv 2018](<https://arxiv.org/abs/1809.06995>)
6. [Towards Better Interpretability in Deep Q-Networks, by Raghuram Mandyam Annasamy et al. in arXiv 2018](<https://arxiv.org/abs/1809.05630>)
7. [Combined Reinforcement Learning via Abstract Representations, by Vincent François-Lavet et al. in arXiv 2018](<https://arxiv.org/abs/1809.04506>)
8. [Toward Interpretable Deep Reinforcement Learning with Linear Model U-Trees, by Guiliang Liu et al. in ECML 2018](<http://www.ecmlpkdd2018.org/wp-content/uploads/2018/09/246.pdf>)
9. [Hierarchical and Interpretable Skill Acquisition in Multi-task Reinforcement Learning, by Tianmin Shu et al. in ICLR 2018](<https://openreview.net/forum?id=SJJQVZW0b>)
10. [SDRL: Interpretable and Data-efficient Deep Reinforcement Learning Leveraging Symbolic Planning, by Daoming Lyu et al. in AAAI 2019](<https://arxiv.org/abs/1811.00090>)
11. [InfoRL: Interpretable Reinforcement Learning using Information Maximization, by Aadil Hayat et al. in arXiv 2019](<https://arxiv.org/abs/1905.10404>)
12. [Keep it stupid simplem, by Erik J Peterson et al. in AAAI 2019](<https://arxiv.org/abs/1809.03406>)
13. [Imitation-Projected Policy Gradient for Programmatic Reinforcement Learning, by Abhinav Verma et al. in arXiv 2019](<https://arxiv.org/pdf/1907.05431.pdf>)
14. [Interpretable Dynamics Models for Data-Efficient Reinforcement Learning, by Markus Kaiser et al. in arXiv 2019](<https://arxiv.org/pdf/1907.04902.pdf>)
15. [An Approximate Bayesian Approach to Surprise-Based Learning, by Vasiliki Liakoni et al. in arXiv 2019](<https://arxiv.org/pdf/1907.02936.pdf>)
16. [Conservative Q-Improvement: Reinforcement Learning for an Interpretable Decision-Tree Policy, by Aaron M. Roth et al. in arXiv 2019](<https://arxiv.org/pdf/1907.01180.pdf>)
17. [A Theoretical Connection Between Statistical Physics and Reinforcement Learning, by Jad Rahme et al. in arXiv 2019](<https://arxiv.org/pdf/1906.10228.pdf>)
18. [MoËT: Interpretable and Verifiable Reinforcement Learning via Mixture of Expert Trees, by Marko Vasic et al. in arXiv 2019](<https://arxiv.org/abs/1906.06717>)
19. [Direct Policy Gradients: Direct Optimization of Policies in Discrete Action Spaces, by Guy Lorberbom et al. in arXiv 2019](<https://arxiv.org/abs/1906.06062>)
20. [Towards Interpretable Reinforcement Learning Using Attention Augmented Agents, by Alex Mott et al in arXiv 2019](<https://arxiv.org/abs/1906.02500>)
21. [Finding Friend and Foe in Multi-Agent Games, by Jack Serrino et al. in arXiv 2019](<https://arxiv.org/abs/1906.02330>)
22. [Sequence Modeling of Temporal Credit Assignment for Episodic Reinforcement Learning, by Yang Liu et al. in arXiv 2019](<https://arxiv.org/abs/1905.13420>)
23. [Learning Compositional Neural Programs with Recursive Tree Search and Planning, by Thomas Pierrot et al. in arXiv 2019](<https://arxiv.org/abs/1905.12941>)
24. [***Neural Logic Reinforcement Learning***, by Zhengyao Jiang et al. in arXiv 2019](<https://arxiv.org/abs/1904.10729>)
25. [Object-Oriented Dynamics Learning through Multi-Level Abstraction, by Guangxiang Zhu et al. in arXiv 2019](<https://arxiv.org/abs/1904.07482>)
26. [Interpretable Reinforcement Learning via Differentiable Decision Trees, by Ivan Dario Jimenez Rodriguez et al. in arXiv 2019](<https://arxiv.org/abs/1903.09338>)
27. [Learn to Interpret Atari Agents, by Zhao Yang et al. in arXiv 2019](<https://arxiv.org/abs/1812.11276>)
28. [Connecting the Dots Between MLE and RL for Sequence Prediction, by Eric Xing et al. in arXiv 2019](<https://arxiv.org/abs/1811.09740>)
29. [Evolving intrinsic motivations for altruistic behavior, by Jane X. Wang et al. in arXiv 2019](<https://arxiv.org/abs/1811.05931>)
30. [Bayesian Action Decoder for Deep Multi-Agent Reinforcement Learning, by Jakob N. Foerster et al. in arXiv 2019](<https://arxiv.org/abs/1811.01458>)



##### 8. General Game Playing

1. [Game Description Language Compiler Construction, by Jakub Kowalski et al. in AI 2013](<https://link.springer.com/chapter/10.1007/978-3-319-03680-9_26>)
2. [GDL-III: A Description Language for Epistemic General Game Playing, by Michael Thielscher in IJCAI 2017](<https://www.ijcai.org/proceedings/2017/0177.pdf>)

