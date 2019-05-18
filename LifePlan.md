## 						Life Plan



------

### 									RL

------

1. [Reinforcement Learning by Probability Matching](https://papers.nips.cc/paper/1042-reinforcement-learning-by-probability-matching.pdf): it shows RL algorithm limits ensemble of networks to only one of them and cut-off the rest. Probability Matching cost function based on energy function can avoid this problem.
2. [Memory Augmented Policy Optimization for Program Synthesis and Semantic Parsing](https://arxiv.org/pdf/1807.02322.pdf)
3. [Reinforcement Learning, Fast and Slow, by Matthew Botvinick et al. in Trends in Cognitive Science 2019](https://www.cell.com/action/showPdf?pii=S1364-6613%2819%2930061-0): *Episode RL and Meta RL*
4. [RL2: Fast Reinforcement Learning via Slow Reinforcement Learning, by Yan Duan et al. in arXiv 2016](https://arxiv.org/abs/1611.02779):*Meta RL*
5. [Learning to reinforcement learn, by Jane X Wang et al, in arXiv 2016](https://arxiv.org/abs/1611.05763)
6. [Deep Reinforcement Learning that Matters by Peter Henderson et al. in AAAI 2018](https://arxiv.org/abs/1709.06560)
7. Unobservable imitation learning

______

### 									Sequence

------

5. [Probabilistic Deterministic Infinite Automata, by David Pfau et al. in NIPS 2010](http://www.stat.columbia.edu/~fwood/Papers/Pfau-NIPS-2010.pdf): Finite state machine is a generative model for sequence, depicted by a 5-tuple $M=\{Q, \sum,\delta,\pi,q_0\}​$, where $Q​$ is a set of finite states, $\sum​$ is a finite alphabet of observable symbols, $\pi: Q \times \sum \rightarrow [0,1]​$ is the probability of the next symbol, $\delta: Q\times \sum \rightarrow Q​$ is the trasition function from a state-symbol pair to the next state, , and $q_0​$ is the initial state distribution. 

   1. When $\pi$ is probabilistic while $\delta$ is deterministic, it termed Probilistic Deterministic Finite Automata (PDFA). Given observed data, $q_0$ and $\delta$, there is no uncertainty about state paths, which saves computations compared to a HMM.  PDFA works as: at current state $q_t$, take action $x_t$ (given), and determinitically transites to next state $q_{t+1}$. N-th order Makov model is a subclass of PDFA with $|\sum|^N$ states (only observations).

      The maze agent works as a 1st order Markov model , e.g. : $[observation_t, memory_{1\sim t-1}, move_{t-1}]-[memory_t, move_t]-[observation_{t+1},memory_{1\sim t}, move_t]​$.

      ​    	      State	 $-(little$ $dynamics)-$       observation     $-(deterministic)-$           State

   2. When both $\pi​$ and $\delta​$ are probabilistic, it termed Probabilistic Non-deterministic Finite Automata (PNFA). Expressional ability: n-th order Markov model<PDFA< PDIA <HMM<PNFA.

      