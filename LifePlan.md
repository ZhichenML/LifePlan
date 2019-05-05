## 						Life Plan



------

### 									RL

------

1. [Memory Augmented Policy Optimization for Program Synthesis and Semantic Parsing](https://arxiv.org/pdf/1807.02322.pdf)
2. [Reinforcement Learning, Fast and Slow, by Matthew Botvinick et al. in Trends in Cognitive Science 2019](https://www.cell.com/action/showPdf?pii=S1364-6613%2819%2930061-0): *Episode RL and Meta RL*
3. [RL2: Fast Reinforcement Learning via Slow Reinforcement Learning, by Yan Duan et al. in arXiv 2016](https://arxiv.org/abs/1611.02779):*Meta RL*
4. [Learning to reinforcement learn, by Jane X Wang et al, in arXiv 2016](https://arxiv.org/abs/1611.05763)

______

### 									Sequence

------

5. [Probabilistic Deterministic Infinite Automata, by David Pfau et al. in NIPS 2010](http://www.stat.columbia.edu/~fwood/Papers/Pfau-NIPS-2010.pdf): Finite state machine is a generative model for sequence, depicted by a 5-tuple $M={Q, \sum,\delta,\pi,q_0}$, where $Q$ is a set of finite states, $\sum$ is a finite alphabet of observable symbols, $\pi: Q \times \sum \rightarrow [0,1]$ is the probability of the next symbol, $\delta: Q\times \sum \rightarrow Q$ is the trasition function from a state-symbol pair to the next state, , and $q_0$ is the initial state distribution. 
   1. When $\pi$ is probabilistic while $\sum$ is deterministic, it termed Probilistic Deterministic Finite Automata (PDFA). Given observed data, $q_0$ and $\delta$, there is no uncertainty about state paths, which saves computations compared to a HMM.  PDFA works as: at current state $q_t$, take action $x_t$ (given), and determinitically transites to next state $q_{t+1}$. N-th order Makov model is a subclass of PDFA with $|\sum|^N$ states.
   2. When both $\pi$ and $\sum​$ are probabilistic, it termed Probabilistic Non-deterministic Finite Automata (PNFA). Expressional ability: n-th Markov model<PDFA< PDIA <HMM<PNFA

