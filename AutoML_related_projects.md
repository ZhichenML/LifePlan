AutoML related projects

1. [Modular Agent Based Evolution Framework](https://github.com/Hintzelab/MABE): RL env: berry harvesting, prisoner dilemma
2. [MarkovNetwork](https://github.com/rhiever/MarkovNetwork)
3. [eos-selfish-herd](https://github.com/adamilab/eos-selfish-herd/blob/master/abeeda/main.cpp): predator-prey, random search

TODO

1. implement ANN based connection

   1) search space: number of ANNs, wiring mask of each ANN, connection weights of each ANN 

   2) fully connected logistic regression trained via policy gradient

   3) HMM trained via policy gradient 

2. implement optimisers based on differential mechanism or RL policy gradient.

3. having multiple brain update, and each brain update has a lookahead step to try the result.

4. mutation w.r.t. the gates

5. ~~selection do not chose elite, to keep diversity/ limit the times an agent can be selected~~

6. constrain the input/output number of each gate

7. ~~empty gates when initialisation~~ not work

8. ~~remember the gate direction~~ done

9. ~~guidance for the first door~~ done

10. ~~turn if encount a wall~~ done

11. ~~Blank, wall, random, wall~~ done

12. ~~revise optimization objective~~ : every move matters, done

13. ~~initial position in front of the first door~~ not right

14. ~~initial position in front of the last door~~

15. experiments: 

    ~~empty genome,~~ not work

    Init genome done

    No door guidance for the first door (door_guidance) done

    no keeping door signals (perception) done

    no return when left/right encount walls (step) done

16. Genome length circular

17. ~~Use deterministic gates~~ done, no improvement, but help debug

18. ~~Copy.deepcopy improves! Otherwise, mutation changes elites and spread bad agents~~ done

19. ~~save the pop, elite sign, fitness~~ done

20. ~~Crossover~~ done 

21. offspring sub-population

22. separate agent and maze envioment, 

23. ~~replace deepcopy with creating new (test first)~~ deepcopy is more efficient 

24. experience gained:

    1. know the work procedure of an agent
    2. Know the optimal solution of a task
    3. set deterministic algorithms for testing in the beginning 
    4. Fitness values does not say every thing, find the unreasonable results and explain
    5. Initial conditions: environment init, agent model init (location, brain), init and reinit in loops
    6. keep in mind higher objectives not just a result to report
    7. objective function choice is important
    8. Nan 0/0, negative numbers' sqrt

    

    ```
    [7.63 0.65 7.63 7.63 0.65 0.65 0.65 0.65]
    [3]
    [7.63 7.63 0.65 7.63 0.65 0.65 0.65 0.65] 
    
    [0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65]
    [7]
    [0.65 0.65 0.65 0.65 0.65 0.65 0.65 0.65]
    ```

    

    ![image-20190429162609813](/Users/zhichengong/Library/Application Support/typora-user-images/image-20190429162609813.png)larger mutation rate 

    

    

    

    

    

    

    

    

    

    

    

    

    $[\delta(r,s,s') = (r + \gamma \arg\max_{a'} Q(s',a')) - Q(s,a)] $

    $[sigmoid(x, W, b) = \frac{1}{1+exp^{W \cdot x + b}} ]​$

    $[a^{*} = \arg\max_{a} Q(s,a) = \arg\max_{a} sigmoid(x, \theta) ]$

    $[cost(R, S, S') = sum(0.5 * \delta(R, S, S')^{2})]$

    $[cost(R, S, S', \theta) = sum(0.5 * \delta(R, S, S', \theta)^{2})]$

    $ \frac{ \partial cost}{ \partial W}, \frac{ \partial cost} {\partial b}$

    $[ W' = W + (-\frac{ \partial cost}{ \partial W} * \alpha) ]
    [ b' = b + (-\frac{ \partial cost}{ \partial b} * \alpha) ]$

    











ALREADY DONE

1. implement Markov gates based agent, optimized via evolutionary algorithm with 

   1) circular genome which encodes a list of Markov gates, as reported in the paper 

   2) direct genome which mutates the Markov gates directly 

   

   Both cases do not pass even small maze.  Poor performance (in contrast to reported result) is hypothetized due to 

   1) large search space (5000 gene sites as reported in the paper) and 

   2) random evolutionary behaviour, where the site mutation, segment copy insertion and deletion have unpredictable influence on the Markov gates.

   

------

TODO

1. literature review for efficient network architecture search
2. ~~revise the agent input/output to be non-identical and sorted,~~  circular, ~~more initial gates, smaller deletion rate~~
3. Gradient trained maze agent (two step/ over-parameterised)
4. search for first year viva topic :
   1. network grounding: Unlike symbols, a network has no intrinsic meaning, it may not be suitable to ground a network to its referer (if there is). I understand it as to decide which type of network to use when encounting which type of task. One-shot Neural Architecture Search does this. It starts from a hyper-network which is composed of multiple-branch of candidate networks, only one of which is activated according to the task in evaluation.

ALREADY DONE

1. Finish the evolutionary maze agent implementation

   1. probabilistic transition
   2. deterministic transition
   3. Result: in 10k generation, the best agent shows a sign to store door cue into one of the memory unit and then read from it to output. 

   ------

   PROBLEM:

   1. The experiment: i use a MLP for memory nodes and a MLP for control nodes respectively.  This helps finding the floppy connection for memory nodes if a node appears in input and output simultaneously.  The problem is we do not have motivation for doing so. The multi-agent input/output structure performs poorly and cannot find reasonable weights. 
   2. The report: I wrote that our goal is to design an agent. What we actually do is searching for input/output structure of agents. This does not have much relevance to NAS and I haven't found enough motivation and research challenge for NAS in general RL model.

   TODO

   1. Finish the report 

   2. Task: Searching for input/output structure for maze agent
      1. use the best trajectory for initialization

   3. first year viva topic: Optimizing agent architecture and policy

   4. subspace of network architecture as representation 3-d tensor with all connections and all operations, distance metric, Bayes optimisation acquisition function, dev accuracy regression, if two networks are similar, they are likely to produce similar performance, [Neural Architecture Search with Bayesian Optimisation and Optimal Transport, by Eric P. Xing et al. in NIPS 2018](http://papers.nips.cc/paper/7472-neural-architecture-search-with-bayesian-optimisation-and-optimal-transport)

   5. partial unobservable state, imitation learning, with Inverse RL,[IRLAS: Inverse Reinforcement Learning for Architecture Search, by Minghao Guo et al. in arXiv 2018](https://arxiv.org/pdf/1812.05285.pdf)

   6. graph cnn for NAS

   7. automata for NAS

   8. in the maze agent, redudant memory nodes are used to induce perturbation as the inputs. Therefore, even if the perception does not change, the agent has possibility to take different actions. Noting that this is in spirit similar to exploration. However, EA cannot take advantage of the state-action pairs in exploration.That explains why the agent works sometimes but cannot be optimal. However, the agent with the network architecture that probability leads to useful random exploration behaviors are kept as the output. That exlains why neural architecture has an influence. This leads to questions:

      1. does current NAS work because of counting on the randomness from redudant nodes?
      2. can we remove the redudant nodes and design effecitive training for supervised learning?
      3. is this decoupled netwrok randomness component a kind of genrative model?
      4. is the classifier assisted by the generative model?
      5. $f(z_t+z'_{t}), z'_t\sim P_{random}​$
      6. if this is true, it does not matter the architecture, just add a generative model to help classification.
      7. [Resnet](<https://www2.securecms.com/CCNeuro/docs-0/5928796768ed3f664d8a2560.pdf>) explicitly perform generative modelling?
      8. random network as feature, like esn
      9. what's the relation between generation and classification?

      For this reason, I remove redudant nodes and use exploration directly. Then train it with RL. 

      exploration for supervised learning, stochastic neural network

   ALREADY DONE

   1. Impletement a Monte Carlo policy gradient to optimize a fully connected MLP with 12 inputs and 6 outputs. A separated input/output structure is not necessary as we are using universal function approximator. However, the learning does not work. 

   2. gate-by-gate grid search:

      input_ids = [[4,7,8,6],[3],[0,3,4,6]]
                  output_ids = [[6,7,8],[6],[6,8]]

   3. constraint grid search

      agent.input_ids = [[0,8,3,6],[1,2,3,6],[2,4,3,6]]
              agent.output_ids = [[6,7,8],[6,7,8],[6,7,8]]

   4. Task: Searching for input/output structure for maze agent

      1. fully connected MLP among input/output nodes, policy gradient trained maze agent, not work
      2. Select one (including none) connection from perception, memory and control, and do differentiable NAS. Over-parameterised super-net as state-of-the-art differential NAS does is not directly applicable, due to their layer-wise one-hot selection.
      3. adjust the agent action strategy
      4. Use two MLPs for memory and control respectively, the input nodes are sampled according to Bernoulli distritbution or a continuous space policy network, worse
         1. two-step learning: 
            1. input/output structure: learn the sampling probability of each input node  output to construct a set of MLPs, by RL. But how to assign credits? In this multi-agent cooperation?
            2. RL trained weights.

      

      

