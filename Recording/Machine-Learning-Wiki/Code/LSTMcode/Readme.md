Run with mytest.py. It runs example_Beef() function by default. The output:
![](https://github.com/Scott-Alex/Machine-Learning-Wiki/blob/master/Code/LSTMcode/TrainingLoss.png)

The output layer could be implemented as you wish. 
There are two existing functions in the file, i.e. Euclidean loss layer and linear regression layer.

The code structure is as the following figure: 
![](https://github.com/Scott-Alex/Machine-Learning-Wiki/blob/master/Code/LSTMcode/LSTM_Code_Structure.png)

There is function calling for LSTMParam from both LSTMnetwork and LSTMnode. Otherwise, when the node list is empty, there would be 
no access to network parameters. So the LSTMnetwork should main a copy of parameters.


