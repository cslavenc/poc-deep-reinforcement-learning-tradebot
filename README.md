# poc-deep-reinforcement-learning-tradebot
A custom implementation of the paper [**A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem**](https://arxiv.org/pdf/1706.10059.pdf).  
This project implements the EIIE CNN at different levels of abstraction:
- simple_eiie_cnn.py implements the basic EIIE CNN without adding any weights or biases (figure 2, page 11) or the reinforcement environment.
- eiie_cnn_with_weights.py implements a custom training loop to make proper use of the portfolio vector memory  
**Note**: it does not make use of the RL reward function yet and trading fees are ignored!
- TODO


## GET STARTED
- first, run `generate_data_for_trading_pair.py` and make sure to have the folder `src/datasets`. Feel free to delete the placeholder file `.empty`.  
This creates the csv files that are used for static backtesting throughout this project.
- then, run a neural network file of your choice for analysis. Note that depending which neural network you are running, they might implement only
some aspects of the actual neural network in the paper as well as evaluate the data differently.

Make sure, you are using the CPU mode on laptop. Optionally, you can use the GPU mode where the support has been implemented.

## NOTES
**Flakiness**:
- Sometimes, the loss gets stuck at the same value from the very beginning.
Usually, restarting helps
- Sometimes, the loss or gradients are `nan`. Closing the current ipython console
and running the file in a new one helps.  
This is probably due to things being cached in the background and after changes are made
tensorflow becomes confused.