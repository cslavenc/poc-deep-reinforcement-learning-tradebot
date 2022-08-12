# poc-deep-reinforcement-learning-tradebot
A custom implementation of the paper [**A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem**](https://arxiv.org/pdf/1706.10059.pdf).  
  
Keep in mind that the original paper was published in 2017. The markets has matured quite
a lot since then. In particular, the trade volume and the total market cap is much higher
for assets like BTC, ETH, ADA etc. The original paper only considered those coins with a high market cap.
They would dynamically determine which assets have a high market cap and adapt what assets to trade. 
Back in these days, this was quite necessary in order to guarantee sufficient liquidity. 
Nowadays, there are assets like BTC, ETH, ADA etc. who always have a really high liquidity, so this was not necessary. 
At the same, these assets do not tend to experience sudden, strong pumps as some microcaps.
This has to be taken into account when evaluating the performance of a neural network as the original authors 
likely chose microcaps dynamically (which have more upside potential temporarily), while 
the current implementation only considers well-established crypto assets who typically do not tend 
to experience sudden pumps like microcaps.  

This project implements the EIIE CNN at different levels of abstraction:

- `simple_eiie_cnn.py` implements the basic EIIE CNN without adding any weights or biases (figure 2, page 11) or the reinforcement environment.
- `eiie_cnn_with_weights.py` implements a custom training loop to make proper use of the portfolio vector memory  
**Note**: it does not make use of the RL reward function yet and trading fees are ignored!
- `deep_rl_eiie_cnn_with_weights.py` uses the reward function as the custom loss function.
This effectively enables the RL environment, since that function is maximized instead of minimized in a traditional setting.  
However, it does not include trading fees currently nor does it choose the minibatches based on a geometric distribution.
Moreover, the cash bias that is concatenated in the neural network is not present as it
still uses **BUSDUSDT** as an asset to simulate cash.
- `eiie_cnn_with_weights_and_cash_bias.py` does not perform so well, since it tries to learn the argmax function basically,
but the concatenated cash bias is a big hindrance to that.  
This is not so grave, since this file was an initial POC for the deep RL EIIE CNN.
- `deep_rl_eiie_cnn_with_weights_and_cash_bias.py`: preliminary results have shown that concatenating a cash bias on the level of logits
leads to very bad results. In fact, it seems to prefer cash over assets most of the time.
From a mathematical perspective, the cash bias was chosen to be 1. and when applying the softmax
function with the other logits, it usually allocates the entire portfolio in cash.  
Since logits can be many magnitudes of order bigger than 1. or negative even, the cash bias
seems like a strange outlier and thus, softmaxing with this artificial cash bias leads to
unusable weights (weight 1 almost always in cash bias, while others are basically 0)
-`deep_rl_eiie_cnn_with_weights_and_trading_fees.py` did not show an improvement in preliminary test trials.  
In fact, it performed worse and it was not able to avoid larger drops. Qualitatively, the graph looked similar to the version without trading fees.  
- `deep_rl_eiie_cnn_with_weights_online_training.py` implements online training functionality.
In preliminary experiments, a small improvement could be observed. Also, tuning the hyperparameters (online epochs and window size) is crucial.  
- TODO


## GET STARTED
- first, run `generate_data_for_trading_pair.py` and make sure to have the folder `src/datasets`. Feel free to delete the placeholder file `.empty`.  
This creates the csv files that are used for static backtesting throughout this project.
- then, run a neural network file of your choice for analysis. Note that depending which neural network you are running, they might implement only
some aspects of the actual neural network in the paper as well as evaluate the data differently.

Make sure, you are using the CPU mode on laptop. Optionally, you can use the GPU mode where the support has been implemented.

## NOTES
Preliminary results have shown that **ignoring to divide by the size of the individual rewards list**
leads to basically the same results, but upside seems to have more potential.  

**Feature Engineering**:
- in preliminary experiments, using volume did not seem to really improve learning
- moreover, using the middle, higher, lower Bollinger Bands also did not improve learning

**Flakiness**:
- Sometimes, the loss gets stuck at the same value from the very beginning.
Usually, restarting helps
- Sometimes, the loss or gradients are `nan`. Closing the current ipython console
and running the file in a new one helps. The `nan` value used to occur when using logits as input
values for the custom loss function. Using the corresponding weights did not lead to that problem anymore. 
This is probably due to things being cached in the background and after changes are made
tensorflow becomes confused.  

**tf.float32/64**:
- there are some issues around tf.float64, so tf.float32 has been used instead where applicable

**epochs**:
- it is unclear yet, what increasing epochs really do
- with the current final model (weights, custom loss function, **no** separate cash bias),
more epochs look qualitatively the same with a potential improvement, but they take longer to train on CPU
- epochs compared were 100 vs 1000

**Using w_t-1 in the cumulated return loss function**:
- since I did not find a way to simply save the previous portfolio weights and use the in the next
step during the "loss" calculation (gradients were always `None`), I used a trick, where the current
predicted portfolio weights are used for the next priceRelativeVectors (at t+1).  
The operations for forward pass and backpropagation are not available when iterating within a loop,
as you can only save the tensor of the portfolio weights. Tensorflow does not know what to 
do with that exactly trying to compute gradients for the following period, since
the operations are not recorded anymore. Only those operations are available that are
within the same loop/GradientTape.

**Cash Bias**:
- concatenating the cash bias as proposed in the paper did not yield good results in preliminary trials.
- it seems that using cash as another asset (can be simulated as **BUSDUSDT** for example) gives much more meaningful weight distributions

**Starting weights w_0**:
- optimal weights as starting weights during minibatches seem to work better than using weights, where everything is in cash initially

**15mins timeframe VS 30mins timeframe**:
- the paper uses a 30mins timeframe. When choosing a large train dataset (around 6 weeks) it has a reasonably good performance.
In the end, it still underperformed the 15mins timeframe in initial experiments.