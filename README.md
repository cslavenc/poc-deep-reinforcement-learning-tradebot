# poc-deep-reinforcement-learning-tradebot
A custom implementation of the paper [**A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem**](https://arxiv.org/pdf/1706.10059.pdf).  
  
Keep in mind that the original paper was published in 2017. The markets have matured quite
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

- `deep_rl_eiie_cnn_with_weights_and_trading_fees.py` did not show an improvement in preliminary test trials.  
In fact, it performed worse and it was not able to avoid larger drops. Qualitatively, the graph looked similar to the version without trading fees.  

- `deep_rl_eiie_cnn_with_weights_online_training.py` implements online training functionality.
In preliminary experiments, a small improvement could be observed. Also, tuning the hyperparameters (online epochs and window size) is crucial.  

- `deep_rl_eiie_cnn_with_weights_online_training_gpu.py` implements the GPU accelerated version.   
Keep in mind that it only activates a GPU and updates the shapes in the neural network. 
Further work is necessary: the train and loss function need to be annotated with `@tf.function` 
and the for-loops have to be translated into a tf.while_loop and no list.append() calls are allowed. 
list.append() calls should be replaced with `tf.TensorArray()`. Moreover, all variables should be wrapped either with `tf.constant()` or `tf.Variable`.  
These graph optimizations will leverage the full power of a GPU. On the other hand, 
the improved performance might not be as high, since data still has to be copied from CPU to GPU and back to CPU, which adds a considerable overhead.  
Therefore, this implementation has not been continued.

- `deep_rl_eiie_cnn_with_weights_online_training_safety_mechanisms.py`: **The final version** implements custom safety mechanisms. 
These safety mechanisms are tradestops that should simulate holding cash to protect against large downside, 
such as a sudden BTC crash.  
Parameters were chosen in a way to identify sustained downtrends (bear markets) with respect to the **portfolio value**. 
Sometimes, a safety measure is activated after a sharp drop as well. Parameter tuning depends on the predicted time period and timeframe.  
There are two mechanisms for continuous tradestops that one can evaluate:
  - predict one timestep, get the weights, calculate portfolio value and evaluate whether a tradestop is needed or not. 
  Evaluating every single timestep separately leads to very long online training durations (if using a test period of more than a year, this can easily last days), 
  since online training is executed after every single timestep (instead of waiting N timesteps and using training on aggregated data). 
  Keep in mind that the online data trainset advances by one datapoint everytime which can lead to over 100000 epochs!   
  - predict many timesteps, get the weights, calculate the portfolio value, evaluate a potential tradestop and 
  collect the test data as the new online train data. This approach is equally valid as the previous one. 
  As there are as many online train periods as in those files without safety mechanisms, the entire simulation is faster.  
The second approach has been preferred as it is much faster on CPU alone.  
**Note**: Zooming into a tradestop period, there is still a very slight increase due to cash not being valued at exactly 1, but sometimes at 1.00001 etc.

- `deep_rl_eiie_cnn_with_weights_online_training_discrete.py` performs online training on every single new datapoint, 
which leads to a massively bigger training time. Keep in mind that this POC is without safety mechanisms.  

## GET STARTED
### INSTALLATION
- tensorflow=2.9.1
  - cudatoolkit=11.2 (GPU)
  - cudnn=8.1.0 (GPU)
- pandas-ta=0.3.14

GPU tests were performed on NVIDIA® GeForce® RTX 2060 SUPER:  
NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7  
with cudatoolkit=11.2, cudnn=8.1.0 and cupy-cuda11x

### RUNNING THE SIMULATION
- First, run `generate_data_for_trading_pair.py` and make sure to have the folder `src/datasets`. Feel free to delete the placeholder file `.empty`.  
This creates the csv files that are used for static backtesting throughout this project.
- then, run a neural network file of your choice for analysis. Note that depending which neural network you are running, they might implement only
some aspects of the actual neural network in the paper as well as evaluate the data differently.  

Make sure, you are using the CPU mode on laptop. Optionally, you can use the GPU mode where the support has been implemented.  

## Training
The network is sensitive to the train period. It should be chosen carefully and with empirical testing. 
Based on initial testing, it seems to be more favourable to chose a train period during a bull market.
These dates have been used as start dates for training:

- Good: `datetime.datetime(2020,12,24,0,0,0)` before the distribution phase (bull market)
- Underperforming: `datetime.datetime(2020,9,7,0,0,0)` shortly before the start of the bull market

### Online Training
**TODO**: try tradestop of 3 days? or 4 even?  
Shorter weekly increments for retraining make the neural network adapt better to the current 
situation. On the other hand, this might make the neural network too reactive or have too many tradestops 
during bullish periods (1 week, 2 days) when fewer tradestop signals would have been okay. On the other hand, 
retraining the neural network more frequently gives better range bound results and apparently, 
they tend to capture intermediate bullish times during long extended bear markets after the obvious 
downturn is over.

- weeksIncrement: 6, 3, 4, what about 1 or 2 weeks?
  - 6 weeks with a tradestop of 2 days performs significantly worse than 3 weeks with 2 days tradestop.
  - 3 weeks with 2 days tradestop takes longer during the simulation, but it performs really well.
    During the real-time scenario it should actually be quicker since preparing data takes less time.
  - whether training on weekly increments of 3 or 6 takes about the same time, since the number of train steps adapts accordingly, 
    but smaller weekly increments prepare data more often which seems to last a bit longer in total.
  - An interesting find was the **1 week, 2 days** tradestop configuration. During the bull market, its 
    tradestops were a bit too long at the initial phases of it, but it performed well during the bear market 
    as it was basically rangebound and even better during intermediate runs by BTC about 1 year after 
    the obvious onset of the bear market. In thus far, it outperformed the 3 weeks, 2 days during the 
    **later stages of the bear market** (small intermediate bull markets), while the other configuration 
    remained range bound during the same period.

- **tradestop duration: 1d vs 2d**
In general, longer tradestops encourage holding everything in cash longer. Often, a bad period lasts 
for some time and shorter tradestops cannot capture the entire length properly and begin to trade too early, 
even though the tradestops reactivate quite quickly again, some trades are executed during a downturn anyway 
and lead to loss. Longer tradestops tend to avoid downturn periods better. Right now, a 2 days tradestop 
seems to be outperforming the 1 day tradestop.
  - **1 day**: While a tradestop of 1 day works very well during bullish periods it seems to be underperform during clearly bearish periods. 
    This can be fixed manually by turning off the tradebot during clearly bearish periods.
  - **2 days**: A tradestop for 2 days with weekly increments of 3 seems to perform better during the bear market as it goes 
    quite sideways and is thus more stable during a clear bear market. During a bull market, 
    it even outperformed the 1 day tradestop configuration.
  - **3 days**:

## NOTES
Preliminary results have shown that **ignoring to divide by the size of the individual rewards list**
leads to basically the same results, but upside potential seems to be higher

### Choosing Crypto Assets
- crypto assets were chosen using the following criteria:
  - **high liquidity**: an asset must possess a high daily trade volume
  - **instant selling and buying**: thanks to high liquidity, an asset can be sold or bought immediately
  - **zero slippage**: the asset is definitely sold/bought at the chosen **spot** price
  - **zero market impact**: buying or selling an asset should not influence the market at all
  Therefore, an initial candidate list has been identified by sorting assets on binance by descending volume. 
  The final list has been chosen through testing.

### 15mins timeframe VS 30mins timeframe
- The paper uses a 30mins timeframe. When choosing a large train dataset (around 6 weeks) it has a reasonably good performance.
In the end, it still underperformed the 15mins timeframe in initial experiments.  
Therefore, the 15mins timeframe has been used in all experiments. Moreover, the hyperparameters are tuned for 
the 15mins timeframe.  

### Feature Engineering
- in preliminary experiments, using volume did not seem to really improve learning
- moreover, using the middle, higher, lower Bollinger Bands also did not improve learning

### Flakiness
- Sometimes, the loss gets stuck at the same value from the very beginning when using a simple EIIE CNN 
or an EIIE CNN with weights. Restarting helps usually.
- Sometimes, when training a neural network after some changes have been made to its topology can give the same results as the previous version. 
This is probably due to TensorFlow not correctly updating the graph structure. 
Running the neural network again (potentially in a new ipython console) should solve this confusion.
- In very rare circumstances, the predictions can be really bad or strange. Since neural network weights are initialized randomly in the beginning,
very unfavourable initial values can lead to a broken prediction. Retraining from scratch usually helps.

### CuPy vs numpy
- initial trials showed a decline in performance when generating the dataset (X_tensor) when using `cupy` probably due to the GPU overhead.
- `cupy` is essentially the GPU version of `numpy`. It still has some pecularities about it though...
- normally, `import cupy as np` is preferred as the API is basically the same as numpy.
- `cupy` requires cupy arrays instead of lists though. Thus: if myList is a python list, 
numpy usually has no trouble with it. On the other hand, cupy requires a conversion 
(myList = cupy.asarray(myList)) to perform other operations such as swapaxes etc.
- `cupy` and `numpy` can often interfere with each other. It was only meant to speedup 
data preparation time. If in doubt, simply use `numpy`.

### tf.float32/64
- there are some issues around tf.float64, so tf.float32 has been used instead where applicable

### epochs
- epochs for 100 and 1000 showed similar, but 100 epochs was faster in training. Most experiments later on used 300 epochs
- with the current final model (weights, custom loss function, **no** separate cash bias),
more epochs look qualitatively the same with a potential improvement, but they take longer to train on CPU
- epochs compared were 100 vs 1000

### Using w_t-1 in the cumulated return loss function
- since I did not find a way to simply save the previous portfolio weights and use them in the next
step during the "loss" calculation (gradients were always `None`), I used a trick, where the current
predicted portfolio weights are used for the next priceRelativeVectors (at t+1).  
The operations for forward pass and backpropagation are not available when iterating within a loop,
as you can only save the tensor of the portfolio weights. Tensorflow does not know what to 
do with that exactly trying to compute gradients for the following period, since
the operations are not recorded anymore. Only those operations are available that are
within the same loop/GradientTape.

### Cash Bias
The paper suggests to use a separat cash layer as cash bias. This design decision was likely due to the fact 
that it was written in 2017 and at that time, there were no stablecoins to be used as cash.

- concatenating the cash bias as proposed in the paper did not yield good results in preliminary trials.
- it seems that using cash as another asset (can be simulated as **BUSDUSDT** for example) gives much more meaningful weight distributions

### Starting weights w_0
- optimal weights as starting weights during minibatches seem to work better than using weights, 
where everything is in cash initially
