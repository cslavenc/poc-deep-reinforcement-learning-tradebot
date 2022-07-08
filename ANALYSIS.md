# ANALYSIS OF DEEP PORTFOLIO MANAGEMENT PAPER IMPLEMENTATION

## General Information
Another user provides the pseudocode for the learning part [see github](https://github.com/EthanBraun/DeepPortfolioManagement/blob/master/pseudocode).  
The same user also has a simplified starting point for the CNN (using keras, albeit it is still `tensorflow v1`):  
[see github](https://github.com/EthanBraun/DeepPortfolioManagement/blob/master/backtest.py#L36)

## Main folders
- root src (pgportfolio)
  - constants.py contains often used constants, net_config.json contains a json config to train also including tuned hyperparams, which is used in main.py
  - main.py simply starts training an agent based on json input config and plots in the end
- autotrain
  - used for training one or all agents
  - it also publishes things to tensorboard (probably won't use that anyway)
  - it also loads the training config from a json file
- learn
  - the main neural network directory
  - it first generates a general neural network, then it becomes a CNN in a different class
  - the CNN is inputted into the NN agent. This agent is then used in other operations elsewhere
  - the class CNN contains the CNN architecture (**keep in mind that they use the deprecated `tensorflow v1` from the past**)
  - the NN agent initializes many other things such as portfolio weights etc. and has helper functions for training the network amongst other things, besides initializing the CNN
  - the tradertrainer takes an agent and other information and basically trains the NN agent on train data and outputs results as csv and prepares the tensorboard
- marketdata
  - get market data, a bit like a controller
  - poloniex.py has the api calls, coinlist.py creates a top N coinlist to be used for trading
  - it also creates a DB table via sqlite in globalmarketdata.py via the class HistoryManager
  - this data is subsequently used, processed and stored in datamatrices.py, which also prepares train and test sets
- resultprocess
  - simply plots things and also compares how all the different algos performed compared to each other
- tdagent
  - contains algos for traditional tradebots (NO Neural Networks!)
- tools
  - basically a utils directory with tools to make some technical processes better (filling NANs, adding missing layer, helping out with creating a CNN, RNN etc.)
- trade
  - utils for executing a trading step basically
  - it seems to be more of an infrastructural class

## Analysis of individual files and folders

### autotrain
- **NOTE(!)**: likely irrelevant for me
- generate.py prepares some directories etc
- training.py trains an agent via a config file

### learn
- **WARNING(!)**: It uses the library `tflearn` which is a high level API of `TensorFlow V1`, therefore it is deprecated as tensorflow v2+ should be used
  - I can probably use `keras` instead of `tflearn` now
- network.py specifies a general network and it has the class CNN which specifies the CNN
  - the CNN class uses the old style for creating a neural network, meaning, it is much easier to do in `keras`
  - It will be somewhat tedious to update the code to the tensorflow v2 and **you have to make sure, you get the DETAILS right**
- nnagent.py takes the CNN as input and defines more configs such as loss functions etc. and is responsible for dealing with portfolio weights more concretely
- tradertrainer.py effectively executes the CNN and trains it

### marketdata
- poloniex.py has to be adapted for the binance exchange
- coinlist.py might not be as necessary, since i will use a pre-defined list anyway
  - pre-defined list will include: BTC, ETH, ADA, AVAX, MATIC, DOT, SOL, BNB, XRP, ATOM, LINK, PAXG
  - whether to use their BTC or USDT valuations or both remains to be determined
  - whether to use additional top volume coins (those who are pumping or have generally high volume) remains to be determined
- globalmarketdata.py fills up the database (they don't seem to use csv files)
  - it is basically a holder for configs and data
- marketdata.py is responsible for managing the train and test sets as well as keeping track of time and the weights
  - PVM is their array to keep the weights w, which are also set and updated in other functions
    - **NOTE(!)**: this is something I need to keep track of very closely, as this is crucial for proper portfolio balancing/selection
  - it also keeps track of the batches, since it uses batches for training
    - not sure if I have to keep track of the batches myself or simply let tensorflow do it...
  - replaybuffer.py keeps track of the weights and data as well and is involved in generating the next batch
    - not quite clear to me what this does

### resultprocess
- **NOTE(!)**: likely irrelevant for me as this is only used to plot and compared different algos with each other
- plot.py plots and compares the performance of different algos

### tdagent
- **NOTE(!)**: As I will be working with neural networks mainly, I do not need to take this functionality in account - only for comparing performances if at all
- the algorithms subdirectory contains many files for traditional tradebot algorithms
- tdagent.py basically initializes a traditional algo and contains some helper functions

### tools
- configprocess.py contains some helper functions for tasks like filling in NANs, creating NN layers etc.
- data.py has more helper functions as well as functions to normalize matrices and tensors
- indicator.py is mainly used in plot.py to plot things like positive and negative days, sharpe ratio etc.
- shortcut.py prepares an agent based on an algo name or number, mainly used in main.py and plot.py
- trade.py contains some more utils to prepare coinlists and test data sets etc.
  - `calculate_pv_after_commission` seems to be the only important function appearing in backtest.py as well

### trade
- **NOTE(!)**: I probably do not need this class in that fashion
- trader.py executes trading steps and is involved in backtesting
  - it is also given the trade type, such as NN or another traditional algo
- backtest.py backtests things, although it is not quite clear to me what it is good for



## Paper Analysis (incomplete)
### Introduction
At the heart of the algorithm lie Ensemble of Identical Independent Evaluators (EIIE). They take into consideration the market history **as well as the portfolio weights from the previous trading period**.  
When the portfolio weights change, some assets are bought in more, others sold. Portfolio weights are saved in a portfolio vector memory array (PVM) from each period. Thus,
it this array becomes bigger every time a new period has new weights. The **Reward function R*- is the explicit average of the periodic log-returns.  

- their trading period is **30 mins**

### Problem Definition
- trading period T is 30 mins
- for large market cap coins with high trade volume, one can assume zero slippage and instant buying or selling (sell at open price which is previous close price)


