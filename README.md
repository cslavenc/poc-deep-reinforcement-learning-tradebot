# poc-deep-reinforcement-learning-tradebot
## NOTES
- **custom gradient, functional API, Sequential API**: it seems that it is quite tricky to get the custom gradient...
    - if using the functional API, it is easy to create the CNN network, but the tensors are **KerasTensor** 
    instead of **tf.Tensor** which is expected in the **tape.gradient()** part
    - if using the Sequential API, it becomes difficult to glue the weights and bias layers to the main neural network.
    - **some sort of refactoring is necessary to get gradient and losses**

## TODO
- **I need to get an overview of what works and what not...**
- add comment where a variable first appeared in the paper for future reference
- when it runs properly: FINISH ALL TODOS
- enforce typing in functions to make it easier to understand
- TODO(investigate) : what about BTC markets as currency?