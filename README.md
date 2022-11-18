# Machine-Learning-Process-Workflow-in-Python
Basic Walk-Through: k-nearest Neighbors This notebook contains several examples that illustrate the machine learning workflow using a dataset of house prices.  We will use the fairly straightforward k-nearest neighbors (KNN) algorithm that allows us to tackle both regression and classification problems.
The Machine Learning Workflow
This chapter starts part 2 of this book where we illustrate how you can use a range of supervised and unsupervised machine learning (ML) models for trading. We will explain each model's assumptions and use cases before we demonstrate relevant applications using various Python libraries. The categories of models that we will cover in parts 2-4 include:

Linear models for the regression and classification of cross-section, time series, and panel data
Generalized additive models, including nonlinear tree-based models, such as decision trees
Ensemble models, including random forest and gradient-boosting machines
Unsupervised linear and nonlinear methods for dimensionality reduction and clustering
Neural network models, including recurrent and convolutional architectures
Reinforcement learning models
We will apply these models to the market, fundamental, and alternative data sources introduced in the first part of this book. We will build on the material covered so far by demonstrating how to embed these models in a trading strategy that translates model signals into trades, how to optimize portfolio, and how to evaluate strategy performance.

There are several aspects that many of these models and their applications have in common. This chapter covers these common aspects so that we can focus on model-specific usage in the following chapters. They include the overarching goal of learning a functional relationship from data by optimizing an objective or loss function. They also include the closely related methods of measuring model performance.

We distinguish between unsupervised and supervised learning and outline use cases for algorithmic trading. We contrast supervised regression and classification problems, the use of supervised learning for statistical inference of relationships between input and output data with its use for the prediction of future outputs. We also illustrate how prediction errors are due to the model's bias or variance, or because of a high noise-to-signal ratio in the data. Most importantly, we present methods to diagnose sources of errors like overfitting and improve your model's performance.

If you are already quite familiar with ML, feel free to skip ahead and dive right into learning how to use ML models to produce and combine alpha factors for an algorithmic trading strategy.

# Content
* The key challenge: Finding the right algorithm for the given task
* Supervised Learning: teaching a task by example
* Unsupervised learning: Exploring data to identify useful patterns
* Use cases for trading strategies: From risk management to text processing
* Reinforcement learning: Learning by doing, one step at a time
* The Machine Learning Workflow
* Code Example: ML workflow with K-nearest neighbors
* Frame the problem: goals & metrics
* Collect & prepare the data
* How to explore, extract and engineer features
* Code Example: Mutual Information
* Select an ML algorithm
* Design and tune the model
* Code Example: Bias-Variance Trade-Off
* How to use cross-validation for model selection
* Code Example: How to implement cross-validation in Python
* Parameter tuning with scikit-learn
* Code Example: Learning and Validation curves with yellowbricks
* Code Example: Parameter tuning using GridSearchCV and pipeline
* Challenges with cross-validation in finance
* Purging, embargoing, and combinatorial CV

# How machine learning from data works
Many definitions of ML revolve around the automated detection of meaningful patterns in data. Two prominent examples include:

AI pioneer Arthur Samuelson defined ML in 1959 as a subfield of computer science that gives computers the ability to learn without being explicitly programmed.
Tom Mitchell, one of the current leaders in the field, pinned down a well-posed learning problem more specifically in 1998: a computer program learns from experience with respect to a task and a performance measure of whether the performance of the task improves with experience (Mitchell, 1997).
Experience is presented to an algorithm in the form of training data. The principal difference to previous attempts at building machines that solve problems is that the rules that an algorithm uses to make decisions are learned from the data as opposed to being programmed by humans as was the case, for example, for expert systems prominent in the 1980s.
