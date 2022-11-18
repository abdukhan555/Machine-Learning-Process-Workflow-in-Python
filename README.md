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

Content
How machine learning from data works
The key challenge: Finding the right algorithm for the given task
Supervised Learning: teaching a task by example
Unsupervised learning: Exploring data to identify useful patterns
Use cases for trading strategies: From risk management to text processing
Reinforcement learning: Learning by doing, one step at a time
The Machine Learning Workflow
Code Example: ML workflow with K-nearest neighbors
Frame the problem: goals & metrics
Collect & prepare the data
How to explore, extract and engineer features
Code Example: Mutual Information
Select an ML algorithm
Design and tune the model
Code Example: Bias-Variance Trade-Off
How to use cross-validation for model selection
Code Example: How to implement cross-validation in Python
Parameter tuning with scikit-learn
Code Example: Learning and Validation curves with yellowbricks
Code Example: Parameter tuning using GridSearchCV and pipeline
Challenges with cross-validation in finance
Purging, embargoing, and combinatorial CV
How machine learning from data works
Many definitions of ML revolve around the automated detection of meaningful patterns in data. Two prominent examples include:

AI pioneer Arthur Samuelson defined ML in 1959 as a subfield of computer science that gives computers the ability to learn without being explicitly programmed.
Tom Mitchell, one of the current leaders in the field, pinned down a well-posed learning problem more specifically in 1998: a computer program learns from experience with respect to a task and a performance measure of whether the performance of the task improves with experience (Mitchell, 1997).
Experience is presented to an algorithm in the form of training data. The principal difference to previous attempts at building machines that solve problems is that the rules that an algorithm uses to make decisions are learned from the data as opposed to being programmed by humans as was the case, for example, for expert systems prominent in the 1980s.

Recommended textbooks that cover a wide range of algorithms and general applications include

An Introduction to Statistical Learning, James et al (2013)
The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Hastie, Tibshirani, and Friedman (2009)
Pattern Recognition and Machine Learning, Bishop (2006)
Machine Learning, Mitchell (1997).
The key challenge: Finding the right algorithm for the given task
The key challenge of automated learning is to identify patterns in the training data that are meaningful when generalizing the model's learning to new data. There are a large number of potential patterns that a model could identify, while the training data only constitutes a sample of the larger set of phenomena that the algorithm may encounter when performing the task in the future.

Supervised Learning: teaching a task by example
Supervised learning is the most commonly used type of ML. We will dedicate most of the chapters in this book to applications in this category. The term supervised implies the presence of an outcome variable that guides the learning process—that is, it teaches the algorithm the correct solution to the task at hand. Supervised learning aims to capture a functional input-output relationship from individual samples that reflect this relationship and to apply its learning by making valid statements about new data.

Unsupervised learning: Exploring data to identify useful patterns
When solving an unsupervised learning problem, we only observe the features and have no measurements of the outcome. Instead of predicting future outcomes or inferring relationships among variables, unsupervised algorithms aim to identify structure in the input that permits a new representation of the information contained in the data.

Use cases for trading strategies: From risk management to text processing
There are numerous trading use cases for unsupervised learning that we will cover in later chapters:

Grouping together securities with similar risk and return characteristics (see hierarchical risk parity in Chapter 13
Finding a small number of risk factors driving the performance of a much larger number of securities using principal component analysis) or autoencoders (Chapter 20
Identifying latent topics in a body of documents (for example, earnings call transcripts) that comprise the most important aspects of those documents (Chapter 15)
Reinforcement learning: Learning by doing, one step at a time
Reinforcement learning (RL) is the third type of ML. It centers on an agent that needs to pick an action at each time step based on information provided by the environment. The agent could be a self-driving car, a program playing a board game or a video game, or a trading strategy operating in a certain security market.

You find an excellent introduction in Sutton and Barto, 2018.

The Machine Learning Workflow
Developing an ML solution requires a systematic approach to maximize the chances of success while proceeding efficiently. It is also important to make the process transparent and replicable to facilitate collaboration, maintenance, and subsequent refinements.

The process is iterative throughout, and the effort at different stages will vary according to the project. Nonethelesee, this process should generally include the following steps:

Frame the problem, identify a target metric, and define success
Source, clean, and validate the data
Understand your data and generate informative features
Pick one or more machine learning algorithms suitable for your data
Train, test, and tune your models
Use your model to solve the original problem
Code Example: ML workflow with K-nearest neighbors
The notebook machine_learning_workflow contains several examples that illustrate the machine learning workflow using a simple dataset of house prices.

sklearn Documentation
k-nearest neighbors tutorial and visualization
Frame the problem: goals & metrics
The starting point for any machine learning exercise is the ultimate use case it aims to address. Sometimes, this goal will be statistical inference in order to identify an association between variables or even a causal relationship. Most frequently, however, the goal will be the direct prediction of an outcome to yield a trading signal.

Collect & prepare the data
We addressed the sourcing of market and fundamental data in Chapter 2, and for alternative data in Chapter 3. We will continue to work with various examples of these sources as we illustrate the application of the various models in later chapters.

How to explore, extract and engineer features
Understanding the distribution of individual variables and the relationships among outcomes and features is the basis for picking a suitable algorithm. This typically starts with visualizations such as scatter plots, as illustrated in the companion notebook (and shown in the following image), but also includes numerical evaluations ranging from linear metrics, such as the correlation, to nonlinear statistics, such as the Spearman rank correlation coefficient that we encountered when we introduced the information coefficient. It also includes information-theoretic measures, such as mutual information

Code Example: Mutual Information
The notebook mutual_information applies information theory to the financial data we created in the notebook feature_engineering, in the chapter [Alpha Factors – Research and Evaluation]((../04_alpha_factor_research).

Select an ML algorithm
The remainder of this book will introduce several model families, ranging from linear models, which make fairly strong assumptions about the nature of the functional relationship between input and output variables, to deep neural networks, which make very few assumptions.

Design and tune the model
The ML process includes steps to diagnose and manage model complexity based on estimates of the model's generalization error. An unbiased estimate requires a statistically sound and efficient procedure, as well as error metrics that align with the output variable type, which also determines whether we are dealing with a regression, classification, or ranking problem.

Code Example: Bias-Variance Trade-Off
The errors that an ML model makes when predicting outcomes for new input data can be broken down into reducible and irreducible parts. The irreducible part is due to random variation (noise) in the data that is not measured, such as relevant but missing variables or natural variation.

The notebook bias_variance demonstrates overfitting by approximating a cosine function using increasingly complex polynomials and measuring the in-sample error. It draws 10 random samples with some added noise (n = 30) to learn a polynomial of varying complexity. Each time, the model predicts new data points and we capture the mean-squared error for these predictions, as well as the standard deviation of these errors. It goes on to illustrate the impact of overfitting versus underfitting by trying to learn a Taylor series approximation of the cosine function of ninth degree with some added noise. In the following diagram, we draw random samples of the true function and fit polynomials that underfit, overfit, and provide an approximately correct degree of flexibility.

How to use cross-validation for model selection
When several candidate models (that is, algorithms) are available for your use case, the act of choosing one of them is called the model selection problem. Model selection aims to identify the model that will produce the lowest prediction error given new data.

Code Example: How to implement cross-validation in Python
The script cross_validation illustrates various options for splitting data into training and test sets by showing how the indices of a mock dataset with ten observations are assigned to the train and test set.

Parameter tuning with scikit-learn
Model selection typically involves repeated cross-validation of the out-of-sample performance of models using different algorithms (such as linear regression and random forest) or different configurations. Different configurations may involve changes to hyperparameters or the inclusion or exclusion of different variables.

Code Example: Learning and Validation curves with yellowbricks
The notebook machine_learning_workflow) demonstrates the use of learning and validation illustrates the use of various model selection techniques.

Yellowbrick: Machine Learning Visualization docs
Code Example: Parameter tuning using GridSearchCV and pipeline
Since hyperparameter tuning is a key ingredient of the machine learning workflow, there are tools to automate this process. The sklearn library includes a GridSearchCV interface that cross-validates all combinations of parameters in parallel, captures the result, and automatically trains the model using the parameter setting that performed best during cross-validation on the full dataset.

In practice, the training and validation sets often require some processing prior to cross-validation. Scikit-learn offers the Pipeline to also automate any requisite feature-processing steps in the automated hyperparameter tuning facilitated by GridSearchCV.

The implementation examples in the included machine_learning_workflow.ipynb notebook to see these tools in action.

The notebook machine_learning_workflow) also demonstrates the use of these tools.

Challenges with cross-validation in finance
A key assumption for the cross-validation methods discussed so far is the independent and identical (iid) distribution of the samples available for training. For financial data, this is often not the case. On the contrary, financial data is neither independently nor identically distributed because of serial correlation and time-varying standard deviation, also known as heteroskedasticity

Purging, embargoing, and combinatorial CV
For financial data, labels are often derived from overlapping data points as returns are computed from prices in multiple periods. In the context of trading strategies, the results of a model's prediction, which may imply taking a position in an asset, may only be known later, when this decision is evaluated—for example, when a position is closed out.

The resulting risks include the leaking of information from the test into the training set, likely leading to an artificially inflated performance that needs to be addressed by ensuring that all data is point-in-time—that is, truly available and known at the time it is used as the input for a model. Several methods have been proposed by Marcos Lopez de Prado in Advances in Financial Machine Learning to address these challenges of financial data for cross-validation:

Purging: Eliminate training data points where the evaluation occurs after the prediction of a point-in-time data point in the validation set to avoid look-ahead bias.
Embargoing: Further eliminate training samples that follow a test period.
