# Iris LDA Classifier
This is a simple implementation of a Linear Discriminant Analysis (LDA) classifier for the Iris dataset. The purpose of this project is to demonstrate the use of LDA for classification tasks.

## Dataset

The Iris dataset is a well-known dataset in the machine learning community. It consists of 150 samples of iris flowers, each with four features (sepal length, sepal width, petal length, and petal width) and a corresponding target class (Iris setosa, Iris versicolor, or Iris virginica).

## Algorithm
Linear Discrimnant Analysis (LDA) as a Classifier:

    1. It is a discriminative Technique which can be used as both Classifier and Dimensionality Reduction Technique.
    2. Assumptions:
        1. All independent features must be continuous.
        2. Also, The independent fetaures must follow Multivariate Guassian Distribution.
        3. All Class Covariance Matrix are Equal.
                  Σ1 = Σ2 = ...... = Σk [k classes in dependent feature].

#### Given: D = { xi , yi} i=1n
#### Objective: f: Rd → {Ci,...,Ck}
#### Steps :

    1. Find Class Mean and Covariance Matrix.
    2. Calculate Linear score function for each class. 
    3. Prediciton Rule: Class with highest Linear Score is the yhat for given Xi
    
## Requirements

This project requires Python 3.x and the following libraries:

    1. NumPy
    2. Pandas
    3. Scikit-learn
    
## License

This project is licensed under the MIT License - see the LICENSE file for details.
