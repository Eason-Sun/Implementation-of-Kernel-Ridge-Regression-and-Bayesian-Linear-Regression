# Implementation-of-Kernel-Ridge-Regression-and-Bayesian-Linear-Regression

## Objective:
This project implements two Non-linear Regression algorithms from scratch (without using any existing machine learning libraries e.g. sklearn):  
1) Kernel Ridge Regression
2) Bayesian Linear Regression   

Both approach adopt monomial basis function up to degree d.

## Dataset:
The data used in this project corresponds to samples from a 3D surface.

### Format:
There is one row per data instance and one column per attribute. The targets are real values. The training set is already divided into 10 subsets for 10-fold cross validation.

### Data Visualization:
![Capture](https://user-images.githubusercontent.com/29167705/63808707-95a68280-c8ee-11e9-9dbf-cba62fbe893f.JPG)

## Mean Squared Error Comparision (w.r.t degree d):
1) Kernel Ridge Regression:

![cs_680_a3_y494sun](https://user-images.githubusercontent.com/29167705/63809312-fbdfd500-c8ef-11e9-9480-2b4c4af8a79a.jpg)

2) Bayesian Linear Regression:   

![cs_680_a3_y494sun](https://user-images.githubusercontent.com/29167705/63809337-0f8b3b80-c8f0-11e9-81bd-0c32787dbbbb.jpg)

## Running Time Comparision (w.r.t degree d):
1) Kernel Ridge Regression: Since kernel technique is applied in this case, the time efficiency for regularized generalized linear regression is Ο(1) with respect to the maximum degree of monomial basis functions.
After we run test cases for each degree 100 times, the generated plot shows that the actual running time is invariant to the increase of the degree, which is consistent with the asymptotical analysis.

![cs_680_a3_y494sun](https://user-images.githubusercontent.com/29167705/63809410-43fef780-c8f0-11e9-9780-e17fbd77074b.jpg)

2) Bayesian Linear Regression: Theoretically, for Bayesian generalized linear regression, the complexity is exponentially increasing Ο(𝑎 𝑑𝑑) as the maximum degree of monomial basis functions goes up.
After we run test cases for each degree 100 times, the generated plot shows that the actual running time is consistent with the asymptotical analysis.

![cs_680_a3_y494sun](https://user-images.githubusercontent.com/29167705/63809450-52e5aa00-c8f0-11e9-8a37-d5dad965e665.jpg)

## Overall Comparision and Analysis:
Both kernel ridge regression and Bayesian generalized linear regression are forms of generalized linear model which aims to find the linear relationship w* that fits the data best. However, kernel ridge regression uses ordinary least squares (OLS) to obtain the single exact estimate w from a closed-form solution to the optimization problem, whereas Bayesian learning approach tries to find the posterior distribution of parameter w from the prior knowledge and the likelihood, which allows us to quantify the uncertainty of the model.

From the complexity point of view, kernel ridge regression takes advantage of the kernel technique. For higher order basis functions with a small dataset, it reduces the complexity to O(n3) which is cubic to the size of the dataset. On the other hand, Bayesian generalized linear regression is bounded by the cube of the number of basis functions, and the number of basis functions grows exponentially with respective to the maximum degree. Therefore, we will expect a much poorer time efficiency for Bayesian learning approach with a higher degree.
