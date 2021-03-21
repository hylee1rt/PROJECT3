The goal of this project is to compile and apply different models and methods for nonlinear regression on the Boston Housing Prices dataset. This analysis will cover regularization techniques on polynomial features, other nonlinear models, hyperparameter tuning and model validation. 


## Boston Housing Data

Let's First look at a heat map of the correlationa in the data. The colors of the heat map will show us the strength of correlations between features and help us understand the data better. 

![download](https://user-images.githubusercontent.com/66886936/110899861-d2b74780-82cf-11eb-97b6-159cbda0f83e.png)

# Regularized Regression on Polynomial Features

To apply polynomial features to our linear models with different regularization algorithms, we will initialize standard scalar to scale our data and polynomial features to transform our data.

```python
scale = StandardScaler()
poly = PolynomialFeatures(degree=3)
```

Then we define a general function *DoKFold_SK* that will take the model as a parameter (which will be one of the regularization models) and return the average MAE of each of the k-folds. The k-fold cross-validation a resampling procedure used to evaluate machine learning models on a limited data sample. It splits the data into *k* equal groups, and uses each group once as the test data while the rest is used as training data. By averaging many 'trials,' it gives us a more accurate estimate of our mean absolute error.

```python
def DoKFold_SK(X,y,model,k):
  PE = []
  kf = KFold(n_splits=k,shuffle=True,random_state=1234)
  pipe = Pipeline([('scale',scale),('polynomial features',poly),('model',model)])
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    y_test  = y[idxtest]
    pipe.fit(X_train,y_train)
    yhat_test = pipe.predict(X_test)
    PE.append(MAE(y_test,yhat_test))
  return 1000*np.mean(PE)
```

### L2 (Ridge) Regularization - Tikhonov 1940's

The first regularization technique is **Ridge regression**.  In addition to minimizing the sum of squared errors, Ridge regression also penalizes a model for having more parameters and/or larger parameters.  This is accomplished by modifying the cost function:

![\sum_{i=1}^N(y_i-\hat{y}_i)^2+\alpha\sum_{i=1}^p\beta_i^2](https://render.githubusercontent.com/render/math?math=%5CLARGE+%5Cdisplaystyle+%5Csum_%7Bi%3D1%7D%5EN%28y_i-%5Chat%7By%7D_i%29%5E2%2B%5Calpha%5Csum_%7Bi%3D1%7D%5Ep%5Cbeta_i%5E2)


The new term in this equation, the one following *alpha*, is called the **regularization term**.  In words, it just says to square all of the model coefficients, add them together, multiply that sum by some number *a* and then add that to the cost function.  As a result, the model will be simultaneously trying to minimize both the sum of the squared errors as well as the number/magnitude of the model's parameters. The alpha, or *a* in the regularization term is referred to as a **hyperparameter** - which is a number that is not determined during the fitting/training procedure. Different values of *a* can be manually specified or found via various methods of hyperparameter tuning. 





