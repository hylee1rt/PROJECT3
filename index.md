The goal of this project is to compile and apply different models and methods for nonlinear regression on the Boston Housing Prices dataset. This analysis will cover regularization techniques on polynomial features, other nonlinear models, hyperparameter tuning and model validation. 


### Boston Housing Data

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

### L1 (Lasso) Regularization - Tibshirani 1993

For **Lasso regression**, in stead of squaring the coefficients, we are taking their **absolute value**. Unlike Ridge, that is more reliable for predictive power, Lasso can set some of the model coefficients to zero, effectively removing variables from the model.

The cost function is defined as:

![\sum_{i=1}^N(y_i-\hat{y}_i)^2+\alpha\sum_{i=1}^p\lvert\beta_i\rvert](https://render.githubusercontent.com/render/math?math=%5CLARGE+%5Cdisplaystyle+%5Csum_%7Bi%3D1%7D%5EN%28y_i-%5Chat%7By%7D_i%29%5E2%2B%5Calpha%5Csum_%7Bi%3D1%7D%5Ep%5Clvert%5Cbeta_i%5Crvert)


The plots below to visualizes how the coefficients of the ridge and lasso regression models change as we adjust *a*. We see that the coefficients from lasso regression do not asymptotically reach near zero like ridge regression, but rather hit zero and disappear.



| Ridge                          | Lasso       | 
|--------------------------------|-----------|
| <img src="https://user-images.githubusercontent.com/66886936/110906664-42323480-82da-11eb-8b51-8e6ad47eb43d.png" width="400" height="300"  /> | <img src="https://user-images.githubusercontent.com/66886936/110906859-93dabf00-82da-11eb-8e2f-de64cad57d75.png" width="400" height="300"  /> |                    



### Elastic Net

**Elastic net** is a penalized linear regression model that includes both the *L_1* and *L_2* penalties during training. The cost function is: 

![\hat{\beta} = argmin_\beta \left\Vert  y-X\beta \right\Vert ^2 + \lambda_2\left\Vert  \beta \right\Vert ^2 + \lambda_1\left\Vert  \beta\right\Vert_1
](https://render.githubusercontent.com/render/math?math=%5Clarge+%5Cdisplaystyle+%5Chat%7B%5Cbeta%7D+%3D+argmin_%5Cbeta+%5Cleft%5CVert++y-X%5Cbeta+%5Cright%5CVert+%5E2+%2B+%5Clambda_2%5Cleft%5CVert++%5Cbeta+%5Cright%5CVert+%5E2+%2B+%5Clambda_1%5Cleft%5CVert++%5Cbeta%5Cright%5CVert_1%0A)


for *a* in *[0,1]*. 

Using this model, you can find the combination that fits both multiple correlations in the data and the sparsity pattern. 


### SCAD - Fan & Li 2001

The **smoothly clipped absolute deviation (SCAD) penalty** was designed to encourage sparse solutions to the least squares problem, while also allowing for large values of *Beta*.

The cost function looks like:

![download (7)](https://user-images.githubusercontent.com/66886936/110971819-55202580-8329-11eb-94f8-10259f542246.png)

the SCAD penalty is often defined primarily by its first derivative 
p'(β), rather than p(β). Its derivative is


![p'_\lambda(\beta) = \lambda \left\{ I(\beta \leq \lambda) + \frac{(a\lambda - \beta)_+}{(a - 1) \lambda} I(\beta > \lambda) \right\}
](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+p%27_%5Clambda%28%5Cbeta%29+%3D+%5Clambda+%5Cleft%5C%7B+I%28%5Cbeta+%5Cleq+%5Clambda%29+%2B+%5Cfrac%7B%28a%5Clambda+-+%5Cbeta%29_%2B%7D%7B%28a+-+1%29+%5Clambda%7D+I%28%5Cbeta+%3E+%5Clambda%29+%5Cright%5C%7D%0A)

where *a* is a tunable parameter that controls how quickly the penalty drops off for large values of β.

The penalty is defined as:

![\begin{cases} \lambda & \text{if } |\beta| \leq \lambda \\ \frac{(a\lambda - \beta)}{(a - 1) } & \text{if } \lambda < |\beta| \leq a \lambda \\ 0 & \text{if } |\beta| > a \lambda \\ \end{cases}
](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cbegin%7Bcases%7D+%5Clambda+%26+%5Ctext%7Bif+%7D+%7C%5Cbeta%7C+%5Cleq+%5Clambda+%5C%5C+%5Cfrac%7B%28a%5Clambda+-+%5Cbeta%29%7D%7B%28a+-+1%29+%7D+%26+%5Ctext%7Bif+%7D+%5Clambda+%3C+%7C%5Cbeta%7C+%5Cleq+a+%5Clambda+%5C%5C+0+%26+%5Ctext%7Bif+%7D+%7C%5Cbeta%7C+%3E+a+%5Clambda+%5C%5C+%5Cend%7Bcases%7D%0A)


### Square Root Lasso

**Square root lasso** is a modification of the Lasso. Belloni et al.  proposed a pivotal method for estimating high-dimensional sparse linear regression models. For this model, we do not need to know the standard deviation of the noise. 

The cost function is:

![\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}+\alpha\sum_{i=1}^{p}|\beta_i|
](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Csqrt%7B%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28y_i-%5Chat%7By%7D_i%29%5E2%7D%2B%5Calpha%5Csum_%7Bi%3D1%7D%5E%7Bp%7D%7C%5Cbeta_i%7C%0A)
