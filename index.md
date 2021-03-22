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

### K-fold cross-validation 

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

```python
def scad_penalty(beta_hat, lambda_val, a_val):
    is_linear = (np.abs(beta_hat) <= lambda_val)
    is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
    is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
    linear_part = lambda_val * np.abs(beta_hat) * is_linear
    quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
    constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant
    return linear_part + quadratic_part + constant_part
    
def scad_derivative(beta_hat, lambda_val, a_val):
    return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))
```
```python
def scad_model(X,y,lam,a):
  n = X.shape[0]
  p = X.shape[1]
  def scad(beta):
    beta = beta.flatten()
    beta = beta.reshape(-1,1)
    n = len(y)
    return 1/n*np.sum((y-X.dot(beta))**2) + np.sum(scad_penalty(beta,lam,a))
  
  def dscad(beta):
    beta = beta.flatten()
    beta = beta.reshape(-1,1)
    n = len(y)
    return np.array(-2/n*np.transpose(X).dot(y-X.dot(beta))+scad_derivative(beta,lam,a)).flatten()
  b0 = np.ones((p,1))
  output = minimize(scad, b0, method='L-BFGS-B', jac=dscad,options={'gtol': 1e-8, 'maxiter': 1e7,'maxls': 25,'disp': True})
  return output.x
```

```python
def DoKFoldScad(X,y,lam,a,k):
  PE = []
  kf = KFold(n_splits=k,shuffle=True,random_state=1234)
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    X_train_scaled = scale.fit_transform(X_train)
    X_train_poly = poly.fit_transform(X_train_scaled)
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    X_test_scaled = scale.transform(X_test)
    X_test_poly = poly.fit_transform(X_test_scaled)
    y_test  = y[idxtest]
    beta_scad = scad_model(X_train_poly,y_train,lam,a)
    n = X_test_poly.shape[0]
    p = X_test_poly.shape[1]
    yhat_scad = X_test_poly.dot(beta_scad)
    PE.append(MAE(y_test,yhat_scad))
  return 1000*np.mean(PE)
```



### Square Root Lasso

**Square root lasso** is a modification of the Lasso. Belloni et al.  proposed a pivotal method for estimating high-dimensional sparse linear regression models. For this model, we do not need to know the standard deviation of the noise. 

The cost function is:

![\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}+\alpha\sum_{i=1}^{p}|\beta_i|
](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Csqrt%7B%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%28y_i-%5Chat%7By%7D_i%29%5E2%7D%2B%5Calpha%5Csum_%7Bi%3D1%7D%5E%7Bp%7D%7C%5Cbeta_i%7C%0A)

and is implemented as follows:

```python
def sqrtlasso_model(X,y,alpha):
  n = X.shape[0]
  p = X.shape[1]
  
  def sqrtlasso(beta):
    beta = beta.flatten()
    beta = beta.reshape(-1,1)
    n = len(y)
    return np.sqrt(1/n*np.sum((y-X.dot(beta))**2)) + alpha*np.sum(np.abs(beta))
  
  def dsqrtlasso(beta):
    beta = beta.flatten()
    beta = beta.reshape(-1,1)
    n = len(y)
    return np.array((-1/np.sqrt(n))*np.transpose(X).dot(y-X.dot(beta))/np.sqrt(np.sum((y-X.dot(beta))**2))+alpha*np.sign(beta)).flatten()
  b0 = np.ones((p,1))
  output = minimize(sqrtlasso, b0, method='L-BFGS-B', jac=dsqrtlasso,options={'gtol': 1e-8, 'maxiter': 1e8,'maxls': 25,'disp': True})
  return output.x
```

```python
def DoKFoldSqrt(X,y,a,k,d):
  PE = []
  scale = StandardScaler()
  poly = PolynomialFeatures(degree=d)
  kf = KFold(n_splits=k,shuffle=True,random_state=1234)
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    X_train_scaled = scale.fit_transform(X_train)
    X_train_poly = poly.fit_transform(X_train_scaled)
    y_train = y[idxtrain]
    X_test  = X[idxtest,:]
    X_test_scaled = scale.transform(X_test)
    X_test_poly = poly.fit_transform(X_test_scaled)
    y_test  = y[idxtest]
    beta_sqrt = sqrtlasso_model(X_train_poly,y_train,a)
    n = X_test_poly.shape[0]
    p = X_test_poly.shape[1]
    # we add an extra columns of 1 for the intercept
    #X1_test = np.c_[np.ones((n,1)),X_test]
    yhat_sqrt = X_test_poly.dot(beta_sqrt)
    PE.append(MAE(y_test,yhat_sqrt))
  return 1000*np.mean(PE)
```


# Stepwise Selection

Stepwise selection is a combination of the forward and backward variable selection techniques and was originally developed as a feature selection technique for linear regression models. The forward stepwise regression approach uses a sequence of steps to allow features to be added or dropped one at a time. The add and drop criteria is commonly based on a p-value threshold. Typically, a p-value must be less than 0.15 for a feature to enter the model and must be greater than 0.15 for a feature to leave the model. The function is defined below. 

```python
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details """
    
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included
```
The output is a list of the indices for the columns (features) that are added by the function. The added variables are the new set of variables that will be used for regression. Both sets of features are used for linear regression, and based on the mean absolute error, we found that the model performed significantly better on the new set of variables. 


# Results
With manually inputed alpha values, we can see that the performance of our regularization techniques and the square root lasso model are quite competitive. Although the mean absolute error for the linear model and stepwise selection is higher than our regularization models, the stepwise variable selection method improved our normal linear model's mean absolute error. All of these models have been cross-validated by k-fold validation.

| Model                          | Alpha     | Validated MAE       
|--------------------------------|-----------|--------------------|
| Linear Model                   |           | $4027.26          |
| Stepwise Selection             |           | $3509.20          |
| Ridge                          |        20 | $2187.94           |                     
| Lasso                          |      0.05 | $2210.72           |    
| Elastic Net                    |      0.05 | $2170.19          |
| SCAD                           |      0.15 | $2638.63          |
| Square Root Lasso              |      0.01 | $2138.62          |


Now, let's see if tuning the hyperparameters improve our results even further. 

### GridSearchCV

Grid Search is an effective method for adjusting the parameters in supervised learning and improve the generalization performance of a model. With Grid Search, we try all possible combinations of the parameters of interest and find the best ones.

| GridSearchCV                   | Alpha     | Validated MAE      |
|--------------------------------|-----------|--------------------|
| Ridge                          |     6.22  | $2855.70           |                     
| Lasso                          |      0.03 | $2948.42         |    
| Elastic Net                    |      0.15 | $3820.86          |   




# Kernel Weighted Regressions

In kernel regression, each of the kernels use their functions to determine the weights of our data points for our locally weighted regression. Locally weighted regresssions are non-parametric regression methods that combine multiple regression models in a k-nearest-neighbor-based meta-model. They are used to fit simple models to localized subsets of the data to build up a function that describes the variation in the data. Weights applied to each point help identify regions that contribute more heavily to the model, and the different kernels apply different weights to each point. Kernel weighted regressions work well for data that does not show linear qualities. The weights are typically obtained by applying a distance-based kernel function to each of the samples. 



# Random Forest and XGBoost 

Random Forest is a classification model that consists of multiple, independent decision trees. XGBoost is also a decision-tree-based algorithm that uses an advanced implementation of gradient boosting and regularization framework for speed and performance. It can best be used to solve structured data such as regression, classification, ranking, and user-defined prediction problems. XGBoost focuses on minimizing the errors to turn weak learners into strong learners and "boost" performance.

```python
kf = KFold(n_splits=10, shuffle=True, random_state=1234)
rf = RandomForestRegressor(n_estimators=1000,max_depth=3)
mae_rf = []

for idxtrain, idxtest in kf.split(X):
  X_train = X[idxtrain,:]
  y_train = y[idxtrain]
  X_test  = X[idxtest,:]
  y_test  = y[idxtest]
  rf.fit(X_train,y_train.ravel())
  yhat_rf = rf.predict(X_test)
  mae_rf.append(MAE(y_test, yhat_rf))
print("Validated MAE RF = ${:,.2f}".format(1000*np.mean(mae_rf)))
```

```python
model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
mae_xgb = []

for idxtrain, idxtest in kf.split(X):
  X_train = X[idxtrain,:]
  y_train = y[idxtrain]
  X_test  = X[idxtest,:]
  y_test  = y[idxtest]
  model_xgb.fit(X_train,y_train)
  yhat_xgb = model_xgb.predict(X_test)
  mae_xgb.append(mean_absolute_error(y_test, yhat_xgb))
print("Validated MAE XGBoost Regression = ${:,.2f}".format(1000*np.mean(mae_xgb)))
```
# Neural Network

Neural networks are a set of algorithms, modeled after the human brain, that are designed to recognize recognize hidden patterns and correlations in raw data, cluster and classify it, and – over time – continuously learn and improve. Machine learning algorithms that use neural networks generally do not need to be programmed with specific rules that define what to expect from the input. The neural net learning algorithm instead learns from processing many labeled examples that are supplied during training and using this answer key to learn what characteristics of the input are needed to construct the correct output estimates. 


```python
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import r2_score
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
```
```python
model = Sequential()
model.add(Dense(128, activation="relu", input_dim=11))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))
# Since the regression is performed, a Dense layer containing a single neuron with a linear activation function.
# Typically ReLu-based activation are used but since it is performed regression, it is needed a linear activation.
model.add(Dense(1, activation="linear"))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-3, decay=1e-3 / 200))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=800)
history = model.fit(dat_train[:,:-1], dat_train[:,11], validation_split=0.3, epochs=1000, batch_size=100, verbose=0, callbacks=[es])
```
```python
yhat_nn = model.predict(dat_test[:,:-1])
mae_nn = mean_absolute_error(dat_test[:,-1], yhat_nn)
print("MAE Neural Network = ${:,.2f}".format(1000*mae_nn))
```

# Results

| Model                          |  Validated MAE     |  
|--------------------------------|--------------------|
| Kernel Regressions             | $2854.57           |
| Random Forest Regression       | $2877.17           |
| XGBoost                        |  $2313.58          |                     
| Neural Network                 |  $2476.62          |    


