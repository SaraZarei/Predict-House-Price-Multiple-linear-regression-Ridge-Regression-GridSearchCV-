# 10)	Predict house price (Multiple linear regression, Ridge Regression, GridSearchCV)
## House Sales in King County, USA
This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.
In this notebook after cleaning and preprocessing and exploratory on data I developed 3 models and got R-squared for all of them.At first  with **multiple  linear regression** on all data without split data and  then I defined a **polynomial function on multiple features** and find best order for polynomial function.Then I developed a **Ridge regression** with define a  hyperparameter for control magnetitude coefficients in function and  used **GrisSearchCV** method for finding best hyperparameter for RidgeRegression
## Exploratory Data
This plot shows the correlation between price and sqrt_living is positive and strong and this feature can affect on predicting price
![1](https://user-images.githubusercontent.com/56628918/114891544-463b0000-9e0c-11eb-919c-949e0955ebaf.png)
![2](https://user-images.githubusercontent.com/56628918/114891549-46d39680-9e0c-11eb-8049-e006c07bf6b5.png)
## Model Development
Fit a linear regression model to predict the 'price' using the list of features

![3](https://user-images.githubusercontent.com/56628918/114891552-46d39680-9e0c-11eb-92aa-ab1106e45aaa.png)

Fit a linear regression with polynomial function

![4](https://user-images.githubusercontent.com/56628918/114891554-476c2d00-9e0c-11eb-8d3c-0381d7da3c55.png)

we see the best R-squared for model is belong to order=5 that it is 0.72

![5](https://user-images.githubusercontent.com/56628918/114891557-476c2d00-9e0c-11eb-9602-351d963cc1d1.png)
## Evaluate Model
For evaluating model we divided data to two part:trainset and testset ,at first train the model with polynomial function on trainset and then test the model on testset.
This negative score means the model can be involve overfitting, we plot R-squared in different orders
According this R-squared from different order of polynomial function we see that the best score is for order=2

![6](https://user-images.githubusercontent.com/56628918/114891558-476c2d00-9e0c-11eb-9272-737cafeca65a.png)

## Conclusion
After clean and preprocessing data,with exploratory on data I found out features 'sqft_living', 'grade','sqft_above' ,'sqft_living15','bathrooms','bedrooms'and 'view' that have more correlate with price and I select them for model developing. In model developing at first,I tried to develop a multiple regression on all data and got R-squared equal 0.58 for model.Then I tried to develop a model with polynomial function with different order from 1 to 8 and it was resulted best R-Squared equal 0.72 with order 5.Then I tested this model on unseen data and got an overfitting with order 2, therefore best score for polynomial function was 0.66 with order 2. then tried my model with cross validation with 3 fold and it was result score equal 0.57. then I developed a model with Ridge regression with alpha equal 0.1 and got score 0.66 and I used GridSearchCV for determination best alpha for Ridge regression and it was result score 0.62 with alpha 100000.
