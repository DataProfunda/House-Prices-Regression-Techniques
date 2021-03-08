# House Prices Regression Techniques
![](https://storage.googleapis.com/kaggle-competitions/kaggle/5407/media/housesbanner.png)

Competition Description:

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. 
But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

Acknowledgments:

The Ames Housing dataset was compiled by Dean De Cock for use in data science education.
It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset.

# Let's start!


This project contains two Python files: <br>
EvaluateMulti.py - basic preprocessing, fill Nan values, detect and remove outliers, train/test split, MultiRegressor compile and fit <br>
MultiRegressorModule.py - MultiRegressor class contains RandomForestRegressor, ExtraTreesRegressor, XGBRegressor, VotingRegressor. <br>

In this project I wanted to try out  regression techniques I've learnt recently. For customizing Regressors I've created MultiRegressorModule, that contains class MultiRegressor. 
I also want to use MultiRegressorModule in future projects, so I decided to add funtions such as:

-compile_fit() - compile all of regressors and fit with train data <br>
-evaluate() - evaluate regressors on test data <br>
-save_model() -  <br>
-fit_with_test_data() - fit with test data. I do not recommend using it in this project <br>
-load_models() <br>
-predict_save() <br>

For finding best hyperparameters I've used GridSearchCV and RandomizedSearchCV( in xgboost for optimization) <br>

Same regressors with same hyperparameters can still produce different results. Therefore we can perform repetition for finding best one 
example: MultiRegressor(n_repetition=10)


# 1. Read training data.
![s1](https://user-images.githubusercontent.com/69935274/101414364-3a7f8000-38e6-11eb-8402-7837483ec16f.png)
# 2. Drop columns with too many NaN values.
Columns such as Fence, Alley, MiscFeature, PoolQC have over 70% NaN values.
We should get rid of them.

![nan](https://user-images.githubusercontent.com/69935274/110354257-43443700-8038-11eb-8507-c35ad54f5ae3.png)

We won't lose much information, because they are not so important, depending on correlation.
![corra](https://user-images.githubusercontent.com/69935274/110363069-6247c680-8042-11eb-8bfa-4d16186764e2.png)


# 3. Map string values to int with dictionary.
d_full dictionary contains dictionary for each column that has to be mapped.
![s3](https://user-images.githubusercontent.com/69935274/101414465-6ac71e80-38e6-11eb-8dcf-e04f7cf1ab1c.png)
# 4. Preprocessing data for training and remove outliers
NaN values are filled with mean value of each column.
With MinMaxScaler we normalize values in columns.
In other words, we scale values to be in the range between 0 and 1.
For outliers detection I've used GaussianMixture. It deleted 59 rows. 
Train test split size is set to 0.15.

# 5. MultiRegressor training.
To use MultiRegressor first we want to make object.
N_repetition we set to 10.
We want to train all of the Regressors so we don't set up regressors argument.
Because we use GridSearchCv after code run I suggest to make some coffee, because it can take several minutes to end.


# 6. Preprocessing data for testing.
Variable dropped_columns was defined to drop same columns as in the training data 
![s6](https://user-images.githubusercontent.com/69935274/101414514-82060c00-38e6-11eb-8095-b75eb87bae82.png)
# 7. Predicting house price for unseen rows.
To predict and save submission I've used predict_save() funtion in MultiRegressors

# Conclusions
There are a lot of things that I could do better. Order of the categorical data could retain some more information, but with auto-mapping we sacrifise it for the time save. Other notebooks that I've read, spend more time on feature engineering. They usually produce better results.  
Although we splitted data, when we set big n_repetition overfitting occur. I think KFold would produce better results.    
