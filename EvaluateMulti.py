from MultiRegressorModule import MultiRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.mixture import GaussianMixture

#########################################################################
# 1. Read training data.
#########################################################################
data = pd.read_csv('train.csv')
data = data.drop('Id',axis=1)
data_prep = data.copy() #Data to preprocess

#########################################################################
# 2. Drop columns with too many NaN values.
#########################################################################

dropped_columns = []

dropped_columns = ['Fence','Alley','MiscFeature','PoolQC']
data_prep = data_prep.drop(dropped_columns, axis=1)

'''
#If we want automatically delete garbage columns
for x in data_prep.columns:
    if( len(data_prep.loc[ data[x].isnull() , :]) > (data_prep.shape[0] / 1.3)  ):
        print('Column dropped: ', x, ' ',len(data_prep.loc[ data_prep[x].isnull() , :]))
        data_prep = data_prep.drop(x, axis=1)
        dropped_columns.append(x)
'''

#########################################################################
# 3. Map string values to int with dictionary.
#########################################################################
d_full = {} #Dictionary for mapping dictionaries

for x in data_prep.columns:
    if(data_prep[x].dtype == np.object):    #If column has object type it has to be mapped
        d = {}                              #Dictionary for mapping one column
        for y in range(len(pd.unique(data_prep[x]))):
            d[ pd.unique(data_prep[x])[y] ] = y
        data_prep[x] = data_prep[x].map(d)
        d_full[x] = d
        
       
#########################################################################        
# 4. Preprocessing data for training and remove outliers
#########################################################################        
data_prep = data_prep.fillna( data_prep.mean() ) #Filling NaN values with mean value of each column 
data_prep.reset_index(inplace=True, drop=True)

X = data_prep.drop('SalePrice', axis=1)
target = data_prep['SalePrice']

gm = GaussianMixture( n_components=len(X.columns), n_init=10 )

gm.fit(X)

densities = gm.score_samples(X)

density_threshold = np.percentile(densities, 4)

outliers = X[densities < density_threshold]

X = pd.DataFrame(X).drop(outliers.index) #Drop outliers
target = pd.DataFrame(target).drop(outliers.index) #Drop outliers


scaler = MinMaxScaler() #With MinMaxScaler we scale values in each column into 0-1
scaler.fit(X)

X = scaler.transform(X)


X_train, X_val, y_train, y_val = train_test_split(X, target, test_size=0.15)  #Splitting train dataset in 85% train and 15% for evaluating regressors


print("Deleted outliers: ", len(outliers))


#########################################################################
# 5. MultiRegressor training
#########################################################################

multi_clf = MultiRegressor(X_train, X_val, y_train, y_val,n_repetition = 10)
multi_clf.compile_fit() #Compiling and fitting with train data
multi_clf.evaluate() #Evaluate regressors with validation data


#########################################################################
# 6. Preprocessing data for predicting.
#########################################################################

data_test = pd.read_csv('test.csv')


data_test = data_test.drop(dropped_columns, axis=1) # Drop before dropped columns
col_id = data_test['Id'].copy()
data_test = data_test.drop('Id', axis=1)

for x in data_test.columns:
    if(data_test[x].dtype == np.object):
        data_test[x] = data_test[x].map(d_full[x])
        
data_test = data_test.fillna( data_test.mean() ) 

data_test.reset_index(inplace=True, drop=True)

X = data_test.copy()

scaler = MinMaxScaler()
scaler.fit(X)
X_std = scaler.transform(X)
X_std = pd.DataFrame(X_std, columns = X.columns)


multi_clf.predict_save(X_std, col_id) #Predict test data and save 


