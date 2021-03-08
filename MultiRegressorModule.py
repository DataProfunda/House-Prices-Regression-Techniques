#Multi clf + grid search cv


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sklearn

from sklearn.model_selection import train_test_split

import time

from sklearn.model_selection import RandomizedSearchCV


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

import pickle


class MultiRegressor ():
    
    #MultiRegressor contains RandomForestRegressor, ExtraTreesRegressor, XGBRegressor, VotingRegressor 
    #and funtions to perform compiling and fitting for the best outcome.
    #n_repetition (int) - how many times we want to perform compiling and fitting on regressors to find the best one
    #Too many repetition will result in overfitting!
    #regressors (string) - as default it contains ('extra_reg', 'rnd_reg','voting_reg','xgb_reg'), 
    #it is used if you want to use for example only ExtraTreesRegressor
    #X_train,X_test, y_train, y_test are DataFrames for performing fitting and testing regressors
    
    def __init__(self,X_train,X_test, y_train, y_test, n_repetition=20, regressors = ('extra_reg', 'rnd_reg','voting_reg','xgb_reg')):
       
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_repetition = n_repetition
        self.regressors = regressors
        
    
    def compile_fit(self):
        
        #This function contain funtions for compiling and fitting regressors
        
        self.reg_models = [] #List of regressors, it is used for further fitting VotingRegressor
        
        if 'xgb_reg' in self.regressors:                    
            self.xgb_compile_fit()

        if 'extra_reg' in self.regressors:       
            self.extra_compile_fit()
   
        if 'rnd_reg' in self.regressors:   
            self.rnd_compile_fit()    
            
        if 'voting_reg' in self.regressors:
            self.voting_compile_fit()

                
    def evaluate(self):  
        
        #This function perform evaluation on test data
        #Mse is divided by 1 mln to show more clearly mse
        
        if 'extra_reg' in self.regressors:
            y_pred = self.extra_reg.predict(self.X_test)
            print("Extra Trees Reg", mean_squared_error(self.y_test, y_pred, squared=True) / 1000000)
            
        if 'rnd_reg' in self.regressors:
            y_pred = self.rnd_reg.predict(self.X_test)
            print("Random Forest Reg", mean_squared_error(self.y_test, y_pred,squared=True) / 1000000)
        
        if 'xgb_reg' in self.regressors:
            y_pred = self.xgb_reg.predict(self.X_test)
            print("XBGboost Reg", mean_squared_error(self.y_test, y_pred, squared=True) / 1000000)
        
        if 'voting_reg' in self.regressors:
            y_pred = self.voting_reg.predict(self.X_test)
            print("Voting Reg", mean_squared_error(self.y_test, y_pred, squared=True) / 1000000)

        
    def save_model(self):
        
        #This function save trained models
        #It is called once, and ask for approval to save models
        
        if 'extra_reg' in self.regressors:
            
            print("Do you want to save Extra Regressor? y/n")
            
            want_save = input()
            
            if want_save=='y':
                filename = 'extra_reg.sav'
                pickle.dump(self.extra_clf, open(filename, 'wb'))   
                print('Model saved!')
            else:
                print('Model not saved')
            
        if 'rnd_reg' in self.regressors:
            
            print("Do you want to save Random Regressor? y/n")
            
            want_save = input()
            
            if want_save=='y':
                filename = 'rnd_reg.sav'
                pickle.dump(self.rnd_clf, open(filename, 'wb'))   
                print('Model saved!')
            else:
                print('Model not saved')
     
        if 'xgb_reg' in self.regressors:
            
            print("Do you want to save XGB Regressor? y/n")
            
            want_save = input()
            
            if want_save=='y':
                filename = 'xgb_reg.sav'
                pickle.dump(self.voting_clf, open(filename, 'wb'))   
                print('Model saved!')
            else:
                print('Model not saved')
                
        if 'voting_reg' in self.regressors:
            
            print("Do you want to save Voting Regressor? y/n")
            
            want_save = input()
            
            if want_save=='y':
                filename = 'voting_reg.sav'
                pickle.dump(self.voting_clf, open(filename, 'wb'))   
                print('Model saved!')
            else:
                print('Model not saved')
            

    
    def fit_with_test_data(self):
        
        #This funtion train choosen regressors with data. 
        #It also loads models if we want to perform predicting on saved regressors
        
        if 'extra_reg' in self.regressors:

            if self.extra_reg==None:   
                self.extra_reg = pickle.load(open('extra_reg.sav', 'rb'))
                
            self.extra_reg = self.extra_reg.fit(self.X_test,self.y_test.values.ravel())
            
        if 'rnd_reg' in self.regressors:
            
            if self.rnd_reg==None:   
                self.rnd_reg = pickle.load(open('rnd_reg.sav', 'rb'))
                
            self.rnd_reg.fit(self.X_test,self.y_test.values.ravel())
            
            
        if 'xgb_reg' in self.regressors:
            
            if self.xgb_reg==None:   
                self.xgb_reg = pickle.load(open('xgb_reg.sav', 'rb'))
            
            self.xgb_reg.fit(self.X_test,self.y_test.values.ravel())

        if 'voting_reg' in self.regressors:
            
            #For voting regressor we have to load/compile other regressors
            
            if self.extra_reg==None:   
                self.extra_reg = pickle.load(open('extra_reg.sav', 'rb'))
                
            if self.rnd_reg==None:   
                self.extra_reg = pickle.load(open('extra_reg.sav', 'rb'))
                
            if self.xgb_reg==None:   
                self.extra_reg = pickle.load(open('extra_reg.sav', 'rb'))
            
            self.rnd_reg.fit(self.X_test,self.y_test.values.ravel())
            self.extra_reg.fit(self.X_test,self.y_test.values.ravel())
            self.xgb_reg.fit(self.X_test,self.y_test.values.ravel())
            
            self.voting_reg.fit(self.X_test,self.y_test.values.ravel())
            y_pred = self.voting_reg.predict(self.X_train)
            print("Voting Reg", mean_squared_error(self.y_train, y_pred, squared=True) / 1000000)
            


    def load_models(self):
        
        #This funtion loads saved regressors
        
        if 'extra_reg' in self.regressors:
            self.extra_reg = pickle.load(open('extra_reg.sav', 'rb'))
            
        if 'rnd_reg' in self.regressors:       
            self.rnd_reg = pickle.load(open('rnd_reg.sav', 'rb'))
            
        if 'xgb_reg' in self.regressors:
            
            self.xgb_reg = pickle.load(open('xgb_reg.sav', 'rb'))

        if 'voting_reg' in self.regressors:
            
            #If we want to load VotingRegressor we have to load other regressors as well
            self.xgb_reg = pickle.load(open('xgb_reg.sav', 'rb'))
            self.extra_reg = pickle.load(open('extra_reg.sav', 'rb'))
            self.rnd_reg = pickle.load(open('rnd_reg.sav', 'rb'))
            self.voting_reg = pickle.load( open('voting_reg.sav', 'rb')  )

    def predict_save(self, data_to_predict, col_id, filename = 'submission.csv'):
                   
        #This funtion does prediction and save it as a 'submission.csv' file
        #Because voting regressor usually shows best results we only implement this function for only this regressor
        
        if 'voting_reg' in self.regressors:

            submisson = pd.DataFrame( None, columns = ['Id','SalePrice'] )
            submisson['Id'] = col_id
            submisson['SalePrice'] = np.arange(len(data_to_predict))
                
            #print(data_to_predict.columns)
            submisson['SalePrice'] = self.voting_reg.predict(data_to_predict.values).astype(int) 

            submisson.to_csv(filename,index=False)

            print("Submission saved!")

        
    def xgb_compile_fit(self):
        
        #This funtion does compiling and fitting on XGBoost regressor
        #For finding best hyperparameters we use RandomizedSearchCV
        #This model has many hyperparameters so GridSearchCV would consume a lot of time
        
        
        param_tuning = {
                        'learning_rate': [0.01,0.02, 0.05,  0.1],
                        'max_depth': [3, 5, 7, 10],
                        'min_child_weight': [1, 3, 5],
                        'subsample': [0.5, 0.7],
                        'colsample_bytree': [0.5, 0.7],
                        'n_estimators' : [100, 150, 200, 250,  500],
                        'objective': ['reg:squarederror']
                        }
                    
        self.xgb_reg = XGBRegressor()
        grid_search = RandomizedSearchCV(self.xgb_reg,
                                   param_tuning,  
                                   scoring="neg_mean_squared_error",
                                   verbose=2)
        grid_search.fit(self.X_train, self.y_train.values.ravel())
        #print(grid_search.best_params_)
        
        #After finding suited hyperparameters we do n fitting and compling to find the best one
        
        prev_mse = 0        
        i = 0
        
        while(i < self.n_repetition):
            
            if i == 0:
                #As a first tested XGBoost regressor we use best performing regressor from RandomSearchCV  
                self.xgb_reg = grid_search.best_estimator_

                y_pred = self.xgb_reg.predict(self.X_test)  
                    
                prev_mse = mean_squared_error(self.y_test, y_pred, squared=True)
                print(i + 1 , ". ", "XGBoost_clf", prev_mse / 1000000)
                    
            else:
                
                #For further testing we use new XGBRegressor with suited hyperparameters
                current_reg = XGBRegressor(n_estimators=grid_search.best_params_['n_estimators'])
                current_reg.fit(self.X_train, self.y_train.values.ravel())
                y_pred = current_reg.predict(self.X_test)  
                mse = mean_squared_error(self.y_test, y_pred, squared=True)
        
                print(i + 1, ". ", "XGBoost_clf", mse/ 1000000)
                
                if mse < prev_mse:
                    self.xgb_reg = current_reg
                    prev_mse = mse

            i = i + 1
        #At the end we add to the list best performing XGBoost regressor to furter train VotingRegressor    
        self.reg_models.append( ("xgb_reg" , self.xgb_reg) )
            
            
    def extra_compile_fit(self):
        
        #This funtion does compiling and fitting on ExtraTreesRegressor 
        #For finding best hyperparameters we use GridSearchCV
        
        param_grid = [ {"n_estimators" : [150,200,250,300,350, 400], "max_depth":[10,15,20,40,50, 60, 70 ]}]
            
        grid_search = GridSearchCV(ExtraTreesRegressor(), param_grid,  scoring="neg_mean_squared_error", verbose=2)
        grid_search.fit(self.X_train, self.y_train.values.ravel())
                
        print(grid_search.best_params_)
                                
        prev_mse = 0     
        i = 0
        
        #After finding suited hyperparameters we do n fitting and compling to find the best ExtraTreesRegressor        
        while(i < self.n_repetition):
                    
            if i == 0:
                #As a first tested ExtraTreesRegressor we use best performing regressor from GridSearchCV  
                self.extra_reg = grid_search.best_estimator_
                y_pred = self.extra_reg.predict(self.X_test)  
                prev_mse = mean_squared_error(self.y_test, y_pred, squared=True)
                        
                print(i + 1, ". ", "Extra_reg", prev_mse/ 1000000)
            else:
                #For further testing we use new ExtraTreesRegressor with suited hyperparameters
                current_reg = ExtraTreesRegressor(n_estimators=grid_search.best_params_['n_estimators'])
                current_reg.fit(self.X_train, self.y_train.values.ravel())
                y_pred = current_reg.predict(self.X_test)  
                mse = mean_squared_error(self.y_test, y_pred, squared=True)
                        
                print(i + 1, ". ", "Extra_reg", mse/ 1000000)
                        
                if mse < prev_mse:
                    self.extra_reg = current_reg
                    prev_mse = mse
        
            i = i + 1
            
        #At the end we add to the list best performing ExtraTreesRegressor to furter train VotingRegressor     
        self.reg_models.append( ("extra_reg" , self.extra_reg) )
        
        
    def rnd_compile_fit(self):
        
        #This funtion does compiling and fitting on RandomForestRegressor
        #For finding best hyperparameters we use GridSearchCV
        
        param_grid = [ {"n_estimators" : [250,300,350,400,450, 500, 550], "max_depth":[10,15,20,40,50 ]}]
            
        grid_search = GridSearchCV(RandomForestRegressor(), param_grid,  scoring="neg_mean_squared_error", verbose=2)
        grid_search.fit(self.X_train, self.y_train.values.ravel())
        
        print(grid_search.best_params_)
              
        prev_mse = 0     
        i = 0
        
        #After finding suited hyperparameters we do n fitting and compling to find the best RandomForestRegressor   
        while(i < self.n_repetition):
                
            if i == 0:
                #As a first tested RandomForestRegressor we use best performing regressor from GridSearchCV  
                self.rnd_reg = grid_search.best_estimator_
                y_pred = self.rnd_reg.predict(self.X_test)  
                prev_mse = mean_squared_error(self.y_test, y_pred, squared=True)
                    
                print(i + 1 , ". ", "Rnd_reg", prev_mse/ 1000000)
            else:
                #For further testing we use new ExtraTreesRegressor with suited hyperparameters
                current_reg = RandomForestRegressor(n_estimators=grid_search.best_params_['n_estimators'])
                current_reg.fit(self.X_train, self.y_train.values.ravel())
                y_pred = current_reg.predict(self.X_test)  
                mse = mean_squared_error(self.y_test, y_pred,  squared=True)
                    
                print(i + 1 , ". ", "Rnd_reg", mse/ 1000000)                
                if mse < prev_mse:
                    self.rnd_reg = current_reg
                    prev_mse = mse
    
            i = i + 1
            
        self.reg_models.append(  ("rnd_reg" , self.rnd_reg) )
            
    
    def voting_compile_fit(self):
        
        #This funtion does compiling and fitting on VotingRegressor
        
        prev_mse = 0     
        i = 0
        
        #We do n fitting and compling to find the best VotingRegressor  
        while(i < self.n_repetition):
                
            if i == 0:
                self.voting_reg = VotingRegressor(estimators=self.reg_models)
                self.voting_reg.fit(self.X_train, self.y_train.values.ravel())
                y_pred = self.voting_reg.predict(self.X_test)  
                prev_mse = mean_squared_error(self.y_test, y_pred)
                    
                print(i + 1 , ". ", "Voting_reg", prev_mse/ 1000000)
                    
            else:
                current_reg = VotingRegressor(estimators=self.reg_models)
                current_reg.fit(self.X_train, self.y_train.values.ravel())
                y_pred = current_reg.predict(self.X_test)  
                mse = mean_squared_error(self.y_test, y_pred)
                    
                print(i + 1, ". ", "Voting_reg", mse/ 1000000)
                    
                if mse < prev_mse:
                    self.voting_reg = current_reg
                    prev_mse = mse
    
            i = i + 1
            
        