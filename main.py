import pandas as pd
import matplotlib as plt
import numpy as np
import sklearn as sl

'''importing train dataset and test dataset'''
train_df = pd.read_csv(r'C:\Users\Utente\Desktop\TitanicKaggle\csv\train.csv')
test_df=pd.read_csv(r'C:\Users\Utente\Desktop\TitanicKaggle\csv\test.csv')
train_df.info()