"""
Predicting Employee Productivity
"""

#1. Import the necessary packages and dataset
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.callbacks import EarlyStopping, TensorBoard
import datetime, os
import pandas as pd

data_path = r"C:\Users\muhdi\Documents\Deep Learning\Employee-Productivity-Project\data\garments_worker_productivity.csv"
df = pd.read_csv(data_path)

#%%
#2. DATA PREPARATION!
print(df.isna().sum())

#%%
#checking skewness
print(df.wip.skew())
#data is positively skewed

wip_median_value = df['wip'].median()

#Fill NA values with mean value of the column
gworker = df.copy()
gworker['wip'].fillna(value=wip_median_value, inplace=True)
print(gworker.wip.median())

print(gworker.isna().sum())

#%%
#checking for anomalies
#number of workers in each team should be a whole number
print(gworker['no_of_workers'].unique())

#there are some values with decimal points. 
#to deal with the anomalies above, the figures will be truncated

gworker['no_of_workers'] = gworker['no_of_workers'].apply(lambda x: int(x))
#checking that the figures were truncated

print(gworker['no_of_workers'].unique())

#%%
#Quarter is supposed to be 4, but the df has 5. Checking what it is
head = gworker[gworker['quarter'] == 'Quarter5']
print(head)

#Seems like they are dates after the 28th of each month. Hence we replace Quarter 5 to Quarter 4
gworker['quarter'] = gworker.quarter.str.replace('Quarter5', 'Quarter4')
gworker.quarter.unique()

#%%
#dropping date as there is insufficient data on different months, as we have days for that
gworker = gworker.drop(['date'], axis=1)

#%%
#replacing spelling error in department
print(gworker['department'])
#Change 'sweing' to 'sewing'
gworker['department'] = gworker['department'].replace(['sweing'],['sewing'])
#Change 'finishing ' to 'finishing'
gworker['department'] = gworker['department'].replace(['finishing '], ['finishing'])
print(gworker['department'])

#idle time and idle men might contribute to the prediction, as they directly affects productivity
#hence we will not be dropping these columns

#%%
#Label encode the Federation column
encoder1 = LabelEncoder()
encoder2 = LabelEncoder()
encoder3 = LabelEncoder()
encoder1.fit(gworker['quarter'])
gworker['quarter'] = encoder1.transform(gworker['quarter'])
encoder2.fit(gworker['day'])
gworker['day'] = encoder2.transform(gworker['day'])
encoder2.fit(gworker['department'])
gworker['department'] = encoder2.transform(gworker['department'])

#%%
#5. Split data into features and labels
features = gworker.copy()
labels = features.pop('actual_productivity')

#%%
#6. Do a train-test split
SEED=12345
x_train, x_iter, y_train, y_iter = train_test_split(features,labels,test_size=0.4,random_state=SEED)
x_val, x_test, y_val, y_test = train_test_split(x_iter,y_iter,test_size=0.5,random_state=SEED)

#7. Perform data normalization
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# DATA PREPARATION ends at this line

#%%
#8. Build NN model
no_input= x_train.shape[-1]

# Use functional API
model = keras.Sequential()
model.add(layers.InputLayer(input_shape=no_input))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))

#view your model
model.summary()

#to view a representation of your model structure
#tf.keras.utils.plot_model(model, show_shapes=True)

#%%
#9. Compile and train the model
base_log_path = r"C:\Users\muhdi\Documents\Deep Learning\Employee-Productivity-Project\logs"
log_path= os.path.join(base_log_path, datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '__Employee_Productivity_Project')
es = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
#use TensorBoard to view your training graph
tb = TensorBoard(log_dir=log_path)
BATCH_SIZE = 64

model.compile(optimizer='adam', loss='mse', metrics=['mae','mse'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=BATCH_SIZE, epochs=100, callbacks=[tb,es])

#%%
#to view graph of train against validation, run in the code below without the # key in prompt
#tensorboard --logdir "C:\Users\muhdi\Documents\Deep Learning\Employee-Productivity-Project\logs"

#%%
#10. Evaluate with test data for wild testing
test_result = model.evaluate(x_test,y_test,batch_size=BATCH_SIZE)
print(f"Test loss = {test_result[0]}")
print(f"Test MAE = {test_result[1]}")
print(f"Test MSE = {test_result[2]}")

#%%
import matplotlib.pyplot as plt
#11. Plot a prediction graph vs label on the test data
predictions = np.squeeze(model.predict(x_test))
labels = np.squeeze(y_test)
plt.plot(predictions,labels,".")
plt.xlabel("Predictions")
plt.ylabel("Labels")
plt.title("Graph of Predictions vs Labels with Test Data")
save_path = r"C:\Users\muhdi\Documents\Deep Learning\Employee-Productivity-Project\img"
plt.savefig(os.path.join(save_path,"result.png"),bbox_inches='tight')
plt.show()