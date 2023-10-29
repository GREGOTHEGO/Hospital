import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder 
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import PCA
from tensorflow.keras import regularizers
from pickle import dump

def data_preprocessing(df):
    
    col_to_keep = ['timeknown','cost', 'reflex','sex','blood', 'bloodchem1', 'bloodchem2','temperature','race','heart','psych1','glucose','psych2','dose', 'psych3', 'bp', 'bloodchem3','confidence', 'bloodchem4', 'comorbidity', 'totalcost','breathing','age', 'sleep', 'dnr','bloodchem5','pdeath','meals','pain','primary', 'psych4','disability','administratorcost','urine','diabetes','income','extraprimary','bloodchem6','education','psych5', 'psych6','information','cancer','death']
    df = df[col_to_keep]
# 'disability',
# encode sex cancer diabetes
    df.replace('nan',0,inplace=True)

    df.replace('', 0, inplace=True)
    df.replace('male', 0, inplace=True)
    df.replace('1', 1, inplace=True)
    df.replace('M', 0, inplace=True)
    df.replace('Male', 0, inplace=True)
    df.replace('female', 1, inplace=True)
    
    df.replace('yes', 1, inplace=True)
    df.replace('no', 0, inplace=True)
    df.replace('metastatic', -1, inplace=True)
    
    df.replace('white', 5, inplace=True)
    df.replace('black', 1, inplace=True)
    df.replace('hispanic', 2, inplace=True)
    df.replace('other', 3, inplace=True)
    df.replace('asian', 4, inplace=True)
    
    df.replace('dnr', 1, inplace=True)
    df.replace('no dnr', 4, inplace=True)
    df.replace('dnr before sadm', 2, inplace=True)
    df.replace('dnr after sadm', 3, inplace=True)
    
    df.replace('Cirrhosis', 1, inplace=True)
    df.replace('Colon Cancer', 8, inplace=True)
    df.replace('ARF/MOSF w/Sepsis', 2, inplace=True)
    df.replace('COPD', 3, inplace=True)
    df.replace('MOSF w/Malig', 4, inplace=True)
    df.replace('CHF', 5, inplace=True)
    df.replace('Lung Cancer', 6, inplace=True)
    df.replace('Coma', 7, inplace=True)
    
    df.replace('under $11k', 1, inplace=True)
    df.replace('$11-$25k', 2, inplace=True)
    df.replace('$25-$50k', 3, inplace=True)
    df.replace('>$50k', 4, inplace=True)
    
    df.replace('ARF/MOSF', 1, inplace=True)
    df.replace('Cancer', 2, inplace=True)
    df.replace('COPD/CHF/Cirrhosis', 3, inplace=True)
    
    df.replace('<2 mo. follow-up', 1, inplace=True)
    df.replace('no(M2 and SIP pres)', 2, inplace=True)
    df.replace('SIP>=30', 3, inplace=True)
    df.replace('adl>=4 (>=5 if sur)', 4, inplace=True)
    df.replace('Coma or Intub', 5, inplace=True)
    
    df.fillna(0, inplace=True)
    #df = df[df['confidence'] >= 10]
    one_hot = pd.get_dummies(df, columns = ['sex', 'cancer','diabetes']) 
    
    return one_hot
    
def split_feature_label(df):
    y = df['death']
    X = df.drop(columns=['death'])
    #pca = PCA()
    #principal = pca.fit_transform(X)
    # Convert to dataframe
    #component_names = [f"PC{i+1}" for i in range(X.shape[1])]
    #X = pd.DataFrame(principal, columns=component_names)

    #X.head()
    return y, X
    # print(X)
    # print(y)

    # death_0 = y.tolist().count(0)
    # death_1 = y.tolist().count(1)
    # percent_death_0 = 100 * death_0 / (death_0 + death_1)
    # percent_death_1 = 100 * death_1 / (death_0 + death_1)
    # print(f'Survived: {death_0}, or {percent_death_0:.2f}%')
    # print(f'Died: {death_1}, or {percent_death_1:.2f}%')

def standardize(X):
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(X.select_dtypes(include=['float64']))
    X[X.select_dtypes(include=['float64']).columns] = X_numeric
    X = scaler.fit_transform(X)
    dump(scaler, open('scaler.pkl', 'wb'))
    return X

def train_model(X, y):
    # Split data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=.3, random_state=42)
    #X_train, scaler = standardize(X_train)
    #X_val = scaler.transform(X_val)
    #X_test = scaler.transform(X_test)
    # Define the neural network model
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),  # Input layer
        layers.Dense(47, activation='relu',kernel_regularizer=keras.regularizers.L1L2(l1=1e-3,l2=1e-2),bias_regularizer=regularizers.L2(1e-2),activity_regularizer=regularizers.L2(1e-3)),     # Hidden layer with 128 neurons and ReLU activation
        layers.Dense(47, activation='selu',kernel_regularizer=keras.regularizers.L1L2(l1=1e-3,l2=1e-2),bias_regularizer=regularizers.L2(1e-2),activity_regularizer=regularizers.L2(1e-3)),      # Another hidden layer with 64 neurons and ReLU activation 
        layers.Dense(1, activation='sigmoid')     # Output layer with sigmoid activation for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model 'binary_crossentropy'
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    model.save('example.h5')
    
    print(f'Test accuracy: {test_accuracy}')

    # Optionally, you can plot training history to visualize model performance
    import matplotlib.pyplot as plt

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig('greg')


if __name__ == "__main__":
    data_path = './TD_HOSPITAL_TRAIN.csv'
    df = pd.read_csv(data_path)
    cleaned_data = data_preprocessing(df)
    y, X = split_feature_label(cleaned_data)
    X = standardize(X)
    X = np.asarray(X).astype('float32')
    train_model(X, y)
    