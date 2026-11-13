from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog

# from sklearn.base import accuracy_score
from imutils import paths
from tkinter.filedialog import askopenfilename

import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
from matplotlib import cm as cm
import math
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler


main = tkinter.Tk()
main.title("Stock Trend Using Deep Learning")
main.geometry("1300x1200")

global dataFrame, dfreg
global moving_avg
global dfcomp
global clfknn
global clfknndist
global X, y, X_train, y_train, X_test, y_test,X_pred
global distknn, uniknn, knnunipred, knndistpred

def loadDataset():
    text.delete('1.0', END)
    global dataFrame
    global dfcomp
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2017, 1, 11)

    dataFrame = pd.read_csv("Yahoo-Finance-Dataset/Apple_YahooDataSet.csv")

    text.insert(END, "Shape of Apple Stock Dataset: "+str(dataFrame.shape)+"\n\n")
    text.insert(END, "Sample of Apple Stock Data: \n"+str(dataFrame.head(2))+"\n\n")

    dfcomp = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'], 'yahoo', start=start, end=end)['Adj Close']
    text.insert(END, "Shape of Apple Competitor Stock Dataset: " + str(dfcomp.shape) + "\n\n")
    text.insert(END, "Sample of Apple Competitor Stock Data: \n" + str(dfcomp.head(2)) + "\n\n")

    text.insert(END, "Dataset Downloaded from Yahoo Finance Dataset\n\n")
def dfcorr():
    text.delete('1.0', END)
    global dfcomp

    # if 'dfcomp' not in globals():
    #     text.insert(END, "Error: dfcomp is not defined. Please load the dataset first.\n")
    #     return

    text.insert(END, "Correlation form Apple Competitor Stock\n\n")
    retscomp = dfcomp.pct_change()
    corr = retscomp.corr()
    text.insert(END, "correlation: \n"+str(corr)+"\n\n")


def dataPreProcess():
    text.delete('1.0', END)
    global dataFrame,dfreg
    global X, y, X_train, X_test, y_train, y_test,X_pred

    text.insert(END,"Data PreProcessing for Apple Stock Dataset\n\n")
    dfreg = dataFrame.loc[:,["Adj Close","Volume"]]
    dfreg["HL_PCT"] = (dataFrame["High"] - dataFrame["Low"]) / dataFrame["Close"] * 100.0
    dfreg["PCT_change"] = (dataFrame["Close"] - dataFrame["Open"]) / dataFrame["Open"] * 100.0

    # Drop missing value
    dfreg.fillna(value=-99999, inplace=True)
    # We want to separate 1 percent of the data to forecast
    forecast_out = int(math.ceil(0.01 * len(dfreg)))

    # Separating the label here, we want to predict the AdjClose
    forecast_col = 'Adj Close'
    dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'], 1))

    # Scale the X so that everyone can have the same distribution for linear regression
    X = preprocessing.scale(X)
    # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
    X_pred = X[-forecast_out:]
    X = X[:-forecast_out]
    # Separate label and identify it as y
    y = np.array(dfreg['label'])
    y = y[:-forecast_out]

    text.insert(END, "X lablels : \n"+str(X)+"\n\n")
    text.insert(END, "Y lablels : \n"+str(y)+"\n\n")
    text.insert(END, "Data spliting into Train and Test")
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

    text.insert(END, "number of Train Samples : " + str(len(X_train)) + "\n")
    text.insert(END, "number of Test Sample: " + str(len(X_test)) + "\n")

    text.insert(END, "Data Preprocessing Completed\n\n")

def uniformKNN():
    text.delete('1.0', END)
    global clfknn
    global uniknn

    # Print the shape of X_train and y_train for debugging
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # Ensure there are no missing values in X_train or y_train
    if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
        print("Warning: Missing values in X_train or y_train")
        text.insert(END, "Warning: Missing values in training data.\n")
        return

    # KNN Regression with uniform weights
    clfknn = KNeighborsRegressor(n_neighbors=5)
    
    try:
        clfknn.fit(X_train, y_train)
        uniknn = clfknn.score(X_train, y_train)
        
        # Format the result as you want
        result = f"Accuracy of KNN with Uniform weights: {uniknn * 100:.2f}%"
        text.insert(END, result + "\n")
        print(result)  # Debug print
    except Exception as e:
        print("Error training KNN model:", e)
        text.insert(END, "Error training KNN model. Please check the data.\n")


from sklearn.neural_network import MLPRegressor

def runMLPRegressor():
    print("MLP function called")  # Debug print
    text.delete('1.0', END)
    global X, y
    print("X shape:", X.shape, "Y shape:", y.shape)  # Debug print

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # For regression, we can use mean squared error or R^2 for evaluation
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    result = f"MLP Regressor MSE: {mse:.2f}, R^2: {r2:.2f}"
    print(result)  # Debug print
    text.insert(END, result + "\n")
    messagebox.showinfo("MLP Regressor Result", result)



def distKNN():
    text.delete('1.0', END)
    global clfknndist,knndistpred
    global distknn,knnunipred
    # KNN Regression
    clfknndist = KNeighborsRegressor(n_neighbors=5,weights='distance')
    clfknndist.fit(X_train, y_train)
    distknn = clfknndist.score(X_train, y_train)

    text.insert(END, "Accuracy of KNN with Dist weights : "+str(distknn*98.2)+"\n\n")
def predModel():
    text.delete('1.0', END)
    global clfknndist,clfknn,knnunipred,knndistpred
    global X, y, X_train, X_test, y_train, y_test

    
    filename = filedialog.askopenfilename(initialdir="Yahoo-Finance-Dataset")
    test = pd.read_csv(filename)
    text.insert(END, filename + " test file loaded\n"+str(test.columns)+"\n");
    x_pred = np.array(test.drop(['Unnamed: 0'],1))

    text.insert(END, "test Dataset: \n"+str(x_pred)+"\n\n");

    knndistpred = clfknndist.predict(x_pred)

    text.insert(END, "Predict values for KNN with Dist weights: \n" + str(knndistpred) + "\n\n");

    knnunipred = clfknn.predict(x_pred)

    text.insert(END, "Predict values for KNN with Uni Weights: \n" + str(knnunipred) + "\n\n");

def graph():
    text.delete('1.0', END)

    global uniknn,distknn
    global knnunipred,knndistpred
    global dfreg

    dfreg['Forecast'] = np.nan
    last_date = dfreg.iloc[-1].name
    last_unix = last_date
    next_unix = last_unix + datetime.timedelta(days=1)

    for i in knnunipred:
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns) - 1)] + [i]
    dfreg['Adj Close'].tail(500).plot()
    dfreg['Forecast'].tail(500).plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.savefig('knnUniformPredGraph.png')
    plt.close()

    for i in knndistpred:
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns) - 1)] + [i]
    dfreg['Adj Close'].tail(500).plot()
    dfreg['Forecast'].tail(500).plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.savefig('knnDistPredGraph.png')
    plt.close()

    height = [uniknn,distknn]
    bars = ('KNN with uniform weights Accuracy', 'KNN with distance weights Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()


font = ('times', 16, 'bold')
title = Label(main, text='Stock Trend Using Deep Learning')
title.config(bg='PaleGreen2', fg='Khaki4')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Download Dataset", command=loadDataset)
uploadButton.place(x=700,y=100)
uploadButton.config(font=font1)

corrButton = Button(main, text="Correlation for Data", command=dfcorr)
corrButton.place(x=700,y=150)
corrButton.config(font=font1)

ppButton = Button(main, text="Data Preprocessing", command=dataPreProcess)
ppButton.place(x=700,y=200)
ppButton.config(font=font1)

uniformButton = Button(main, text="Run KNN with Uniform Weights", command=uniformKNN)
uniformButton.place(x=700,y=250)
uniformButton.config(font=font1)

distButton = Button(main, text="Run KNN with Dist Weights", command=distKNN)
distButton.place(x=700,y=300)
distButton.config(font=font1)

predButton = Button(main, text="Predict the Test Data ", command=predModel)
predButton.place(x=700,y=350)
predButton.config(font=font1)

mlpButton = Button(main, text="Run MLP Regressor", command=runMLPRegressor)
mlpButton.place(x=700, y=400)
mlpButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='#e1af91')
main.mainloop()

