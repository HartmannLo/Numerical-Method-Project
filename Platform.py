#!/usr/bin/env python
# coding: utf-8

# In[1]:


def openNewWindow_2014Russian():
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(system)
 
    # sets the title of the
    # Toplevel widget
    newWindow.title("Russian data 2014-2017 Russo-Ukraine Conflict")
 
    # sets the geometry of toplevel
    newWindow.geometry("520x720")
 
    # A Label widget to show in toplevel
    frame11=Frame(newWindow)
    frame11.place(x=100,y=0)
    Label11=tk.Label(frame11,
          text ="Russian data"+"\n2014-2017 Russo-Ukraine Conflict")
    Label11.grid(row=0,column=0,padx=1,pady=0)
    
    
    import pandas as pd
    from collections import Counter
    Before_Russian_csv=pd.read_csv('Before_Russian.csv')
    count_status=Counter(Before_Russian_csv['Status'])


    sum_days = Before_Russian_csv['Days'].sum()
    sum_battles = count_status["Lose"]+count_status["Win"]

    count_Participant=Counter(Before_Russian_csv['Participant'])
    sum_Russia = count_Participant["Russia and Separatist"]+count_Participant["Russia"]
    sum_Separatist = count_Participant["Russia and Separatist"]+count_Participant["Separatist"]

    win_percentage= count_status["Win"]/sum_battles *100
    win_percentage=round(win_percentage,4)
    lose_percentage= count_status["Lose"]/sum_battles *100
    lose_percentage=round(lose_percentage,4)

    sum2 = Before_Russian_csv['Aircrafts'].sum()
    sum3 = Before_Russian_csv['Combat Vehicles'].sum()
    sum4 = Before_Russian_csv['Tanks'].sum()
    sum5 = Before_Russian_csv['Helicopters'].sum()
   
    #print output
    ans0="Total battle days: "+str(sum_days)+"\nNumber of battles: "+str(sum_battles)              +"\nNumber of win battles: "+str(count_status["Win"])+"\nNumber of lose battles: "+str(count_status["Lose"])             +"\nWin/lose percentage:"+str(win_percentage)+"%/"+str(lose_percentage)+"%"             +"\n\nRussia participant battles: "+str(sum_Russia)             +"\nSeparatist participant battles: "+str(sum_Separatist)            +"\nRussia and Separatist participant battles: "+str(count_Participant["Russia and Separatist"])             +'\n\nSum of Russian losses: '              +'\nAircrafts: '+ str(sum2)             +'\nCombat Vehicles: ' + str(sum3)             +'\nTanks: ' + str(sum4)             +'\nHelicopters: ' + str(sum5)
    
    frame12=tk.Frame(newWindow)
    frame12.place(x=100,y=100) #location
    Label_12=Label(frame12, text=ans0)
    Label_12.grid(row=0,column=0,padx=10,pady=0)
    
    frame50=Frame(newWindow)
    frame50.place(x=100,y=500) #location
    button_50=tk.Button(frame50, width=28,height=2,text="Open detailed Losses Chart",                   command=openNewWindow_Russian_Chart)
    button_50.grid(row=3,column=0,padx=4,pady=0)


# In[2]:


def openNewWindow_Russian_Chart():
    # Toplevel object which will
    # be treated as a new window
    newWindow1 = Toplevel(system)
 
    # sets the title of the
    # Toplevel widget
    newWindow1.title("Russian data 2014-2017 Russo-Ukraine Conflict Chart")
 
    # sets the geometry of toplevel
    newWindow1.geometry("1020x720")
    
    import pandas as pd
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    Before_Russian_csv=pd.read_csv('Before_Russian.csv')
    rows =Before_Russian_csv['Battle Number']
    columns = ['Battle Number','Aircrafts','Combat Vehicles','Tanks','Helicopters',]

    result = Before_Russian_csv.loc[rows, columns]
    result.set_index('Battle Number', inplace=True)
    plot = result.plot(title="United Russian Army Losses", grid=True,fontsize=12,figsize=(8, 6))
    plot.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
    fig = plot.get_figure()
    
    canvas1=FigureCanvasTkAgg(fig,master=newWindow1)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side="top",padx=7,pady=7)
    
    
    ans1="Battle Number:"+    "\n0 : Battle of Kramatorsk"+"  1 : Battle of Mariupol"+"  2 : First Battle of Donetsk Airport"+    "\n3 : Siege of the Luhansk Border Base"+"  4 : Zelenopillia rocket attack"+"  5 : Battle of Ilovaisk"+    "\n6 : Offensive on Mariupol"+"  7 : Second Battle of Donetsk Airport"+"  8 : Battle of Debaltseve"+    "\n9 : Shyrokyne standoff"+"  10 : Battle of Marinka"+"  11 : Battle of Svitlodarsk"+    "\n12 : Battle of Avdiivka"
        
    
    frame12=tk.Frame(newWindow1)
    frame12.place(x=250,y=550) #location
    Label_12=Label(frame12, text=ans1)
    Label_12.grid(row=0,column=0,padx=10,pady=0)


# In[3]:


def openNewWindow_2022RussianAll():
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(system)
 
    # sets the title of the
    # Toplevel widget
    newWindow.title("2022 Russian all Equipment Losses Predict")
 
    # sets the geometry of toplevel
    newWindow.geometry("1020x520")
 
    import numpy as np
    import numpy
    import matplotlib.pyplot as plt
    from pandas import read_csv
    import math
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


    #create dataset A and B, B is the future data
    def create_dataset(dataset, look_back=1):
        dataA=[]
        dataB=[]
        for i in range(len(dataset)-look_back-1):
            x = dataset[i:(i+look_back), 0]
            dataA.append(x)
            dataB.append(dataset[i + look_back, 0])
        return numpy.array(dataA), numpy.array(dataB)

    #load data
    data = read_csv('Current_Data_Russian.csv', usecols=[5], engine='python')
    dataset = data.values
    dataset = dataset.astype('float32')

    #normalize the data, let the data in range 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 1/2 for training data, 1/2 for test data
    training_data_size = int(len(dataset) * (1/2))
    test_data_size = int(len(dataset) * (1/2))
    train, test = dataset[0:training_data_size,:], dataset[training_data_size:len(dataset),:]

    #Creat train A, train B and test A, test B, B is future data
    look_back = 1
    trainA, trainB = create_dataset(train, look_back)
    testA, testB = create_dataset(test, look_back)
    # let input reshape to [samples, time steps, features]
    trainA = numpy.reshape(trainA, (trainA.shape[0], 1, trainA.shape[1]))
    testA = numpy.reshape(testA, (testA.shape[0], 1, testA.shape[1]))

    #Create LSTM model for training
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back))) #number of neutrons=1+2+1=4
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainA, trainB, epochs=66, batch_size=1, verbose=2)

    # Predict
    training_Predict = model.predict(trainA)
    test_Predict = model.predict(testA)

    # Back to the original data
    training_Predict = scaler.inverse_transform(training_Predict)
    trainB = scaler.inverse_transform([trainB])
    test_Predict = scaler.inverse_transform(test_Predict)
    testB = scaler.inverse_transform([testB])

    # calculate root mean squared error
    # import from sklearn.metrics
    training_RMSE_value = math.sqrt(mean_squared_error(trainB[0], training_Predict[:,0]))
    print("Training RMSE: {:.2f} ".format(training_RMSE_value))
    test_RMSE_value = math.sqrt(mean_squared_error(testB[0], test_Predict[:,0]))
    print("Test RMSE: {:.2f} ".format(test_RMSE_value))

    # Plot training data
    training_Predict_Plot=numpy.empty_like(dataset)
    training_Predict_Plot[:, :]=numpy.nan
    training_Predict_Plot[look_back:len(training_Predict)+look_back, :]=training_Predict

    # Plot test data
    test_Predict_Plot=numpy.empty_like(dataset)
    test_Predict_Plot[:, :] = numpy.nan
    test_Predict_Plot[len(training_Predict)+(look_back*2)+1:len(dataset)-1, :]=test_Predict

    # Plot the original data, training data and test data on tkinter
    
    result = plt.figure(figsize=(13,5))
    plt.plot(scaler.inverse_transform(dataset),label="Original Data")
    plt.plot(training_Predict_Plot,label="Training Predict Data")
    plt.plot(test_Predict_Plot,label="Test Predict Data")
    plt.xticks(np.arange(0, 66, 5))
    plt.title('2022 Russian all Equipment Losses Predict plot graph')
    plt.xlabel('The nth Battle Days')
    plt.ylabel('Russian all Equipment Losses')
    plt.legend(loc="upper right")
    plt.grid()
    fig = result.get_figure()
    
    canvas1=FigureCanvasTkAgg(fig,master=newWindow)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side="top",padx=7,pady=7)


# In[4]:


def openNewWindow_2022RussianA():
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(system)
 
    # sets the title of the
    # Toplevel widget
    newWindow.title("2022 Russian Aircraft Losses Predict")
 
    # sets the geometry of toplevel
    newWindow.geometry("1020x520")
 
    import numpy as np
    import numpy
    import matplotlib.pyplot as plt
    from pandas import read_csv
    import math
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


    #create dataset A and B, B is the future data
    def create_dataset(dataset, look_back=1):
        dataA=[]
        dataB=[]
        for i in range(len(dataset)-look_back-1):
            x = dataset[i:(i+look_back), 0]
            dataA.append(x)
            dataB.append(dataset[i + look_back, 0])
        return numpy.array(dataA), numpy.array(dataB)

    #load data
    data = read_csv('Current_Data_Russian.csv', usecols=[1], engine='python')
    dataset = data.values
    dataset = dataset.astype('float32')

    #normalize the data, let the data in range 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 1/2 for training data, 1/2 for test data
    training_data_size = int(len(dataset) * (1/2))
    test_data_size = int(len(dataset) * (1/2))
    train, test = dataset[0:training_data_size,:], dataset[training_data_size:len(dataset),:]

    #Creat train A, train B and test A, test B, B is future data
    look_back = 1
    trainA, trainB = create_dataset(train, look_back)
    testA, testB = create_dataset(test, look_back)
    # let input reshape to [samples, time steps, features]
    trainA = numpy.reshape(trainA, (trainA.shape[0], 1, trainA.shape[1]))
    testA = numpy.reshape(testA, (testA.shape[0], 1, testA.shape[1]))

    #Create LSTM model for training
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back))) #number of neutrons=1+2+1=4
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainA, trainB, epochs=66, batch_size=1, verbose=2)

    # Predict
    training_Predict = model.predict(trainA)
    test_Predict = model.predict(testA)

    # Back to the original data
    training_Predict = scaler.inverse_transform(training_Predict)
    trainB = scaler.inverse_transform([trainB])
    test_Predict = scaler.inverse_transform(test_Predict)
    testB = scaler.inverse_transform([testB])

    # calculate root mean squared error
    # import from sklearn.metrics
    training_RMSE_value = math.sqrt(mean_squared_error(trainB[0], training_Predict[:,0]))
    print("Training RMSE: {:.2f} ".format(training_RMSE_value))
    test_RMSE_value = math.sqrt(mean_squared_error(testB[0], test_Predict[:,0]))
    print("Test RMSE: {:.2f} ".format(test_RMSE_value))

    # Plot training data
    training_Predict_Plot=numpy.empty_like(dataset)
    training_Predict_Plot[:, :]=numpy.nan
    training_Predict_Plot[look_back:len(training_Predict)+look_back, :]=training_Predict

    # Plot test data
    test_Predict_Plot=numpy.empty_like(dataset)
    test_Predict_Plot[:, :] = numpy.nan
    test_Predict_Plot[len(training_Predict)+(look_back*2)+1:len(dataset)-1, :]=test_Predict

    # Plot the original data, training data and test data on tkinter
    result = plt.figure(figsize=(13,5))
    plt.plot(scaler.inverse_transform(dataset),label="Original Data")
    plt.plot(training_Predict_Plot,label="Training Predict Data")
    plt.plot(test_Predict_Plot,label="Test Predict Data")
    plt.xticks(np.arange(0, 66, 5))
    plt.title('2022 Russian Aircraft Losses Predict plot graph')
    plt.xlabel('The nth Battle Days')
    plt.ylabel('Russian Aircraft Losses')
    plt.legend(loc="upper right")
    plt.grid()
    fig = result.get_figure()
    
    canvas1=FigureCanvasTkAgg(fig,master=newWindow)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side="top",padx=7,pady=7)


# In[5]:


def openNewWindow_2022RussianCV():
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(system)
 
    # sets the title of the
    # Toplevel widget
    newWindow.title("2022 Russian Combat Vehicle Losses Predict")
 
    # sets the geometry of toplevel
    newWindow.geometry("1020x520")
 
    import numpy as np
    import numpy
    import matplotlib.pyplot as plt
    from pandas import read_csv
    import math
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


    #create dataset A and B, B is the future data
    def create_dataset(dataset, look_back=1):
        dataA=[]
        dataB=[]
        for i in range(len(dataset)-look_back-1):
            x = dataset[i:(i+look_back), 0]
            dataA.append(x)
            dataB.append(dataset[i + look_back, 0])
        return numpy.array(dataA), numpy.array(dataB)

    #load data
    data = read_csv('Current_Data_Russian.csv', usecols=[2], engine='python')
    dataset = data.values
    dataset = dataset.astype('float32')

    #normalize the data, let the data in range 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 1/2 for training data, 1/2 for test data
    training_data_size = int(len(dataset) * (1/2))
    test_data_size = int(len(dataset) * (1/2))
    train, test = dataset[0:training_data_size,:], dataset[training_data_size:len(dataset),:]

    #Creat train A, train B and test A, test B, B is future data
    look_back = 1
    trainA, trainB = create_dataset(train, look_back)
    testA, testB = create_dataset(test, look_back)
    # let input reshape to [samples, time steps, features]
    trainA = numpy.reshape(trainA, (trainA.shape[0], 1, trainA.shape[1]))
    testA = numpy.reshape(testA, (testA.shape[0], 1, testA.shape[1]))

    #Create LSTM model for training
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back))) #number of neutrons=1+2+1=4
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainA, trainB, epochs=66, batch_size=1, verbose=2)

    # Predict
    training_Predict = model.predict(trainA)
    test_Predict = model.predict(testA)

    # Back to the original data
    training_Predict = scaler.inverse_transform(training_Predict)
    trainB = scaler.inverse_transform([trainB])
    test_Predict = scaler.inverse_transform(test_Predict)
    testB = scaler.inverse_transform([testB])

    # calculate root mean squared error
    # import from sklearn.metrics
    training_RMSE_value = math.sqrt(mean_squared_error(trainB[0], training_Predict[:,0]))
    print("Training RMSE: {:.2f} ".format(training_RMSE_value))
    test_RMSE_value = math.sqrt(mean_squared_error(testB[0], test_Predict[:,0]))
    print("Test RMSE: {:.2f} ".format(test_RMSE_value))

    # Plot training data
    training_Predict_Plot=numpy.empty_like(dataset)
    training_Predict_Plot[:, :]=numpy.nan
    training_Predict_Plot[look_back:len(training_Predict)+look_back, :]=training_Predict

    # Plot test data
    test_Predict_Plot=numpy.empty_like(dataset)
    test_Predict_Plot[:, :] = numpy.nan
    test_Predict_Plot[len(training_Predict)+(look_back*2)+1:len(dataset)-1, :]=test_Predict

    # Plot the original data, training data and test data on tkinter
    result = plt.figure(figsize=(13,5))
    plt.plot(scaler.inverse_transform(dataset),label="Original Data")
    plt.plot(training_Predict_Plot,label="Training Predict Data")
    plt.plot(test_Predict_Plot,label="Test Predict Data")
    plt.xticks(np.arange(0, 66, 5))
    plt.title('2022 Russian Combat Vehicle Losses Predict plot graph')
    plt.xlabel('The nth Battle Days')
    plt.ylabel('Russian Combat Vehicle Losses')
    plt.legend(loc="upper right")
    plt.grid()
    fig = result.get_figure()

    canvas1=FigureCanvasTkAgg(fig,master=newWindow)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side="top",padx=7,pady=7)


# In[6]:


def openNewWindow_2022RussianT():
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(system)
 
    # sets the title of the
    # Toplevel widget
    newWindow.title("2022 Russian Tank Losses Predict")
 
    # sets the geometry of toplevel
    newWindow.geometry("1020x520")
 
    import numpy as np
    import numpy
    import matplotlib.pyplot as plt
    from pandas import read_csv
    import math
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    #create dataset A and B, B is the future data
    def create_dataset(dataset, look_back=1):
        dataA=[]
        dataB=[]
        for i in range(len(dataset)-look_back-1):
            x = dataset[i:(i+look_back), 0]
            dataA.append(x)
            dataB.append(dataset[i + look_back, 0])
        return numpy.array(dataA), numpy.array(dataB)

    #load data
    data = read_csv('Current_Data_Russian.csv', usecols=[3], engine='python')
    dataset = data.values
    dataset = dataset.astype('float32')

    #normalize the data, let the data in range 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 1/2 for training data, 1/2 for test data
    training_data_size = int(len(dataset) * (1/2))
    test_data_size = int(len(dataset) * (1/2))
    train, test = dataset[0:training_data_size,:], dataset[training_data_size:len(dataset),:]

    #Creat train A, train B and test A, test B, B is future data
    look_back = 1
    trainA, trainB = create_dataset(train, look_back)
    testA, testB = create_dataset(test, look_back)
    # let input reshape to [samples, time steps, features]
    trainA = numpy.reshape(trainA, (trainA.shape[0], 1, trainA.shape[1]))
    testA = numpy.reshape(testA, (testA.shape[0], 1, testA.shape[1]))

    #Create LSTM model for training
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back))) #number of neutrons=1+2+1=4
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainA, trainB, epochs=66, batch_size=1, verbose=2)

    # Predict
    training_Predict = model.predict(trainA)
    test_Predict = model.predict(testA)

    # Back to the original data
    training_Predict = scaler.inverse_transform(training_Predict)
    trainB = scaler.inverse_transform([trainB])
    test_Predict = scaler.inverse_transform(test_Predict)
    testB = scaler.inverse_transform([testB])

    # calculate root mean squared error
    # import from sklearn.metrics
    training_RMSE_value = math.sqrt(mean_squared_error(trainB[0], training_Predict[:,0]))
    print("Training RMSE: {:.2f} ".format(training_RMSE_value))
    test_RMSE_value = math.sqrt(mean_squared_error(testB[0], test_Predict[:,0]))
    print("Test RMSE: {:.2f} ".format(test_RMSE_value))

    # Plot training data
    training_Predict_Plot=numpy.empty_like(dataset)
    training_Predict_Plot[:, :]=numpy.nan
    training_Predict_Plot[look_back:len(training_Predict)+look_back, :]=training_Predict

    # Plot test data
    test_Predict_Plot=numpy.empty_like(dataset)
    test_Predict_Plot[:, :] = numpy.nan
    test_Predict_Plot[len(training_Predict)+(look_back*2)+1:len(dataset)-1, :]=test_Predict

    # Plot the original data, training data and test data on tkinter   
    result = plt.figure(figsize=(13,5))
    plt.plot(scaler.inverse_transform(dataset),label="Original Data")
    plt.plot(training_Predict_Plot,label="Training Predict Data")
    plt.plot(test_Predict_Plot,label="Test Predict Data")
    plt.xticks(np.arange(0, 66, 5))
    plt.title('2022 Russian Tank Losses Predict plot graph')
    plt.xlabel('The nth Battle Days')
    plt.ylabel('Russian Tank Losses')
    plt.legend(loc="upper right")
    plt.grid()
    fig = result.get_figure()
    
    canvas1=FigureCanvasTkAgg(fig,master=newWindow)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side="top",padx=7,pady=7)


# In[7]:


def openNewWindow_2022RussianH():
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(system)
 
    # sets the title of the
    # Toplevel widget
    newWindow.title("2022 Russian Helicopter Losses Predict")
 
    # sets the geometry of toplevel
    newWindow.geometry("1020x520")
 
    import numpy as np
    import numpy
    import matplotlib.pyplot as plt
    from pandas import read_csv
    import math
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    #create dataset A and B, B is the future data
    def create_dataset(dataset, look_back=1):
        dataA=[]
        dataB=[]
        for i in range(len(dataset)-look_back-1):
            x = dataset[i:(i+look_back), 0]
            dataA.append(x)
            dataB.append(dataset[i + look_back, 0])
        return numpy.array(dataA), numpy.array(dataB)

    #load data
    data = read_csv('Current_Data_Russian.csv', usecols=[4], engine='python')
    dataset = data.values
    dataset = dataset.astype('float32')

    #normalize the data, let the data in range 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 1/2 for training data, 1/2 for test data
    training_data_size = int(len(dataset) * (1/2))
    test_data_size = int(len(dataset) * (1/2))
    train, test = dataset[0:training_data_size,:], dataset[training_data_size:len(dataset),:]

    #Creat train A, train B and test A, test B, B is future data
    look_back = 1
    trainA, trainB = create_dataset(train, look_back)
    testA, testB = create_dataset(test, look_back)
    # let input reshape to [samples, time steps, features]
    trainA = numpy.reshape(trainA, (trainA.shape[0], 1, trainA.shape[1]))
    testA = numpy.reshape(testA, (testA.shape[0], 1, testA.shape[1]))

    #Create LSTM model for training
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back))) #number of neutrons=1+2+1=4
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainA, trainB, epochs=66, batch_size=1, verbose=2)

    # Predict
    training_Predict = model.predict(trainA)
    test_Predict = model.predict(testA)

    # Back to the original data
    training_Predict = scaler.inverse_transform(training_Predict)
    trainB = scaler.inverse_transform([trainB])
    test_Predict = scaler.inverse_transform(test_Predict)
    testB = scaler.inverse_transform([testB])

    # calculate root mean squared error
    # import from sklearn.metrics
    training_RMSE_value = math.sqrt(mean_squared_error(trainB[0], training_Predict[:,0]))
    print("Training RMSE: {:.2f} ".format(training_RMSE_value))
    test_RMSE_value = math.sqrt(mean_squared_error(testB[0], test_Predict[:,0]))
    print("Test RMSE: {:.2f} ".format(test_RMSE_value))

    # Plot training data
    training_Predict_Plot=numpy.empty_like(dataset)
    training_Predict_Plot[:, :]=numpy.nan
    training_Predict_Plot[look_back:len(training_Predict)+look_back, :]=training_Predict

    # Plot test data
    test_Predict_Plot=numpy.empty_like(dataset)
    test_Predict_Plot[:, :] = numpy.nan
    test_Predict_Plot[len(training_Predict)+(look_back*2)+1:len(dataset)-1, :]=test_Predict

    # Plot the original data, training data and test data on tkinter
    result = plt.figure(figsize=(13,5))
    plt.plot(scaler.inverse_transform(dataset),label="Original Data")
    plt.plot(training_Predict_Plot,label="Training Predict Data")
    plt.plot(test_Predict_Plot,label="Test Predict Data")
    plt.xticks(np.arange(0, 66, 5))
    plt.title('2022 Russian Helicopter Losses Predict plot graph')
    plt.xlabel('The nth Battle Days')
    plt.ylabel('Russian Helicopter Losses')
    plt.legend(loc="upper right")
    plt.grid()
    fig = result.get_figure()
    
    canvas1=FigureCanvasTkAgg(fig,master=newWindow)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side="top",padx=7,pady=7)


# In[8]:


def openNewWindow_2014Ukraine():
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(system)
 
    # sets the title of the
    # Toplevel widget
    newWindow.title("Russian data 2014-2017 Russo-Ukraine Conflict")
 
    # sets the geometry of toplevel
    newWindow.geometry("520x720")
 
    # A Label widget to show in toplevel
    frame11=Frame(newWindow)
    frame11.place(x=100,y=0)
    Label11=tk.Label(frame11,
          text ="Ukraine data"+"\n2014-2017 Russo-Ukraine Conflict")
    Label11.grid(row=0,column=0,padx=1,pady=0)
    
    
    import pandas as pd
    from collections import Counter
    Before_Ukraine_csv=pd.read_csv('Before_Ukraine.csv')
    count_status=Counter(Before_Ukraine_csv['Status'])


    sum_days = Before_Ukraine_csv['Days'].sum()
    sum_battles = count_status["Lose"]+count_status["Win"]

    win_percentage= count_status["Win"]/sum_battles *100
    win_percentage=round(win_percentage,4)
    lose_percentage= count_status["Lose"]/sum_battles *100
    lose_percentage=round(lose_percentage,4)

    sum2 = Before_Ukraine_csv['Aircrafts'].sum()
    sum3 = Before_Ukraine_csv['Combat Vehicles'].sum()
    sum4 = Before_Ukraine_csv['Tanks'].sum()
    sum5 = Before_Ukraine_csv['Helicopters'].sum()
   
    #print output
    ans2="Total battle days: "+str(sum_days)+"\nNumber of battles: "+str(sum_battles)              +"\nNumber of win battles: "+str(count_status["Win"])+"\nNumber of lose battles: "+str(count_status["Lose"])             +"\nWin/lose percentage:"+str(win_percentage)+"%/"+str(lose_percentage)+"%"             +'\n\nSum of Ukraine losses: '              +'\nAircrafts: '+ str(sum2)             +'\nCombat Vehicles: ' + str(sum3)             +'\nTanks: ' + str(sum4)             +'\nHelicopters: ' + str(sum5)
    
    frame12=tk.Frame(newWindow)
    frame12.place(x=100,y=100) #location
    Label_12=Label(frame12, text=ans2)
    Label_12.grid(row=0,column=0,padx=10,pady=0)
    
    frame50=Frame(newWindow)
    frame50.place(x=100,y=500) #location
    button_50=tk.Button(frame50, width=28,height=2,text="Open detailed Losses Chart",                   command=openNewWindow_Ukraine_Chart)
    button_50.grid(row=3,column=0,padx=4,pady=0)


# In[9]:


def openNewWindow_Ukraine_Chart():
    # Toplevel object which will
    # be treated as a new window
    newWindow1 = Toplevel(system)
 
    # sets the title of the
    # Toplevel widget
    newWindow1.title("Ukraine data 2014-2017 Russo-Ukraine Conflict Chart")
 
    # sets the geometry of toplevel
    newWindow1.geometry("1020x720")
    
    import pandas as pd
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    Before_Ukraine_csv=pd.read_csv('Before_Ukraine.csv')
    rows =Before_Ukraine_csv['Battle Number']
    columns = ['Battle Number','Aircrafts','Combat Vehicles','Tanks','Helicopters',]

    result = Before_Ukraine_csv.loc[rows, columns]
    result.set_index('Battle Number', inplace=True)
    plot = result.plot(title="Ukraine Army losses", grid=True,fontsize=12,figsize=(8, 6))
    plot.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
    fig = plot.get_figure()
    
    canvas1=FigureCanvasTkAgg(fig,master=newWindow1)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side="top",padx=7,pady=7)
    
    ans3="Battle Number:"+    "\n0 : Battle of Kramatorsk"+"  1 : Battle of Mariupol"+"  2 : First Battle of Donetsk Airport"+    "\n3 : Siege of the Luhansk Border Base"+"  4 : Zelenopillia rocket attack"+"  5 : Battle of Ilovaisk"+    "\n6 : Offensive on Mariupol"+"  7 : Second Battle of Donetsk Airport"+"  8 : Battle of Debaltseve"+    "\n9 : Shyrokyne standoff"+"  10 : Battle of Marinka"+"  11 : Battle of Svitlodarsk"+    "\n12 : Battle of Avdiivka"
    
        
    
    frame12=tk.Frame(newWindow1)
    frame12.place(x=250,y=550) #location
    Label_12=Label(frame12, text=ans3)
    Label_12.grid(row=0,column=0,padx=10,pady=0)


# In[10]:


def openNewWindow_2022UkraineAll():
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(system)
 
    # sets the title of the
    # Toplevel widget
    newWindow.title("2022 Ukraine all Equipment Losses Predict")
 
    # sets the geometry of toplevel
    newWindow.geometry("1020x520")
 
    import numpy as np
    import numpy
    import matplotlib.pyplot as plt
    from pandas import read_csv
    import math
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


    #create dataset A and B, B is the future data
    def create_dataset(dataset, look_back=1):
        dataA=[]
        dataB=[]
        for i in range(len(dataset)-look_back-1):
            x = dataset[i:(i+look_back), 0]
            dataA.append(x)
            dataB.append(dataset[i + look_back, 0])
        return numpy.array(dataA), numpy.array(dataB)

    #load data
    data = read_csv('Current_Data_Ukraine.csv', usecols=[5], engine='python')
    dataset = data.values
    dataset = dataset.astype('float32')

    #normalize the data, let the data in range 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 1/2 for training data, 1/2 for test data
    training_data_size = int(len(dataset) * (1/2))
    test_data_size = int(len(dataset) * (1/2))
    train, test = dataset[0:training_data_size,:], dataset[training_data_size:len(dataset),:]

    #Creat train A, train B and test A, test B, B is future data
    look_back = 1
    trainA, trainB = create_dataset(train, look_back)
    testA, testB = create_dataset(test, look_back)
    # let input reshape to [samples, time steps, features]
    trainA = numpy.reshape(trainA, (trainA.shape[0], 1, trainA.shape[1]))
    testA = numpy.reshape(testA, (testA.shape[0], 1, testA.shape[1]))

    #Create LSTM model for training
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back))) #number of neutrons=1+2+1=4
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainA, trainB, epochs=66, batch_size=1, verbose=2)

    # Predict
    training_Predict = model.predict(trainA)
    test_Predict = model.predict(testA)

    # Back to the original data
    training_Predict = scaler.inverse_transform(training_Predict)
    trainB = scaler.inverse_transform([trainB])
    test_Predict = scaler.inverse_transform(test_Predict)
    testB = scaler.inverse_transform([testB])

    # calculate root mean squared error
    # import from sklearn.metrics
    training_RMSE_value = math.sqrt(mean_squared_error(trainB[0], training_Predict[:,0]))
    print("Training RMSE: {:.2f} ".format(training_RMSE_value))
    test_RMSE_value = math.sqrt(mean_squared_error(testB[0], test_Predict[:,0]))
    print("Test RMSE: {:.2f} ".format(test_RMSE_value))

    # Plot training data
    training_Predict_Plot=numpy.empty_like(dataset)
    training_Predict_Plot[:, :]=numpy.nan
    training_Predict_Plot[look_back:len(training_Predict)+look_back, :]=training_Predict

    # Plot test data
    test_Predict_Plot=numpy.empty_like(dataset)
    test_Predict_Plot[:, :] = numpy.nan
    test_Predict_Plot[len(training_Predict)+(look_back*2)+1:len(dataset)-1, :]=test_Predict

    # Plot the original data, training data and test data on tkinter
    result = plt.figure(figsize=(13,5))
    plt.plot(scaler.inverse_transform(dataset),label="Original Data")
    plt.plot(training_Predict_Plot,label="Training Predict Data")
    plt.plot(test_Predict_Plot,label="Test Predict Data") 
    plt.xticks(np.arange(0, 66, 5))
    plt.title('2022 Ukraine all Equipment Losses Predict plot graph')
    plt.xlabel('The nth Battle Days')
    plt.ylabel('Ukraine all Equipment Losses')
    plt.legend(loc="upper right")
    plt.grid()
    fig = result.get_figure()
    
    canvas1=FigureCanvasTkAgg(fig,master=newWindow)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side="top",padx=7,pady=7)


# In[11]:


def openNewWindow_2022UkraineA():
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(system)
 
    # sets the title of the
    # Toplevel widget
    newWindow.title("2022 Ukraine Aircraft Losses Predict")
 
    # sets the geometry of toplevel
    newWindow.geometry("1020x520")
 
    import numpy as np
    import numpy
    import matplotlib.pyplot as plt
    from pandas import read_csv
    import math
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    #create dataset A and B, B is the future data
    def create_dataset(dataset, look_back=1):
        dataA=[]
        dataB=[]
        for i in range(len(dataset)-look_back-1):
            x = dataset[i:(i+look_back), 0]
            dataA.append(x)
            dataB.append(dataset[i + look_back, 0])
        return numpy.array(dataA), numpy.array(dataB)

    #load data
    data = read_csv('Current_Data_Ukraine.csv', usecols=[1], engine='python')
    dataset = data.values
    dataset = dataset.astype('float32')

    #normalize the data, let the data in range 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 1/2 for training data, 1/2 for test data
    training_data_size = int(len(dataset) * (1/2))
    test_data_size = int(len(dataset) * (1/2))
    train, test = dataset[0:training_data_size,:], dataset[training_data_size:len(dataset),:]

    #Creat train A, train B and test A, test B, B is future data
    look_back = 1
    trainA, trainB = create_dataset(train, look_back)
    testA, testB = create_dataset(test, look_back)
    # let input reshape to [samples, time steps, features]
    trainA = numpy.reshape(trainA, (trainA.shape[0], 1, trainA.shape[1]))
    testA = numpy.reshape(testA, (testA.shape[0], 1, testA.shape[1]))

    #Create LSTM model for training
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back))) #number of neutrons=1+2+1=4
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainA, trainB, epochs=66, batch_size=1, verbose=2)

    # Predict
    training_Predict = model.predict(trainA)
    test_Predict = model.predict(testA)

    # Back to the original data
    training_Predict = scaler.inverse_transform(training_Predict)
    trainB = scaler.inverse_transform([trainB])
    test_Predict = scaler.inverse_transform(test_Predict)
    testB = scaler.inverse_transform([testB])

    # calculate root mean squared error
    # import from sklearn.metrics
    training_RMSE_value = math.sqrt(mean_squared_error(trainB[0], training_Predict[:,0]))
    print("Training RMSE: {:.2f} ".format(training_RMSE_value))
    test_RMSE_value = math.sqrt(mean_squared_error(testB[0], test_Predict[:,0]))
    print("Test RMSE: {:.2f} ".format(test_RMSE_value))

    # Plot training data
    training_Predict_Plot=numpy.empty_like(dataset)
    training_Predict_Plot[:, :]=numpy.nan
    training_Predict_Plot[look_back:len(training_Predict)+look_back, :]=training_Predict

    # Plot test data
    test_Predict_Plot=numpy.empty_like(dataset)
    test_Predict_Plot[:, :] = numpy.nan
    test_Predict_Plot[len(training_Predict)+(look_back*2)+1:len(dataset)-1, :]=test_Predict

    # Plot the original data, training data and test data on tkinter
    result = plt.figure(figsize=(13,5))
    plt.plot(scaler.inverse_transform(dataset),label="Original Data")
    plt.plot(training_Predict_Plot,label="Training Predict Data")
    plt.plot(test_Predict_Plot,label="Test Predict Data")
    plt.xticks(np.arange(0, 66, 5))
    plt.title('2022 Ukraine Aircraft Losses Predict plot graph')
    plt.xlabel('The nth Battle Days')
    plt.ylabel('Ukraine Aircraft Losses')
    plt.legend(loc="upper right")
    plt.grid()
    fig = result.get_figure()
    
    canvas1=FigureCanvasTkAgg(fig,master=newWindow)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side="top",padx=7,pady=7)


# In[12]:


def openNewWindow_2022UkraineCV():
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(system)
 
    # sets the title of the
    # Toplevel widget
    newWindow.title("2022 Ukraine Combat Vehicle Losses Predict")
 
    # sets the geometry of toplevel
    newWindow.geometry("1020x520")
 
    import numpy as np
    import numpy
    import matplotlib.pyplot as plt
    from pandas import read_csv
    import math
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    #create dataset A and B, B is the future data
    def create_dataset(dataset, look_back=1):
        dataA=[]
        dataB=[]
        for i in range(len(dataset)-look_back-1):
            x = dataset[i:(i+look_back), 0]
            dataA.append(x)
            dataB.append(dataset[i + look_back, 0])
        return numpy.array(dataA), numpy.array(dataB)

    #load data
    data = read_csv('Current_Data_Ukraine.csv', usecols=[2], engine='python')
    dataset = data.values
    dataset = dataset.astype('float32')

    #normalize the data, let the data in range 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 1/2 for training data, 1/2 for test data
    training_data_size = int(len(dataset) * (1/2))
    test_data_size = int(len(dataset) * (1/2))
    train, test = dataset[0:training_data_size,:], dataset[training_data_size:len(dataset),:]

    #Creat train A, train B and test A, test B, B is future data
    look_back = 1
    trainA, trainB = create_dataset(train, look_back)
    testA, testB = create_dataset(test, look_back)
    # let input reshape to [samples, time steps, features]
    trainA = numpy.reshape(trainA, (trainA.shape[0], 1, trainA.shape[1]))
    testA = numpy.reshape(testA, (testA.shape[0], 1, testA.shape[1]))

    #Create LSTM model for training
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back))) #number of neutrons=1+2+1=4
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainA, trainB, epochs=66, batch_size=1, verbose=2)

    # Predict
    training_Predict = model.predict(trainA)
    test_Predict = model.predict(testA)

    # Back to the original data
    training_Predict = scaler.inverse_transform(training_Predict)
    trainB = scaler.inverse_transform([trainB])
    test_Predict = scaler.inverse_transform(test_Predict)
    testB = scaler.inverse_transform([testB])

    # calculate root mean squared error
    # import from sklearn.metrics
    training_RMSE_value = math.sqrt(mean_squared_error(trainB[0], training_Predict[:,0]))
    print("Training RMSE: {:.2f} ".format(training_RMSE_value))
    test_RMSE_value = math.sqrt(mean_squared_error(testB[0], test_Predict[:,0]))
    print("Test RMSE: {:.2f} ".format(test_RMSE_value))

    # Plot training data
    training_Predict_Plot=numpy.empty_like(dataset)
    training_Predict_Plot[:, :]=numpy.nan
    training_Predict_Plot[look_back:len(training_Predict)+look_back, :]=training_Predict

    # Plot test data
    test_Predict_Plot=numpy.empty_like(dataset)
    test_Predict_Plot[:, :] = numpy.nan
    test_Predict_Plot[len(training_Predict)+(look_back*2)+1:len(dataset)-1, :]=test_Predict

    # Plot the original data, training data and test data on tkinter
    result = plt.figure(figsize=(13,5))
    plt.plot(scaler.inverse_transform(dataset),label="Original Data")
    plt.plot(training_Predict_Plot,label="Training Predict Data")
    plt.plot(test_Predict_Plot,label="Test Predict Data")
    plt.xticks(np.arange(0, 66, 5))
    plt.title('2022 Ukraine Combat Vehicle Losses Predict plot graph')
    plt.xlabel('The nth Battle Days')
    plt.ylabel('Ukraine Combat Vehicle Losses')
    plt.legend(loc="upper right")
    plt.grid()
    fig = result.get_figure()

    canvas1=FigureCanvasTkAgg(fig,master=newWindow)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side="top",padx=7,pady=7)


# In[13]:


def openNewWindow_2022UkraineT():
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(system)
 
    # sets the title of the
    # Toplevel widget
    newWindow.title("2022 Ukraine Tank Losses Predict")
 
    # sets the geometry of toplevel
    newWindow.geometry("1020x520")
 
    import numpy as np
    import numpy
    import matplotlib.pyplot as plt
    from pandas import read_csv
    import math
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    #create dataset A and B, B is the future data
    def create_dataset(dataset, look_back=1):
        dataA=[]
        dataB=[]
        for i in range(len(dataset)-look_back-1):
            x = dataset[i:(i+look_back), 0]
            dataA.append(x)
            dataB.append(dataset[i + look_back, 0])
        return numpy.array(dataA), numpy.array(dataB)

    #load data
    data = read_csv('Current_Data_Ukraine.csv', usecols=[3], engine='python')
    dataset = data.values
    dataset = dataset.astype('float32')

    #normalize the data, let the data in range 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 1/2 for training data, 1/2 for test data
    training_data_size = int(len(dataset) * (1/2))
    test_data_size = int(len(dataset) * (1/2))
    train, test = dataset[0:training_data_size,:], dataset[training_data_size:len(dataset),:]

    #Creat train A, train B and test A, test B, B is future data
    look_back = 1
    trainA, trainB = create_dataset(train, look_back)
    testA, testB = create_dataset(test, look_back)
    # let input reshape to [samples, time steps, features]
    trainA = numpy.reshape(trainA, (trainA.shape[0], 1, trainA.shape[1]))
    testA = numpy.reshape(testA, (testA.shape[0], 1, testA.shape[1]))

    #Create LSTM model for training
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back))) #number of neutrons=1+2+1=4
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainA, trainB, epochs=66, batch_size=1, verbose=2)

    # Predict
    training_Predict = model.predict(trainA)
    test_Predict = model.predict(testA)

    # Back to the original data
    training_Predict = scaler.inverse_transform(training_Predict)
    trainB = scaler.inverse_transform([trainB])
    test_Predict = scaler.inverse_transform(test_Predict)
    testB = scaler.inverse_transform([testB])

    # calculate root mean squared error
    # import from sklearn.metrics
    training_RMSE_value = math.sqrt(mean_squared_error(trainB[0], training_Predict[:,0]))
    print("Training RMSE: {:.2f} ".format(training_RMSE_value))
    test_RMSE_value = math.sqrt(mean_squared_error(testB[0], test_Predict[:,0]))
    print("Test RMSE: {:.2f} ".format(test_RMSE_value))

    # Plot training data
    training_Predict_Plot=numpy.empty_like(dataset)
    training_Predict_Plot[:, :]=numpy.nan
    training_Predict_Plot[look_back:len(training_Predict)+look_back, :]=training_Predict

    # Plot test data
    test_Predict_Plot=numpy.empty_like(dataset)
    test_Predict_Plot[:, :] = numpy.nan
    test_Predict_Plot[len(training_Predict)+(look_back*2)+1:len(dataset)-1, :]=test_Predict

    # Plot the original data, training data and test data on tkinter
    result = plt.figure(figsize=(13,5))
    plt.plot(scaler.inverse_transform(dataset),label="Original Data")
    plt.plot(training_Predict_Plot,label="Training Predict Data")
    plt.plot(test_Predict_Plot,label="Test Predict Data")
    plt.xticks(np.arange(0, 66, 5))
    plt.title('2022 Ukraine Tank Losses Predict plot graph')
    plt.xlabel('The nth Battle Days')
    plt.ylabel('Ukraine Tank Losses')
    plt.legend(loc="upper right")
    plt.grid()
    fig = result.get_figure()
    
    canvas1=FigureCanvasTkAgg(fig,master=newWindow)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side="top",padx=7,pady=7)


# In[14]:


def openNewWindow_2022UkraineH():
    # Toplevel object which will
    # be treated as a new window
    newWindow = Toplevel(system)
 
    # sets the title of the
    # Toplevel widget
    newWindow.title("2022 Ukraine Helicopter Losses Predict")
 
    # sets the geometry of toplevel
    newWindow.geometry("1020x520")
 
    import numpy as np
    import numpy
    import matplotlib.pyplot as plt
    from pandas import read_csv
    import math
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import MultipleLocator
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


    #create dataset A and B, B is the future data
    def create_dataset(dataset, look_back=1):
        dataA=[]
        dataB=[]
        for i in range(len(dataset)-look_back-1):
            x = dataset[i:(i+look_back), 0]
            dataA.append(x)
            dataB.append(dataset[i + look_back, 0])
        return numpy.array(dataA), numpy.array(dataB)

    #load data
    data = read_csv('Current_Data_Ukraine.csv', usecols=[4], engine='python')
    dataset = data.values
    dataset = dataset.astype('float32')

    #normalize the data, let the data in range 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # 1/2 for training data, 1/2 for test data
    training_data_size = int(len(dataset) * (1/2))
    test_data_size = int(len(dataset) * (1/2))
    train, test = dataset[0:training_data_size,:], dataset[training_data_size:len(dataset),:]

    #Creat train A, train B and test A, test B, B is future data
    look_back = 1
    trainA, trainB = create_dataset(train, look_back)
    testA, testB = create_dataset(test, look_back)
    # let input reshape to [samples, time steps, features]
    trainA = numpy.reshape(trainA, (trainA.shape[0], 1, trainA.shape[1]))
    testA = numpy.reshape(testA, (testA.shape[0], 1, testA.shape[1]))

    #Create LSTM model for training
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back))) #number of neutrons=1+2+1=4
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainA, trainB, epochs=66, batch_size=1, verbose=2)

    # Predict
    training_Predict = model.predict(trainA)
    test_Predict = model.predict(testA)

    # Back to the original data
    training_Predict = scaler.inverse_transform(training_Predict)
    trainB = scaler.inverse_transform([trainB])
    test_Predict = scaler.inverse_transform(test_Predict)
    testB = scaler.inverse_transform([testB])

    # calculate root mean squared error
    # import from sklearn.metrics
    training_RMSE_value = math.sqrt(mean_squared_error(trainB[0], training_Predict[:,0]))
    print("Training RMSE: {:.2f} ".format(training_RMSE_value))
    test_RMSE_value = math.sqrt(mean_squared_error(testB[0], test_Predict[:,0]))
    print("Test RMSE: {:.2f} ".format(test_RMSE_value))

    # Plot training data
    training_Predict_Plot=numpy.empty_like(dataset)
    training_Predict_Plot[:, :]=numpy.nan
    training_Predict_Plot[look_back:len(training_Predict)+look_back, :]=training_Predict

    # Plot test data
    test_Predict_Plot=numpy.empty_like(dataset)
    test_Predict_Plot[:, :] = numpy.nan
    test_Predict_Plot[len(training_Predict)+(look_back*2)+1:len(dataset)-1, :]=test_Predict

    # Plot the original data, training data and test data on tkinter
    result = plt.figure(figsize=(13,5))
    plt.plot(scaler.inverse_transform(dataset),label="Original Data")
    plt.plot(training_Predict_Plot,label="Training Predict Data")
    plt.plot(test_Predict_Plot,label="Test Predict Data")
    plt.xticks(np.arange(0, 66, 5))
    plt.title('2022 Ukraine Helicopter Losses Predict plot graph')
    plt.xlabel('The nth Battle Days')
    plt.ylabel('Ukraine Helicopter Losses')
    plt.legend(loc="upper right")
    plt.grid()
    fig = result.get_figure()
    
    canvas1=FigureCanvasTkAgg(fig,master=newWindow)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side="top",padx=7,pady=7)


# In[15]:


from tkinter import *
import tkinter as tk
from tkinter.ttk import *
from PIL import ImageTk,Image
#from show_chart_Russian import*
system=Tk()
system.title("Russo-Ukrainian War data") #name of the window
system.geometry("1550x1050") #setting the size of window

#Russian
#Frame4 for the button
frame4=Frame(system)
frame4.place(x=270,y=200) #location
button_1=tk.Button(frame4, width=28,height=2,text="Russian data"+"\n2014-2017 Russo-Ukraine Conflict",                   command=openNewWindow_2014Russian) 
button_1.grid(row=3,column=0,padx=4,pady=0)

frame5=Frame(system)
frame5.place(x=520,y=200) #location
button_2=tk.Button(frame5, width=28,height=2,text="2022 Aircraft"+"\nLosses Predict",                   command=openNewWindow_2022RussianA)
button_2.grid(row=3,column=0,padx=14,pady=0)

frame6=Frame(system)
frame6.place(x=770,y=200) #location
button_3=tk.Button(frame6, width=28,height=2,text="2022 Combat Vehicle"+"\nLosses Predict",                   command=openNewWindow_2022RussianCV)
button_3.grid(row=3,column=0,padx=24,pady=0)


frame7=Frame(system)
frame7.place(x=250,y=300) #location
button_4=tk.Button(frame7, width=28,height=2,text="2022 All Equipment"+"\nLosses Predict",                   command=openNewWindow_2022RussianAll)
button_4.grid(row=3,column=0,padx=24,pady=0)
#openNewWindow_2022RussianAll
frame8=Frame(system)
frame8.place(x=510,y=300) #location
button_5=tk.Button(frame8, width=28,height=2,text="2022 Helicopter"+"\nLosses Predict",                   command=openNewWindow_2022RussianH)
button_5.grid(row=3,column=0,padx=24,pady=0)

frame9=Frame(system)
frame9.place(x=770,y=300) #location
button_6=tk.Button(frame9, width=28,height=2,text="2022 Tank"+"\nLosses Predict",                   command=openNewWindow_2022RussianT)
button_6.grid(row=3,column=0,padx=24,pady=0)

#Ukraine
frame10=Frame(system)
frame10.place(x=270,y=450) #location
button_7=tk.Button(frame10, width=28,height=2,text="Ukraine data"+"\n2014-2017 Russo-Ukraine Conflict",                   command=openNewWindow_2014Ukraine)
button_7.grid(row=3,column=0,padx=4,pady=0)

frame11=Frame(system)
frame11.place(x=520,y=450) #location
button_8=tk.Button(frame11, width=28,height=2,text="2022 Aircraft"+"\nLosses Predict",                   command=openNewWindow_2022UkraineA)
button_8.grid(row=3,column=0,padx=14,pady=0)
#command=openNewWindow_RussianLossesGraph

frame12=Frame(system)
frame12.place(x=770,y=450) #location
button_9=tk.Button(frame12, width=28,height=2,text="2022 Combat Vehicle"+"\nLosses Predict",                   command=openNewWindow_2022UkraineCV)
button_9.grid(row=3,column=0,padx=24,pady=0)

frame13=Frame(system)
frame13.place(x=250,y=550) #location
button_10=tk.Button(frame13, width=28,height=2,text="2022 All Equipment"+"\nLosses Predict",                   command=openNewWindow_2022UkraineAll)
button_10.grid(row=3,column=0,padx=24,pady=0)

frame14=Frame(system)
frame14.place(x=510,y=550) #location
button_11=tk.Button(frame14, width=28,height=2,text="2022 Helicopter"+"\nLosses Predict",                   command=openNewWindow_2022UkraineH)
button_11.grid(row=3,column=0,padx=24,pady=0)

frame14=Frame(system)
frame14.place(x=770,y=550) #location
button_11=tk.Button(frame14, width=28,height=2,text="2022 Tank"+"\nLosses Predict",                   command=openNewWindow_2022UkraineT)
button_11.grid(row=3,column=0,padx=24,pady=0)

#Frame1 for the 2 inputs and the 2 labels
frame1=Frame(system) 
frame1.place(x=620,y=0) #location
label_1=tk.Label(frame1, text="Russo & Ukrainian War",font=("Arial", 21))
label_1.grid(row=0,column=0,padx=1,pady=0)

frame2=Frame(system) 
frame2.place(x=340,y=150) #location
label_2=tk.Label(frame2, text="Russian data",font=("Arial", 12))
label_2.grid(row=0,column=0,padx=1,pady=0)

frame3=Frame(system) 
frame3.place(x=340,y=400) #location
label_3=tk.Label(frame3, text="Ukraine Data",font=("Arial", 12))
label_3.grid(row=0,column=0,padx=1,pady=0)

system.mainloop()


# In[ ]:





# In[ ]:




