import pandas as pd
import numpy as np
from yahoofinancials import YahooFinancials
import mplfinance
import re
from datetime import *
import numpy as np
# import datetime
import ipywidgets as widgets
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('whitegrid', {'facecolor':'white'})


highs, lows, opens, closes, total = [], [], [], [], []

def get_data(stock_id, start, end):
    global highs, lows, opens, closes, total
    highs, lows, opens, closes, total = [], [], [], [], []

    pattern = re.compile(r'([\S]+)')
    stock = pattern.findall(stock_id)
    
    yahoo_financials= YahooFinancials(stock[0])
    stats = (yahoo_financials.get_historical_price_data(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), "daily"))
    
    for day in stats[stock[0]]["prices"]:
        highs.append(day['high'])
        lows.append(day['low'])
        opens.append(day['open'])
        closes.append(day['close'])

    for i in range(len(opens)):
        total.append([opens[i], lows[i], highs[i], closes[i]])
    
    return total
#     print(total)


list_of_companies = ['GOOGL (GOOGLE)', 'AMZN (AMAZON)', 'NFLX (NETFLIX)', 'IBM (IBM)',
                     'DIS (DISNEY)', 'AAPL (APPLE)', 'NKE (NIKE)', 'MSFT (MICROSOFT)',
                     'FB (FACEBOOK)', 'AXP (AMERICAN EXPRESS)', 'SKX (SKECHERS)']

dropdown_comp = widgets.Dropdown(options = list_of_companies, description = 'Company')
date_picker_1 = widgets.DatePicker(description='Start Date', disabled = False, value = date(2015, 5, 23))
date_picker_2 = widgets.DatePicker(description='End Date', disabled = False, value = date(2020, 5, 23))
Next_Month_Values = widgets.Text(value='Hello World', placeholder='Type something', description='String:', disabled=False)
Training_Data = widgets.Output()
Testing_Data = widgets.Output()
plot_output = widgets.Output()
Next_Month_Values = widgets.Output()

df, dfP = [], []
YValidation, YPrediction, YPrediction_NextMonth = [], [], []

def Predictor(stock, start, end):
    global df, dfp, YValidation, YPrediction, YPrediction_NextMonth
    
    Training_Data.clear_output()
    Testing_Data.clear_output()
    plot_output.clear_output()
    Next_Month_Values.clear_output()

    lst = get_data(stock, start, end)
    i = len(lst)
    days = (len(opens)-1)//5
    total_training=lst[0:i-days]
    total_validation=lst[i-days:]
    
    df = pd.DataFrame(total_training, columns = ['Open', 'Low', 'High', 'Close'], dtype=float)
#     x = df.values #returns a numpy array
#     min_max_scaler = preprocessing.MinMaxScaler()
#     x_scaled = min_max_scaler.fit_transform(x)
#     df = pd.DataFrame(x_scaled)
    XTrain = df.iloc[:, :-1]
    YTrain = df.iloc[:, -1:]
    
    clf = LinearRegression()
    clf.fit(XTrain, YTrain)
    
#     print("\n\n")    
    
    dfP = pd.DataFrame(total_validation, columns = ['Open', 'Low', 'High', 'Close'], dtype=float)
    XValidation = dfP.iloc[:, :-1]
    YValidation = dfP.iloc[:, -1:]
    
    YPrediction = clf.predict(XValidation)
    YPrediction_NextMonth = clf.predict(XValidation.iloc[-30:, :])
    PredictedValuesForNextMonth = ''
    for value in YPrediction_NextMonth:
        count = np.where(YPrediction_NextMonth == value)
        PredictedValuesForNextMonth += f'Day {count[0][0]+1}: {round(value[0], 2)} \n'
    with Training_Data:
        display(df)
    with Testing_Data:
        display(dfP)
    with Next_Month_Values:
        print('Predicted Values for Next Month')
        print(PredictedValuesForNextMonth)
    with plot_output:
        plt.subplots(1, 2, figsize = (15, 7))
        plt.subplot(1, 2, 1)
        plt.title('Original Values', fontweight = 'bold')
        plt.ylabel('Closing values of Stock', fontweight = 'bold', fontsize = 14)
        plt.plot(df.index[-days:], YValidation)
        plt.subplot(1, 2, 2)
        plt.title('Predicted Values', fontweight = 'bold')
        plt.ylabel('Closing values of Stock', fontweight = 'bold', fontsize = 14)
        plt.plot(df.index[-days:], YPrediction)
        plt.show()

        plt.figure(figsize = (10, 10))
        plt.title('Original Vs. Predicted Values', fontweight = 'bold')
        plt.ylabel('Closing values of Stock', fontweight = 'bold', fontsize = 14)
        plt.plot(df.index[-days:], YValidation)
        plt.plot(df.index[-days:], YPrediction)
        plt.show()

#     print(XValidation)
#     print("\n\nPrediction")
#     YPrediction = clf.predict(XValidation)
    
#     print("\nOriginal")
#     print(YValidation)
    
#     print("\nPredicted")
#     print(YPrediction)
    
def dropdown_comp_evenhandler(change):
    Predictor(change.new, date_picker_1.value, date_picker_2.value)

def date_picker_1_eventhandler(change):
    Predictor(dropdown_comp.value, change.new, date_picker_2.value)

def date_picker_2_eventhandler(change):
    Predictor(dropdown_comp.value, date_picker_1.value, change.new)


dropdown_comp.observe(dropdown_comp_evenhandler, names = 'value')
date_picker_1.observe(date_picker_1_eventhandler, names = 'value')
date_picker_2.observe(date_picker_2_eventhandler, names = 'value')

item_layout = widgets.Layout(margin = '10 10 50px 10')
input_widgets = widgets.HBox([dropdown_comp, date_picker_1, date_picker_2])
tabs = widgets.Tab([Training_Data, Testing_Data, plot_output, Next_Month_Values], layout = item_layout)
tabs.set_title(0, 'Training Data')
tabs.set_title(1, 'Testing Data')
tabs.set_title(2, 'Visualisation')
tabs.set_title(3, 'Predicted Values for Next Month')
dashboard = widgets.VBox([input_widgets, tabs])
display(dashboard)
