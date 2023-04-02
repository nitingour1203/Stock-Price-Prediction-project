# Stock-Price-Prediction

<b>Stock Price Prediction using machine learning helps you discover the future value of company stock and other financial assets traded on an exchange. The entire idea of predicting stock prices is to gain significant profits. Predicting how the stock market will perform is a hard task to do. There are other factors involved in the prediction, such as physical and psychological factors, rational and irrational behavior, and so on. All these factors combine to make share prices dynamic and volatile. This makes it very difficult to predict stock prices with high accuracy. .<b>




# Understanding Long Short Term Memory Network

<b> Long Short Term Memory Network (LSTM) for building your model to predict the stock prices of Google.<b> LTSMs are a type of Recurrent Neural Network for learning long-term dependencies. It is commonly used for processing and predicting time-series data. 

![image](https://user-images.githubusercontent.com/48796009/227759775-7a8731c5-b647-42ae-b26e-38d721383599.png)


# LSTMs for the time-series data

 move on to the LSTM model. LSTM, short for Long Short-term Memory, is an extremely powerful algorithm for time series. It can capture historical trend patterns, and predict future values with high accuracy. 

In a nutshell, the key component to understand an LSTM model is the Cell State (Ct), which represents the internal short-term and long-term memories of a cell. 

![image](https://user-images.githubusercontent.com/48796009/227760916-69aac0f4-c338-45af-9564-a164dc6ae883.png)

To control and manage the cell state, an LSTM model contains three gates/layers. It’s worth mentioning that the “gates” here can be treated as filters to let information in (being remembered) or out (being forgotten). 

![Stock-Price-Prediction-project-dashboard-2](https://user-images.githubusercontent.com/48796009/229339469-4f8d3dc2-8c00-438c-97ad-80aaf30cbd5f.gif)


* Forget gate: 

![image](https://user-images.githubusercontent.com/48796009/227760836-a073a267-93d2-455e-9f80-c351aa70b7ac.png)


As the name implies, forget gate decides which information to throw away from the current cell state. Mathematically, it applies a sigmoid function to output/returns a value between [0, 1] for each value from the previous cell state (Ct-1); here ‘1’ indicates “completely passing through” whereas ‘0’ indicates “completely filtering out”

* Input gate:

![image](https://user-images.githubusercontent.com/48796009/227760744-16b2c41c-45ba-46c7-9cc3-3f5685c3a268.png)

It’s used to choose which new information gets added and stored in the current cell state. In this layer, a sigmoid function is implemented to reduce the values in the input vector (it), and then a tanh function squashes each value between [-1, 1] (Ct). Element-by-element matrix multiplication of it and Ct represents new information that needs to be added to the current cell state. 

* output gate : 

![image](https://user-images.githubusercontent.com/48796009/227760778-7db0bb94-842d-432b-94e9-c8d2a37f7167.png)

The output gate is implemented to control the output flowing to the next cell state.  Similar to the input gate, an output gate applies a sigmoid and then a tanh function to filter out unwanted information, keeping only what we’ve decided to let through. 

Knowing the theory of LSTM, you must be wondering how it does at predicting real-world stock prices. We’ll find out in the next section, by building an LSTM model and comparing its performance against the two technical analysis models: SMA and EMA. 





 ## LSTMs for the time-series data:
 <b> LSTM, short for Long Short-term Memory, is an extremely powerful algorithm for time series. It can capture historical trend patterns, and predict future values with high accuracy. 
 <b> In a nutshell, the key component to understand an LSTM model is the Cell State (Ct), which represents the internal short-term and long-term memories of a cell. 

To control and manage the cell state, an LSTM model contains three gates/layers. It’s worth mentioning that the “gates” here can be treated as filters to let information in (being remembered) or out (being forgotten). 

## LSTMs work in a three-step process: -

* The first step in LSTM is to decide which information to be omitted from the cell in that particular time step. It is decided with the help of a sigmoid function. It looks at the previous state (ht-1) and the current input xt and computes the function.
* There are two functions in the second layer. The first is the sigmoid function, and the second is the tanh function. The sigmoid function decides which values to let through (0 or 1). The tanh function gives the weightage to the values passed, deciding their level of importance from -1 to 1.
* The third step is to decide what will be the final output. First, you need to run a sigmoid layer which determines what parts of the cell state make it to the output. Then, you must put the cell state through the tanh function to push the values between -1 and 1 and multiply it by the output of the sigmoid gate.

# Stock analysis:-fundamental analysis vs. technical analysis
<b> When it comes to stocks, fundamental and technical analyses are at opposite ends of the market analysis spectrum.

## 1. Fundamental analysis (you can read more about it here):-

* Evaluates a company’s stock by examining its intrinsic value, including but not limited to tangible assets, financial statements, management effectiveness, strategic initiatives, and consumer behaviors; essentially all the basics of a company.

* Being a relevant indicator for long-term investment, the fundamental analysis relies on both historical and present data to measure revenues, assets, costs, liabilities, and so on.

* Generally speaking, the results from fundamental analysis don’t change with short-term news. 

## 2. Technical analysis (you can read more about it here):-

* Analyzes measurable data from stock market activities, such as stock prices, historical returns, and volume of historical trades; i.e. quantitative information that could identify trading signals and capture the movement patterns of the stock market. 

* Technical analysis focuses on historical data and current data just like fundamental analysis, but it’s mainly used for short-term trading purposes.

* Due to its short-term nature, technical analysis results are easily influenced by news.

Simple MA
SMA, short for Simple Moving Average, calculates the average of a range of stock (closing) prices over a specific number of periods in that range. The formula for SMA is:

, where Pn = the stock price at time point n, N = the number of time points.

For this exercise of building an SMA model, we’ll use the Python code below to compute the 50-day SMA. We’ll also add a 200-day SMA for good measure.

* Popular technical analysis methodologies include moving average (MA), support and resistance levels, as well as trend lines and channels. 

<b>For our exercise, we’ll be looking at technical analysis solely and focusing on the Simple MA and Exponential MA techniques to predict stock prices. Additionally, we’ll utilize LSTM (Long Short-Term Memory), a deep learning framework for time-series, to build a predictive model and compare its performance against our technical analysis. 

<b>As stated in the disclaimer, stock trading strategy is not in the scope of this article. I’ll be using trading/investment terms only to help you better understand the analysis, but this is not financial advice. We’ll be using terms like:

<b>trend indicators:<b> statistics that represent the trend of stock prices,

<b>medium-term movements:<b> the 50-day movement trend of stock prices.

# Evaluation metrics and helper function:


Since stock prices prediction is essentially a regression problem, the RMSE (Root Mean Squared Error) and MAPE (Mean Absolute Percentage Error %) will be our current model evaluation metrics. Both are useful measures of forecast accuracy. 
![image](https://user-images.githubusercontent.com/48796009/227760247-faf0ea5d-e174-4ce9-877b-f931ba37413e.png)

![image](https://user-images.githubusercontent.com/48796009/227760251-79e9632a-1646-403c-9921-e704880b2c5e.png)

, where N = the number of time points, At = the actual / true stock price, Ft = the predicted / forecast value.

# Predicting stock price with Moving Average (MA) technique:
<b>MA is a popular method to smooth out random movements in the stock market. Similar to a sliding window, an MA is an average that moves along the time scale/periods; older data points get dropped as newer data points are added. 

<b>Commonly used periods are 20-day, 50-day, and 200-day MA for short-term, medium-term, and long-term investment respectively. 

<b> Two types of MA are most preferred by financial analysts: Simple MA and Exponential MA.

# Simple MA
SMA, short for Simple Moving Average, calculates the average of a range of stock (closing) prices over a specific number of periods in that range. The formula for SMA is:


![image](https://user-images.githubusercontent.com/48796009/227760366-03844cf9-1e80-4f99-a5bc-92795b094e1e.png)   , where Pn = the stock price at time point n, N = the number of time points.

 * In addition, the trend chart below shows the 50-day, 200-day SMA predictions compared with the true stock closing values.

![image](https://user-images.githubusercontent.com/48796009/227760417-9621460d-be90-4410-bbb3-760d7044e3ee.png)

It’s not surprising to see that the 50-day SMA is a better trend indicator than the 200-day SMA in terms of (short-to-) medium movements. Both indicators, nonetheless, seem to give smaller predictions than the actual values.

For this exercise of building an SMA model, we’ll use the Python code below to compute the 50-day SMA. We’ll also add a 200-day SMA for good measure.
