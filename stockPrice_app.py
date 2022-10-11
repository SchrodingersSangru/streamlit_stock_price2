# CREATED BY ANGEL SANTANA HERNANDEZ
# # 
# #
# #
#



# =========================================================================================
#                                IMPORT LIBRARIES
# =========================================================================================

import pandas as pd
import numpy as np
from datetime import date
import datetime
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import streamlit as st

# STOCK DATA FROM YAHOO FINANCE
import yfinance as yf

# =========================================================================================
#                               CONFIGURE PAGE | PAGE DETAILS
# =========================================================================================

# --- CONGIF PAGE AND PAGE LAYOUT ---

st.set_page_config(page_title='Stock Price Prediction', # APP NAME FOR BROWSER
                    layout= 'wide' # PAGE LAYOUT
                    )

st.title("Stock Price Prediction 🤑💲") # SET PAGE TITLE
# --- END CONFIG PAGE ---


st.markdown('---') # Add a page break


# --- DESCRIBE WEB APPLICATION ---
st.header('How to use the web app?') 

bullet_points = '''
- FIRST INPUT:
    - Enter the ticker symbol for stock of your choice.
    - You can use the table to the side to search through different stocks.
- SECOND INPUT:
    - Enter the amount of days out you would like to see the predicted stock price.
    - ONLY positive numbers, please :)
- As you update the stock you want to see, there will be a line graph to show you the price trend.


'''
with st.expander('🧑‍🏫 INSTRUCTIONS'):
    st.markdown(bullet_points)

not_a_recommendation = '''
- This is not a tool to actually predict stock prices.
- Do not consult this app for your PERSONAL choices on buying stock.
- THIS APP IS JUST FOR FUN AND TO TEST MY PERSONAL KNOWLEDGE OF MACHINE LEARNING.

'''

with st.expander('⚠️ PLEASE READ'):
    st.markdown(not_a_recommendation)

st.markdown('---') # Add a page break

# --- END DESCRIPTION ---
# =========================================================================================


# --- FUNCTION TO READ IN DATA ---
def showStockNames():
    # ONLY NEED SYMBOL AND NAME
    stockName_df = pd.read_csv(
        'Data/stockNames.csv', 
        index_col = 0, 
        usecols= ['Symbol', 'Name']
        )

    return stockName_df

# --- END READ DATA FUNCTION ---


# =========================================================================================
#                                   INPUT LAYOUTS
# =========================================================================================
# PAGE LAYOUT
user_column, forecast_column, stock_column = st.columns((1, 1, 2))

# SYMBOL INPUT
with user_column:
    userInput = st.text_input('Enter a stock you would like to check?')
    userInput = userInput.upper()

# PREDICT NUMBER OF DAYS OUT
with forecast_column:
    forecastDays = st.number_input(label = 'Forecast Days...', step=1)
    if forecastDays < 0:
        st.write('ERROR: Number must be positive')

# SHOW STOCKS FOR USERS TO LOOK THROUGH
with stock_column:
    with st.expander('👀Need help? Look for stock names HERE.'):
        st.dataframe(showStockNames())
# =========================================================================================



# =========================================================================================
#                            RECIEVE DATA FROM YAHOO FINANCE API
# =========================================================================================
# Structure:
# Data recieved from API is based on user input of the stock
# Pass in user input, send the stock name to api
# Download data from that stock


def get_stock_data(userInput):

    # IF USERINPUT IS EMPTY
    if not userInput:
        return userInput
    else: # IF NOT EMPTY

        # GET STOCK NAMES
        stock_df = showStockNames()

        # CHECK TO MAKE SURE USER INPUT IS A VALID STOCK NAME
        if userInput not in stock_df.index:
            no_data = st.write('Not a valid stock')
            return no_data
        else:
            stock_info = yf.download(
                userInput,
                start = '2000-01-01',
                end = f'{date.today()}',
                progress = False
            )

            #DROP VOLUME COLUMN
            stock_info.drop('Volume', axis = 1, inplace = True)

            # CONVERT DATE TO HUMAN READABLE DATE
            stock_info.index = pd.to_datetime(stock_info.index).date

            # SORT INDEX (DATE) FROM CURRENT PRICE TO EARLIEST PRICE
            stock_info.sort_index(inplace=True, ascending=False)

            return stock_info
# =========================================================================================

# =========================================================================================
#                                       VISUALS: PRIOR TO PREDICTION
# =========================================================================================
# Create line plot on stock data to show trend
def line_chart(userInput):
     # IF USERINPUT IS EMPTY
    print("input is ", userInput)
    if not userInput:
        empty = st.write('No Data to show')
        return empty
    else: # IF NOT EMPTY
        stock_info = get_stock_data(userInput)

        visual_df = pd.DataFrame(stock_info, columns = ['Open', 'Adj Close'])

        return visual_df

 
st.markdown('---') # Add a page break
price_trend, graph_details = st.columns((1, 1))
with price_trend:
    st.line_chart(line_chart(userInput))


how_to_use_graph = '''
- X-Axis has all data information
- Y-Axis has the price (US Dollars (100)) 

---

#### Hover over Graph

- When you hover over the graph, two arrows indicating to expand will appear.
- Click on the two arrows to be able to have a wider view.
- You will then be able to move around and interact with different prices on different dates.
'''

disclaimer = '''
#### Real Time Data
- The data you see on the graph is up to date and real time.
- The prices shown for the a certain date is true and based on stored data from the stock market.

--

Do not get this mixed up with thinking that it is a price prediction graph!!

'''
with graph_details:
    with st.expander('GRAPH INSTRUCTIONS 📈'):
        st.markdown(how_to_use_graph)
    
    with st.expander('❗DISCLAIMER ❗'):
        st.markdown(disclaimer)
    


# =========================================================================================

# =========================================================================================
#                                           DATA PREPROCESS
# =========================================================================================
## PART 1:
# - ADJUST OUR DATAFRAME TO INCLUDE FORCAST DAYS FROM USER INPUT
# - ERROR HANDLING

def data_preprocess(forecastDays, userInput):

    if forecastDays < 0:
        st.write('[ERROR]: CHECK NUMBER INPUT')
    else:
        # GET STOCK NAMES
        stock_df = showStockNames()
        if userInput not in stock_df.index:
            no_data = st.write('Not a valid stock')
            return no_data
        else:
            stock_info = get_stock_data(userInput)
            # CREATE NEW STOCK DATAFRAME FOR STOCK PREDICTION PRICES
            stock_prediction = stock_info[['Adj Close']]

            # SHIFT NEW DATAFRAME X AMOUNT OF ROWS BASED ON FORECAST DAYS
            # CALL THIS STOCK PRICE
            stock_prediction['Stock Price'] = stock_prediction.loc[:, 'Adj Close'].shift(-forecastDays)

            return stock_prediction
            



# =========================================================================================
#                                           Data Preperation
# =========================================================================================
# - ADJUST OUR DATAFRAME TO INCLUDE FORCAST DAYS FROM USER INPUT
# - ERROR HANDLING
def data_prep(forecastDays, userInput):

    if forecastDays < 0 or forecastDays == 0:
        st.write('[ERROR]: CHECK NUMBER INPUT')
    else:
        # GET STOCK NAMES
        stock_df = showStockNames()
        if userInput not in stock_df.index:
            no_data = st.write('Not a valid stock')
            return no_data
        else:

            stock = data_preprocess(forecastDays, userInput)

            # CREATE X DATASET
            X_DATA = np.array(stock.drop(['Stock Price'], 1))
            X_DATA = X_DATA[:-forecastDays]

            # CREATE Y DATASET
            Y_DATA = np.array(stock['Stock Price'])
            Y_DATA = Y_DATA[:-forecastDays]

            # TEST SPLIT TRAIN DATA
            x_train, x_test, y_train, y_test = train_test_split(X_DATA, Y_DATA, test_size = 0.2)


            return x_train, x_test, y_train, y_test



# =========================================================================================
#                                           Data Model
# =========================================================================================
# CREATE OUR SVM MODEL FOR PRICE PREDICTION

def svm_model(forecastDays, userInput):
    if forecastDays < 0 or forecastDays == 0:
        st.write('[ERROR]: CHECK NUMBER INPUT')
    else:
        # GET STOCK NAMES
        stock_df = showStockNames()
        if userInput not in stock_df.index:
            no_data = st.write('Not a valid stock')
            return no_data
        else:
            stock = data_preprocess(forecastDays, userInput)
            x_train, x_test, y_train, y_test = data_prep(forecastDays, userInput)

            # INIT CLASSIFIER

            svr_clf = SVR(kernel='rbf', C = 1000.0, gamma = 0.0001)

            svr_clf_fit = svr_clf.fit(x_train, y_train)

            stock_price_pred = np.array(stock.drop(['Stock Price'], 1))[forecastDays:]

            svr_clf_pred = svr_clf_fit.predict(stock_price_pred)

            return svr_clf_pred[:forecastDays]


st.markdown('---') # Add a page break

# =========================================================================================
#                                           Show Stock Prediction
# =========================================================================================
# Create visuals for Stock price vs predicted price
# Show predicted price for number of days our


# def line_chart(userInput):
#      # IF USERINPUT IS EMPTY
#     if not userInput:
#         empty = st.write('No Data to show')
#         return empty
#     else: # IF NOT EMPTY
#         stock_info = get_stock_data(userInput)

#         visual_df = pd.DataFrame(stock_info)

#         return visual_df



def show_prices(forecastDays, userInput):
    if forecastDays < 0 or forecastDays == 0:
        st.write('[ERROR]: CHECK NUMBER INPUT')
    else:
        # GET STOCK NAMES
        stock_df = showStockNames()
        if userInput not in stock_df.index:
            no_data = st.write('Not a valid stock')
            return no_data
        else:
            pred = svm_model(forecastDays, userInput)

            col_vals = [f'Stock Price Prediction for {userInput}']

            pred_df = pd.DataFrame(
                data = pred,
                columns = col_vals
            )

            # pred_df = pred_df.apply(lambda x: round(x, 2))


            return pred_df


def pred_chart(userInput, forecastDays):
     # IF USERINPUT IS EMPTY
    print("input is ", userInput)
    if not userInput:
        empty = st.write('No Data to show')
        return empty
    else: # IF NOT EMPTY
        stock_info = show_prices(forecastDays, userInput)

        visual_df = pd.DataFrame(stock_info)

        return visual_df

 
st.markdown('---') # Add a page break
st.write(" plot for prediction ")
price_trend, graph_details = st.columns((1, 1))
with price_trend:
    st.line_chart(pred_chart(userInput, forecastDays))



stock_price_show, stock_dis = st.columns((1, 1))

with stock_price_show:
    if forecastDays < 0 or forecastDays == 0:
        st.write('[ERROR]: CHECK NUMBER INPUT')
    else:
        # GET STOCK NAMES
        stock_df = showStockNames()
        if userInput not in stock_df.index:
            st.write('Not a valid stock')
        else:
            STOCK_PRICE_DF = show_prices(forecastDays, userInput)
            s = [f'Stock Price Prediction for {userInput}']
            st.write(f'Showing Stock Prices up to: {date.today() + datetime. timedelta(days=forecastDays)}')
            for i in range(len(STOCK_PRICE_DF)):
                st.write(f'${round(STOCK_PRICE_DF.iloc[i, 0], 2)} --- [DAY: {i + 1}]')
                
            
    # st.write(show_prices(forecastDays, userInput))

WARNING_DNU = '''
PLEASE DO NOT USE FOR YOUR DECISIONS ON BUYING STOCK OR PREDICTING PRICES.

---

CONSULT A BROKER FOR PROFESSIONAL ADVICE.
'''
with stock_dis:
    with st.expander('❗NOT FOR PERSONAL USE ❗'):
        st.warning(WARNING_DNU)


# =========================================================================================
#                                           Footer
# =========================================================================================

footer="""<style>
a:link , a:visited{
color: white;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: black;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://twitter.com/iamAngelSH" target="_blank">Angel Santana Hernandez</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

# =========================================================================================
#                                       Testing
# =========================================================================================
# st.write(data_preprocess(forecastDays, userInput))
# st.write(data_prep(forecastDays, userInput))
# st.write(svm_model(forecastDays, userInput))
# st.write(show_prices(forecastDays, userInput))

