import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Waste water predictor",
    page_icon="🌊", 
    layout="wide", 
)
st.title("**Predictor**")

# Read the Excel file for BOD data
df_bod = pd.read_excel('mix.xlsx')  # Assuming you have a file named 'mix.xlsx' with BOD data

# Convert Month to numeric
month_mapping = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}
df_bod['Month'] = df_bod['Month'].str.lower().map(month_mapping)

selected_option = st.selectbox("Select option", ["Data Table", "Graph"])

if selected_option == "Data Table":
    selected_month = st.selectbox("Select a Month:", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    if selected_month:
        # Convert month number to month name
        month_name = list(month_mapping.keys())[list(month_mapping.values()).index(month_mapping[selected_month])]
        df_bod_month = df_bod[df_bod['Month'] == month_mapping[selected_month]]
        df_bod_month['Month'] = month_name
        st.write(df_bod_month)  # Display the data table for the selected month

else:
    user_put = st.selectbox("Select a Parameter:", ['BOD', 'pH', 'Fat oil and grease', 'Settleable solids', 'Sulfide', 'TKN', 'Total dissolved solids', 'Total suspended solids'])

    if user_put == 'BOD':
        # Train-test split for BOD data
        X_bod = df_bod[['Month']]
        y_bod = df_bod['BODaverage']
        X_train_bod, X_test_bod, y_train_bod, y_test_bod = train_test_split(X_bod, y_bod, test_size=0.2, random_state=42)

        # Create Linear Regression model for BOD data
        model_bod = LinearRegression()

        # Train the model for BOD data
        model_bod.fit(X_train_bod, y_train_bod)

        # Predict for BOD data
        y_pred_bod = model_bod.predict(X_test_bod)

        # Calculate Mean Squared Error for BOD data
        mse_bod = mean_squared_error(y_test_bod, y_pred_bod)

        # Display the MSE for BOD data
        st.write(f'Mean Squared Error for BOD: {mse_bod}')

        # User input for prediction for BOD data
        selected_month_bod = st.selectbox("Select a Month for BOD:", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])

        if selected_month_bod:
            # Make prediction for BOD data
            next_month = month_mapping[selected_month_bod] + 1 if month_mapping[selected_month_bod] != 12 else 1
            predicted_value_bod = model_bod.predict([[next_month]])

            # Display the predicted value for BOD data
            st.write(f'Predicted BOD value for next month: {predicted_value_bod[0]}')
        
            # Plot scatter plot for BOD data
            fig, ax = plt.subplots()
            ax.scatter(X_test_bod, y_test_bod, color='blue', label='Actual')
            ax.scatter(next_month, predicted_value_bod[0], color='red', label='Predicted')
            ax.set_xlabel('Month')
            ax.set_ylabel('BODaverage')
            ax.grid(True)
            ax.legend()

            # Set x-axis labels to month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(month_names)

            st.pyplot(fig)

    elif user_put == 'pH':
        # Read the Excel file for pH data
        df_ph = pd.read_excel('mix.xlsx')  # Assuming you have a file named 'mix.xlsx' with pH data
        df_ph['Month'] = df_ph['Month'].str.lower().map(month_mapping)
        
        # Train-test split for pH data
        X_ph = df_ph[['Month']]
        y_ph = df_ph['pHaverage']
        X_train_ph, X_test_ph, y_train_ph, y_test_ph = train_test_split(X_ph, y_ph, test_size=0.2, random_state=42)

        # Create Linear Regression model for pH data
        model_ph = LinearRegression()

        # Train the model for pH data
        model_ph.fit(X_train_ph, y_train_ph)

        # Predict for pH data
        y_pred_ph = model_ph.predict(X_test_ph)

        # Calculate Mean Squared Error for pH data
        mse_ph = mean_squared_error(y_test_ph, y_pred_ph)

        # Display the MSE for pH data
        st.write(f'Mean Squared Error for pH: {mse_ph}')

        # User input for prediction for pH data
        selected_month_ph = st.selectbox("Select a Month for pH:", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])

        if selected_month_ph:
            # Make prediction for pH data
            next_month = month_mapping[selected_month_ph] + 1 if month_mapping[selected_month_ph] != 12 else 1
            predicted_value_ph = model_ph.predict([[next_month]])

            # Display the predicted value for pH data
            st.write(f'Predicted pH value for next month: {predicted_value_ph[0]}')

            # Plot scatter plot for pH data
            fig, ax = plt.subplots()
            ax.scatter(X_test_ph, y_test_ph, color='blue', label='Actual')
            ax.scatter(next_month, predicted_value_ph[0], color='red', label='Predicted')
            ax.set_xlabel('Month')
            ax.set_ylabel('pHaverage')
            ax.grid(True)
            ax.legend()

            # Set x-axis labels to month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(month_names)

            st.pyplot(fig)

    elif user_put == 'Fat oil and grease':
        # Read the Excel file for Fat oil and grease data
        df_fog = pd.read_excel('mix.xlsx')  # Assuming you have a file named 'mix.xlsx' with Fat oil and grease data
        df_fog['Month'] = df_fog['Month'].str.lower().map(month_mapping)
        
        # Train-test split for Fat oil and grease data
        X_fog = df_fog[['Month']]
        y_fog = df_fog['Fat oil and grease']
        X_train_fog, X_test_fog, y_train_fog, y_test_fog = train_test_split(X_fog, y_fog, test_size=0.2, random_state=42)

        # Create Linear Regression model for Fat oil and grease data
        model_fog = LinearRegression()

        # Train the model for Fat oil and grease data
        model_fog.fit(X_train_fog, y_train_fog)

        # Predict for Fat oil and grease data
        y_pred_fog = model_fog.predict(X_test_fog)

        # Calculate Mean Squared Error for Fat oil and grease data
        mse_fog = mean_squared_error(y_test_fog, y_pred_fog)

        # Display the MSE for Fat oil and grease data
        st.write(f'Mean Squared Error for Fat oil and grease: {mse_fog}')

        # User input for prediction for Fat oil and grease data
        selected_month_fog = st.selectbox("Select a Month for Fat oil and grease:", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])

        if selected_month_fog:
            # Make prediction for Fat oil and grease data
            next_month = month_mapping[selected_month_fog] + 1 if month_mapping[selected_month_fog] != 12 else 1
            predicted_value_fog = model_fog.predict([[next_month]])

            # Display the predicted value for Fat oil and grease data
            st.write(f'Predicted Fat oil and grease value for next month: {predicted_value_fog[0]}')

            # Plot scatter plot for Fat oil and grease data
            fig, ax = plt.subplots()
            ax.scatter(X_test_fog, y_test_fog, color='blue', label='Actual')
            ax.scatter(next_month, predicted_value_fog[0], color='red', label='Predicted')
            ax.set_xlabel('Month')
            ax.set_ylabel('Fat oil and grease')
            ax.grid(True)
            ax.legend()

            # Set x-axis labels to month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(month_names)

            st.pyplot(fig)

    elif user_put == 'Settleable solids':
        # Read the Excel file for Settleable solids data
        df_ss = pd.read_excel('mix.xlsx')  # Assuming you have a file named 'mix.xlsx' with Settleable solids data
        df_ss['Month'] = df_ss['Month'].str.lower().map(month_mapping)
        
        # Train-test split for Settleable solids data
        X_ss = df_ss[['Month']]
        y_ss = df_ss['Settleable solids']
        X_train_ss, X_test_ss, y_train_ss, y_test_ss = train_test_split(X_ss, y_ss, test_size=0.2, random_state=42)

        # Create Linear Regression model for Settleable solids data
        model_ss = LinearRegression()

        # Train the model for Settleable solids data
        model_ss.fit(X_train_ss, y_train_ss)

        # Predict for Settleable solids data
        y_pred_ss = model_ss.predict(X_test_ss)

        # Calculate Mean Squared Error for Settleable solids data
        mse_ss = mean_squared_error(y_test_ss, y_pred_ss)

        # Display the MSE for Settleable solids data
        st.write(f'Mean Squared Error for Settleable solids: {mse_ss}')

        # User input for prediction for Settleable solids data
        selected_month_ss = st.selectbox("Select a Month for Settleable solids:", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])

        if selected_month_ss:
            # Make prediction for Settleable solids data
            next_month = month_mapping[selected_month_ss] + 1 if month_mapping[selected_month_ss] != 12 else 1
            predicted_value_ss = model_ss.predict([[next_month]])

            # Display the predicted value for Settleable solids data
            st.write(f'Predicted Settleable solids value for next month: {predicted_value_ss[0]}')

            # Plot scatter plot for Settleable solids data
            fig, ax = plt.subplots()
            ax.scatter(X_test_ss, y_test_ss, color='blue', label='Actual')
            ax.scatter(next_month, predicted_value_ss[0], color='red', label='Predicted')
            ax.set_xlabel('Month')
            ax.set_ylabel('Settleable solids')
            ax.grid(True)
            ax.legend()

            # Set x-axis labels to month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(month_names)

            st.pyplot(fig)

    elif user_put == 'Sulfide':
        # Read the Excel file for Sulfide data
        df_sulfide = pd.read_excel('mix.xlsx')  # Assuming you have a file named 'mix.xlsx' with Sulfide data
        df_sulfide['Month'] = df_sulfide['Month'].str.lower().map(month_mapping)
        
        # Train-test split for Sulfide data
        X_sulfide = df_sulfide[['Month']]
        y_sulfide = df_sulfide['Sulfide']
        X_train_sulfide, X_test_sulfide, y_train_sulfide, y_test_sulfide = train_test_split(X_sulfide, y_sulfide, test_size=0.2, random_state=42)

        # Create Linear Regression model for Sulfide data
        model_sulfide = LinearRegression()

        # Train the model for Sulfide data
        model_sulfide.fit(X_train_sulfide, y_train_sulfide)

        # Predict for Sulfide data
        y_pred_sulfide = model_sulfide.predict(X_test_sulfide)

        # Calculate Mean Squared Error for Sulfide data
        mse_sulfide = mean_squared_error(y_test_sulfide, y_pred_sulfide)

        # Display the MSE for Sulfide data
        st.write(f'Mean Squared Error for Sulfide: {mse_sulfide}')

        # User input for prediction for Sulfide data
        selected_month_sulfide = st.selectbox("Select a Month for Sulfide:", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])

        if selected_month_sulfide:
            # Make prediction for Sulfide data
            next_month = month_mapping[selected_month_sulfide] + 1 if month_mapping[selected_month_sulfide] != 12 else 1
            predicted_value_sulfide = model_sulfide.predict([[next_month]])

            # Display the predicted value for Sulfide data
            st.write(f'Predicted Sulfide value for next month: {predicted_value_sulfide[0]}')

            # Plot scatter plot for Sulfide data
            fig, ax = plt.subplots()
            ax.scatter(X_test_sulfide, y_test_sulfide, color='blue', label='Actual')
            ax.scatter(next_month, predicted_value_sulfide[0], color='red', label='Predicted')
            ax.set_xlabel('Month')
            ax.set_ylabel('Sulfide')
            ax.grid(True)
            ax.legend()

            # Set x-axis labels to month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(month_names)

            st.pyplot(fig)

    elif user_put == 'TKN':
        # Read the Excel file for TKN data
        df_tkn = pd.read_excel('mix.xlsx')  # Assuming you have a file named 'mix.xlsx' with TKN data
        df_tkn['Month'] = df_tkn['Month'].str.lower().map(month_mapping)
        
        # Train-test split for TKN data
        X_tkn = df_tkn[['Month']]
        y_tkn = df_tkn['TKN']
        X_train_tkn, X_test_tkn, y_train_tkn, y_test_tkn = train_test_split(X_tkn, y_tkn, test_size=0.2, random_state=42)

        # Create Linear Regression model for TKN data
        model_tkn = LinearRegression()

        # Train the model for TKN data
        model_tkn.fit(X_train_tkn, y_train_tkn)

        # Predict for TKN data
        y_pred_tkn = model_tkn.predict(X_test_tkn)

        # Calculate Mean Squared Error for TKN data
        mse_tkn = mean_squared_error(y_test_tkn, y_pred_tkn)

        # Display the MSE for TKN data
        st.write(f'Mean Squared Error for TKN: {mse_tkn}')

        # User input for prediction for TKN data
        selected_month_tkn = st.selectbox("Select a Month for TKN:", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])

        if selected_month_tkn:
            # Make prediction for TKN data
            next_month = month_mapping[selected_month_tkn] + 1 if month_mapping[selected_month_tkn] != 12 else 1
            predicted_value_tkn = model_tkn.predict([[next_month]])

            # Display the predicted value for TKN data
            st.write(f'Predicted TKN value for next month: {predicted_value_tkn[0]}')

            # Plot scatter plot for TKN data
            fig, ax = plt.subplots()
            ax.scatter(X_test_tkn, y_test_tkn, color='blue', label='Actual')
            ax.scatter(next_month, predicted_value_tkn[0], color='red', label='Predicted')
            ax.set_xlabel('Month')
            ax.set_ylabel('TKN')
            ax.grid(True)
            ax.legend()

            # Set x-axis labels to month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(month_names)

            st.pyplot(fig)
    elif user_put == 'Total dissolved solids':
        # Read the Excel file for Total dissolved solids data
        df_tds = pd.read_excel('mix.xlsx')  # Assuming you have a file named 'mix.xlsx' with Total dissolved solids data
        df_tds['Month'] = df_tds['Month'].str.lower().map(month_mapping)
        
        # Train-test split for Total dissolved solids data
        X_tds = df_tds[['Month']]
        y_tds = df_tds['Total dissolved solids']
        X_train_tds, X_test_tds, y_train_tds, y_test_tds = train_test_split(X_tds, y_tds, test_size=0.2, random_state=42)

        # Create Linear Regression model for Total dissolved solids data
        model_tds = LinearRegression()

        # Train the model for Total dissolved solids data
        model_tds.fit(X_train_tds, y_train_tds)

        # Predict for Total dissolved solids data
        y_pred_tds = model_tds.predict(X_test_tds)

        # Calculate Mean Squared Error for Total dissolved solids data
        mse_tds = mean_squared_error(y_test_tds, y_pred_tds)

        # Display the MSE for Total dissolved solids data
        st.write(f'Mean Squared Error for Total dissolved solids: {mse_tds}')

        # User input for prediction for Total dissolved solids data
        selected_month_tds = st.selectbox("Select a Month for Total dissolved solids:", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])

        if selected_month_tds:
            # Make prediction for Total dissolved solids data
            next_month = month_mapping[selected_month_tds] + 1 if month_mapping[selected_month_tds] != 12 else 1
            predicted_value_tds = model_tds.predict([[next_month]])

            # Display the predicted value for Total dissolved solids data
            st.write(f'Predicted Total dissolved solids value for next month: {predicted_value_tds[0]}')

            # Plot scatter plot for Total dissolved solids data
            fig, ax = plt.subplots()
            ax.scatter(X_test_tds, y_test_tds, color='blue', label='Actual')
            ax.scatter(next_month, predicted_value_tds[0], color='red', label='Predicted')
            ax.set_xlabel('Month')
            ax.set_ylabel('Total dissolved solids')
            ax.grid(True)
            ax.legend()

            # Set x-axis labels to month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(month_names)

            st.pyplot(fig)

    elif user_put == 'Total suspended solids':
        # Read the Excel file for Total suspended solids data
        df_tss = pd.read_excel('mix.xlsx')  # Assuming you have a file named 'mix.xlsx' with Total suspended solids data
        df_tss['Month'] = df_tss['Month'].str.lower().map(month_mapping)
        
        # Train-test split for Total suspended solids data
        X_tss = df_tss[['Month']]
        y_tss = df_tss['Total suspended solids']
        X_train_tss, X_test_tss, y_train_tss, y_test_tss = train_test_split(X_tss, y_tss, test_size=0.2, random_state=42)

        # Create Linear Regression model for Total suspended solids data
        model_tss = LinearRegression()

        # Train the model for Total suspended solids data
        model_tss.fit(X_train_tss, y_train_tss)

        # Predict for Total suspended solids data
        y_pred_tss = model_tss.predict(X_test_tss)

        # Calculate Mean Squared Error for Total suspended solids data
        mse_tss = mean_squared_error(y_test_tss, y_pred_tss)

        # Display the MSE for Total suspended solids data
        st.write(f'Mean Squared Error for Total suspended solids: {mse_tss}')

        # User input for prediction for Total suspended solids data
        selected_month_tss = st.selectbox("Select a Month for Total suspended solids:", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])

        if selected_month_tss:
            # Make prediction for Total suspended solids data
            next_month = month_mapping[selected_month_tss] + 1 if month_mapping[selected_month_tss] != 12 else 1
            predicted_value_tss = model_tss.predict([[next_month]])

            # Display the predicted value for Total suspended solids data
            st.write(f'Predicted Total suspended solids value for next month: {predicted_value_tss[0]}')

            # Plot scatter plot for Total suspended solids data
            fig, ax = plt.subplots()
            ax.scatter(X_test_tss, y_test_tss, color='blue', label='Actual')
            ax.scatter(next_month, predicted_value_tss[0], color='red', label='Predicted')
            ax.set_xlabel('Month')
            ax.set_ylabel('Total suspended solids')
            ax.grid(True)
            ax.legend()

            # Set x-axis labels to month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticks(range(1, 13))
            ax.set_xticklabels(month_names)

            st.pyplot(fig)
