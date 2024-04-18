
import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)



def main():

  #import modules
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  import warnings
  warnings.filterwarnings("ignore")

  df = pd.read_excel("sample data.xlsx")



  #Data info
  print("Dataset Info:")
  print(df.info())

  print("\nSummary Statistics:")
  print(df.describe())

  #Missing values
  print("Missing Values:")
  if df.isnull().sum().sum() == 0:
      print("No missing values.")
  else:
      print("There are missing values in the DataFrame.")

  #duplicates
  print("\nDuplicate Rows:")
  if df.duplicated().sum() == 0:
      print("No duplicate rows.")
  else:
      print("There are duplicate rows in the DataFrame.")

  #Outliers

  Q1 = df['bandwidth_gb_year'].quantile(0.25)
  Q3 = df['bandwidth_gb_year'].quantile(0.75)

  # Calculate the interquartile range (IQR)
  IQR = Q3 - Q1

  # Define the lower and upper bounds for outliers
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR

  # Identify outliers
  outliers = df[(df['bandwidth_gb_year'] < lower_bound) | (df['bandwidth_gb_year'] > upper_bound)]

  print("Lower Bound for Outliers:", lower_bound)
  print("Upper Bound for Outliers:", upper_bound)
  print("Number of Outliers (IQR):", outliers.shape[0])

  #Create KPI

  # Family Count
  def calculate_family_members(row):
      if row['marital'] == 'Married':
          return row['children'] + 2
      elif row['marital'] == 'Separated':
          return row['children'] + 1
      else:
          return row['children'] + 1

  df['family_members_count'] = df.apply(calculate_family_members, axis=1)

  # Device Count
  tablet_condition = (df['tablet'] == 'Yes').astype(int)
  phone_condition = (df['phone'] == 'Yes').astype(int)
  streaming_tv_condition = (df['streaming_tv'] == 'Yes').astype(int)

  df['devices'] = tablet_condition + phone_condition + streaming_tv_condition

  # Usage activity
  df['usage_activity'] = df['email'] + df['contacts']

  df.to_csv('updated_data.csv', index=False)
  print("CSV file saved successfully.")

  # Drop unnecessary columns from dataset
  columns_to_drop = ['customer_id','uid', 'interaction', 'lat', 'lng', 'job', 'children', 'age', 'marital', 'gender', 'churn',
                     'email', 'contacts', 'yearly_equip_failure', 'port_modem', 'tablet', 'phone', 'multiple',
                     'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
                     'streaming_movies', 'paperless_billing', 'payment_method']

  # Drop the specified columns from DataFrame 2
  df.drop(columns_to_drop, axis=1, inplace=True)

  target_variable = 'bandwidth_gb_year'

  # Get list of features (excluding the target variable)
  features = [col for col in df.columns if col != target_variable]

  # Set up the layout of subplots
  num_rows = (len(features) + 1) // 2
  fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5*num_rows))

  # Flatten axes for easier iteration
  axes = axes.flatten()

  # Plot scatter plots for each feature
  for i, feature in enumerate(features):
      ax = axes[i]
      ax.scatter(df[feature], df[target_variable], alpha=0.5)
      ax.set_xlabel(feature)
      ax.set_ylabel(target_variable)
      ax.set_title(f"{feature} vs. {target_variable}")

  # Adjust layout and display the plots
  plt.tight_layout()
  plt.show()

  # Plot graph between attributes

  # Categorical columns
  categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

  # Set up the layout of subplots
  num_rows = (len(categorical_columns) + 1) // 2
  fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5*num_rows))

  # Flatten axes for easier iteration
  axes = axes.flatten()

  # Plot bar plots for each categorical feature
  for i, column in enumerate(categorical_columns):
      ax = axes[i]
      sns.barplot(x=column, y='bandwidth_gb_year', data=df, ax=ax)
      ax.set_xlabel(column)
      ax.set_ylabel('bandwidth_gb_year')
      ax.set_title(f"{column} vs. bandwidth_gb_year")
      ax.tick_params(axis='x', rotation=45)

  # Remove any empty subplot
  for i in range(len(categorical_columns), num_rows*2):
      fig.delaxes(axes[i])

  # Adjust layout and display the plots
  plt.tight_layout()
  plt.show()

  # Heat map for numerical columns
  # Select numerical columns for correlation analysis
  numerical_columns = df.select_dtypes(exclude=['object']).columns.tolist()


  # Compute the correlation matrix
  correlation_matrix = df[numerical_columns].corr()

  # Plot heatmap
  plt.figure(figsize=(12, 10))
  sns.heatmap(correlation_matrix, annot=True, cmap='summer', fmt=".2f")
  plt.title('Correlation Heatmap of Numerical Variables')
  plt.show()

  selected_columns = ['city', 'state', 'population', 'area', 'internet_service',
                      'outage_sec_perweek', 'tenure', 'monthly_charge',
                      'day_time', 'year', 'family_members_count', 'devices', 'usage_activity', 'bandwidth_gb_year']

  data = df[selected_columns]
  y = data['bandwidth_gb_year']
  print("Selected columns and their data types:")
  for column in data.columns:
      print(f"{column}: {data[column].dtype}")

  # Plot histogram
  plt.figure(figsize=(10, 6))
  plt.hist(y, bins=20, color='skyblue', edgecolor='black')
  plt.title('Histogram of Bandwidth Usage (bandwidth_gb_year)')
  plt.xlabel('Bandwidth (GB)')
  plt.ylabel('Frequency')
  plt.grid(True)
  plt.show()

  # Separate columns into categorical and numerical features based on their data types
  categorical_features = data.select_dtypes(include=['object']).columns.tolist()
  numerical_features = data.select_dtypes(exclude=['object']).columns.tolist()

  print(categorical_features)
  print(numerical_features)

  df_cat = pd.get_dummies(df[categorical_features])

  # Concatenate numerical and encoded categorical features into X
  X = pd.concat([df[numerical_features], df_cat], axis=1)

  # Drop the 'Bandwidth_GB_Year' column from X
  X = X.drop(columns=['bandwidth_gb_year'])

  # Assign the target variable 'Bandwidth_GB_Year' to y
  y = df['bandwidth_gb_year']

  X_int = X.astype(int)

  # Plot histograms for each feature in X
  for column in X_int.columns:
      plt.figure(figsize=(8, 6))
      plt.hist(X_int[column], bins=20, color='skyblue', edgecolor='black')
      plt.title(f'Histogram of {column}')
      plt.xlabel(column)
      plt.ylabel('Frequency')
      plt.grid(True)
      plt.show()

  import numpy as np
  import matplotlib.pyplot as plt
  from scipy.stats import gaussian_kde

  # Assuming 'y' is your target variable
  # Plot histogram for the target variable y
  plt.figure(figsize=(8, 6))
  plt.hist(y, bins=20, density=True, color='salmon', edgecolor='black', alpha=0.7)  # Set density=True to normalize the histogram
  plt.title('Histogram of Target Variable (bandwidth_gb_year)')
  plt.xlabel('bandwidth_gb_year')
  plt.ylabel('Density')  # Update ylabel to indicate density
  plt.grid(True)

  # Overlay a line plot representing the density estimation
  density = gaussian_kde(y)
  x_vals = np.linspace(min(y), max(y), 100)
  plt.plot(x_vals, density(x_vals), color='blue', linestyle='-', linewidth=2)

  plt.show()

  correlation_matrix = X.corr()

  # Create a heatmap
  plt.figure(figsize=(12, 8))
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
  plt.title('Correlation Heatmap')
  plt.show()

  import pandas as pd
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.linear_model import LinearRegression
  from xgboost import XGBRegressor
  from sklearn.preprocessing import LabelEncoder,StandardScaler
  from sklearn.metrics import r2_score
  from sklearn.model_selection import train_test_split

  # Model training

  X = data.drop(columns=['bandwidth_gb_year'])
  y = data['bandwidth_gb_year']

  # print(len(X))
  # print(len(y))
  label_encoder_dict = {}

  for column in X.columns:
      if X[column].dtype == 'object':
          label_encoder = LabelEncoder()
          X[column] = label_encoder.fit_transform(X[column])
          label_encoder_dict[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))


  x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=0)

  scaler = StandardScaler()

  # Fit and transform scaler on training data
  x_train_scaled = scaler.fit_transform(x_train)

  # Transform testing data using the fitted scaler
  x_test_scaled = scaler.transform(x_test)

  x_train = pd.DataFrame(x_train_scaled, columns=x_train.columns)
  x_test = pd.DataFrame(x_test_scaled, columns=x_test.columns)


  models = {
      'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
      'LinearRegression': LinearRegression(),
      'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42)
  }

  for model_name, model in models.items():
      model.fit(x_train, y_train)

  for model_name, model in models.items():
      predicted_bandwidth = model.predict(x_test)
      print(f"\nPredicted Bandwidth Usage using {model_name}:")
      print(predicted_bandwidth)
      r2 = r2_score(y_test, predicted_bandwidth)
      print(f"R2 score for {model_name}: {r2}")

  # Input Data
  input_data = st.file_uploader("Upload CSV file", type=["csv"])
  if input_data is not None:
        # Read the uploaded CSV file
        df = pd.read_csv(input_data)

        # categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        # numerical_features = df.select_dtypes(exclude=['object']).columns.tolist()

        
        # df_cat = pd.get_dummies(df[categorical_features])
        # df = pd.concat([df[numerical_features], df_cat], axis=1)
        df = df[['city', 'state', 'population', 'area', 'internet_service','outage_sec_perweek', 'tenure', 'monthly_charge','day_time', 'year', 'family_members_count', 'devices', 'usage_activity']]  
        for column in df.columns:
            if column in label_encoder_dict:
                df[column] = df[column].map(label_encoder_dict[column])
            else:
                print(f"Warning: Column {column} in input data is not categorical.")   
        # Precition
        print(df)
        bandwidth_form=st.form('bandwidth_form')
        ls=['LinearRegression','RandomForest','XGBoost']
        model_name = bandwidth_form.selectbox('enter the model',ls)
        submit=bandwidth_form.form_submit_button(f'predict')
        if submit:
            model = models[model_name]
            predicted_bandwidth1 = model.predict(df)
        
            st.write(predicted_bandwidth1)   
            



if __name__ == "__main__":
  main()

    

    

    
