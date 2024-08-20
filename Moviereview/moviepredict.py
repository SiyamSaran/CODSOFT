import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#Load the dataset
data = pd.read_csv("E:\\IMDb Movies India.csv", encoding='ISO-8859-1')
data

# Data Cleaning and Preprocessing
data_cleaned = data.dropna(subset=['Rating'])  
data_cleaned['Votes'] = pd.to_numeric(data_cleaned['Votes'], errors='coerce')
data_cleaned = data_cleaned.dropna(subset=['Genre', 'Director', 'Actor 1'])
data_cleaned['Year'] = data_cleaned['Year'].str.extract(r'(\d{4})').astype(int)
data_cleaned['Votes'] = data_cleaned['Votes'].fillna(data_cleaned['Votes'].median())
data_cleaned['Duration'] = data_cleaned['Duration'].str.extract(r'(\d+)').astype(float)
data_cleaned['Duration'] = data_cleaned['Duration'].fillna(data_cleaned['Duration'].median())
data_cleaned = data_cleaned.drop(['Name'], axis=1)

data_encoded = pd.get_dummies(data_cleaned, columns=['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'])
data_encoded.head()

# Split the Data into Training and Testing Sets
X = data_encoded.drop('Rating', axis=1)  # Features (drop the target variable)
y = data_encoded['Rating']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
print(f'Mean Absolute Error (MAE): {mae}')

residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.title('Residual Plot')
plt.xlabel('Predicted Ratings')
plt.ylabel('Residuals')
plt.show()


def predict_movie_rating():
    # Get input from the user
    year = int(input("Enter the Year of the movie: "))
    votes = int(input("Enter the number of Votes: "))
    duration = float(input("Enter the Duration (in minutes): "))
    genre = input("Enter the Genre of the movie: ")
    director = input("Enter the Director's name: ")
    actor1 = input("Enter the name of Actor 1: ")
    actor2 = input("Enter the name of Actor 2 (optional, press Enter to skip): ")
    actor3 = input("Enter the name of Actor 3 (optional, press Enter to skip): ")

    # Create a DataFrame for the input movie details
    input_data = {
        'Year': [year],
        'Votes': [votes],
        'Duration': [duration],
        'Genre': [genre],
        'Director': [director],
        'Actor 1': [actor1],
        'Actor 2': [actor2] if actor2 else None,
        'Actor 3': [actor3] if actor3 else None
    } 

    input_df = pd.DataFrame(input_data)
    input_df['Votes'] = pd.to_numeric(input_df['Votes'], errors='coerce').fillna(data_cleaned['Votes'].median())
    input_df['Duration'] = pd.to_numeric(input_df['Duration'], errors='coerce').fillna(data_cleaned['Duration'].median())
    input_df['Year'] = input_df['Year'].astype(int)
    input_encoded = pd.get_dummies(input_df)
    
    input_encoded = input_encoded.reindex(columns=X_train.columns, fill_value=0)

    predicted_rating = model.predict(input_encoded)[0]

    return predicted_rating

predicted_rating = predict_movie_rating()
print(f'Predicted Rating: {predicted_rating}')
