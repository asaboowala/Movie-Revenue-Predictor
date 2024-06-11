import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
from openai import OpenAI
import os
import torch
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit app
st.title('Movie Gross Revenue Prediction')

st.header('Enter Movie Details')

# Input fields for movie details
movie_title = st.text_input('Movie Title')
budget = st.slider('Budget (in million $)', min_value=0.0, max_value=999.0, step=0.1)
director_name = st.text_input('Director Name')
actor_names = st.text_area('Actor Names (separate with commas)')
genre = st.selectbox('Genre', [
    'Action', 'Adventure', 'Fantasy', 'Sci-Fi', 'Thriller', 'Documentary',
    'Animation', 'Family', 'Western', 'Comedy', 'Drama', 'Romance', 'Horror',
    'Crime', 'Biography', 'Music', 'Mystery', 'History', 'Sport', 'War',
    'Musical', 'Short', 'News', 'Reality-TV'
])
rating = st.selectbox('Rating', ['PG-13', 'PG', 'G', 'R', 'NC-17', 'X', 'Not Rated'])
imdb_score = st.number_input('IMDb', min_value=0.0, max_value=10.0, step=0.1)

languages = [
    'English', 'Mandarin', 'Aboriginal', 'Spanish', 'French', 'Filipino',
    'Hindi', 'Maya', 'Kazakh', 'Telugu', 'Cantonese', 'Japanese', 'Aramaic',
    'Italian', 'Dutch', 'Dari', 'German', 'Hebrew', 'Mongolian', 'Russian',
    'Thai', 'Polish', 'Bosnian', 'Korean', 'Hungarian', 'Portuguese',
    'Icelandic', 'Danish', 'Chinese', 'Norwegian', 'Czech', 'Swedish',
    'None', 'Zulu', 'Dzongkha', 'Arabic', 'Vietnamese', 'Indonesian',
    'Romanian', 'Persian', 'Greek'
]
language = st.selectbox('Language', languages)

countries_list = [
    'USA', 'UK', 'New Zealand', 'Canada', 'Australia', 'Germany', 'China',
    'New Line', 'France', 'Japan', 'Spain', 'Hong Kong', 'Czech Republic',
    'India', 'Peru', 'South Korea', 'Aruba', 'Denmark', 'Mexico', 'Belgium',
    'Ireland', 'South Africa', 'Italy', 'Romania', 'Chile', 'Netherlands',
    'Hungary', 'Russia', 'Greece', 'Taiwan', 'Official site', 'Thailand',
    'Iran', 'Poland', 'West Germany', 'Georgia', 'Iceland', 'Brazil',
    'Finland', 'Norway', 'Sweden', 'Argentina', 'Colombia', 'Israel',
    'Indonesia', 'Afghanistan', 'Cameroon', 'Philippines'
]
country = st.selectbox('Country', sorted(countries_list))

def calculate_director_avg_gross(director_name, data):
    director_movies = data[data['director_name'] == director_name]
    if not director_movies.empty:
        return director_movies['gross'].mean()
    else:
        return 0

if st.button('Predict Revenue'):
    # Load and preprocess data
    unique_data = pd.read_csv('movie_metadata.csv').drop(['facenumber_in_poster', 'movie_imdb_link', 'aspect_ratio', 'plot_keywords', 'title_year'], axis=1)
    unique_data = unique_data.drop_duplicates(subset='movie_title', keep='first')
    unique_data = unique_data.drop('movie_title', axis=1)
    unique_data = unique_data.dropna(subset=['gross', 'budget'], how='any')
    unique_data = unique_data[~unique_data['content_rating'].str.contains("TV", na=False)]

    for column in unique_data.columns:
        if unique_data[column].dtype == 'object':
            unique_data[column].fillna("Unknown", inplace=True)
        else:
            unique_data[column].fillna(unique_data[column].median(), inplace=True)

    # Q1 = unique_data[['gross', 'budget']].quantile(0.25)
    # Q3 = unique_data[['gross', 'budget']].quantile(0.75)
    # IQR = Q3 - Q1
    # lower_bound = Q1 - 1.5 * IQR
    # upper_bound = Q3 + 1.5 * IQR
    # unique_data = unique_data[(unique_data['gross'] >= lower_bound['gross']) & (unique_data['gross'] <= upper_bound['gross']) &
    #                           (unique_data['budget'] >= lower_bound['budget']) & (unique_data['budget'] <= upper_bound['budget'])]

    unique_data['genres'] = unique_data['genres'].str.split('|')
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(unique_data['genres'])
    genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_, index=unique_data.index)
    unique_data = unique_data.drop('genres', axis=1).join(genres_df)

    actor_columns = ['actor_1_name', 'actor_2_name', 'actor_3_name']
    director_feature = 'director_name'
    actor_counts = unique_data[actor_columns].stack().value_counts()
    director_counts = unique_data[director_feature].value_counts()
    star_threshold = 0.1
    num_stars = int(len(actor_counts) * star_threshold)
    stars = set(actor_counts.head(num_stars).index)

    def count_stars(row, stars, actor_columns):
        return sum(row[actor] in stars for actor in actor_columns)

    unique_data['num_stars'] = unique_data[actor_columns].apply(count_stars, axis=1, stars=stars, actor_columns=actor_columns)
    unique_data.drop(columns=actor_columns, inplace=True)
    unique_data['director_star_power'] = unique_data['director_name'].map(director_counts)

    #unique_data.drop(columns=[director_feature], inplace=True)

    # Calculate the average gross revenue for each director and add it as a new column
    unique_data['director_avg_gross'] = unique_data['director_name'].apply(lambda x: calculate_director_avg_gross(x, unique_data))

    copy_data = unique_data.copy()

    categorical_features = ['color', 'language', 'country', 'content_rating']

    def convert_features_to_one_hot(df, feature_name_list):
        for feature_name in feature_name_list:
            if feature_name in df.columns:
                df = pd.get_dummies(df, columns=[feature_name])
        return df

    unique_data.drop(columns=[director_feature], inplace=True)

    data_encoded = convert_features_to_one_hot(unique_data, categorical_features)

    X = data_encoded.drop('gross', axis=1)
    y = data_encoded['gross']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    linear_model = LinearRegression().fit(X_train, y_train)
    ridge_model = Ridge(alpha=1.0).fit(X_train, y_train)
    lasso_model = Lasso(alpha=0.1).fit(X_train, y_train)

    def preprocess_input(input_data, feature_columns):
        input_df = pd.DataFrame([input_data])
        input_df = convert_features_to_one_hot(input_df, categorical_features)
        missing_cols = set(feature_columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[feature_columns]
        return input_df

    # Create a one-hot encoded genre dictionary
    genre_one_hot = {g: 0 for g in mlb.classes_}
    genre_one_hot[genre] = 1

    languages_one_hot = {f'language_{l}': 0 for l in languages}
    languages_one_hot[f'language_{language}'] = 1

    country_one_hot = {f'country_{c}': 0 for c in countries_list}
    country_one_hot[f'country_{country}'] = 1

    content_rating_one_hot = {f'content_rating_{r}': 0 for r in ['PG-13', 'PG', 'G', 'R', 'NC-17', 'X', 'Not Rated']}
    content_rating_one_hot[f'content_rating_{rating}'] = 1

    print(data_encoded.columns)

    input_data = {
        'color': 'Color',
        'num_critic_for_reviews': data_encoded['num_critic_for_reviews'].median(),
        'duration': data_encoded['duration'].median(),
        'director_facebook_likes': data_encoded['director_facebook_likes'].median(),
        'actor_3_facebook_likes': data_encoded['actor_3_facebook_likes'].median(),
        'actor_1_facebook_likes': data_encoded['actor_1_facebook_likes'].median(),
        'actor_2_facebook_likes': data_encoded['actor_2_facebook_likes'].median(),
        'movie_facebook_likes': data_encoded['movie_facebook_likes'].median(),
        'num_voted_users': data_encoded['num_voted_users'].median(),
        'num_user_for_reviews': data_encoded['num_user_for_reviews'].median(),
        'budget': budget,
        'imdb_score': imdb_score,
        'num_stars': count_stars({'actor_1_name': actor_names.split(',')[0].strip(),
                                  'actor_2_name': actor_names.split(',')[1].strip() if len(actor_names.split(',')) > 1 else actor_names.split(',')[0].strip(),
                                  'actor_3_name': actor_names.split(',')[2].strip() if len(actor_names.split(',')) > 2 else actor_names.split(',')[0].strip()},
                                  stars, actor_columns),
        'director_star_power': director_counts.get(director_name, 0),
        'director_avg_gross': calculate_director_avg_gross(director_name, copy_data)
    }

    input_data.update(genre_one_hot)
    input_data.update(languages_one_hot)
    input_data.update(country_one_hot)
    input_data.update(content_rating_one_hot)

    input_df = preprocess_input(input_data, X_train.columns)

    linear_pred = linear_model.predict(input_df)
    ridge_pred = ridge_model.predict(input_df)
    lasso_pred = lasso_model.predict(input_df)

    st.subheader('Predicted Gross Revenue:')
    st.write(f"Linear Regression Prediction: ${linear_pred[0]:,.2f}")
    st.write(f"Ridge Regression Prediction: ${ridge_pred[0]:,.2f}")
    #st.write(f"Lasso Regression Prediction: ${lasso_pred[0]:,.2f}")

    y_pred = linear_model.predict(X_test)
    y_pred_ridge = ridge_model.predict(X_test)
    y_pred_lasso = lasso_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Linear Regression - MSE: {mse}, MAE: {mae}, R-squared: {r2}, RMSE: {rmse}")

    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    rmse_ridge = np.sqrt(mse_ridge)

    print(f"Ridge Regression - MSE: {mse_ridge}, MAE: {mae_ridge}, R-squared: {r2_ridge}, RMSE: {rmse_ridge}")

    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
    r2_lasso = r2_score(y_test, y_pred_lasso)
    rmse_lasso = np.sqrt(mse_lasso)

    print(f"Lasso Regression - MSE: {mse_lasso}, MAE: {mae_lasso}, R-squared: {r2_lasso}, RMSE: {rmse_lasso}")

    def within_percentage_accuracy(outputs, targets, percentage=0.10):
        with torch.no_grad():
            # Convert inputs to torch tensors
            outputs = torch.tensor(outputs)
            targets = torch.tensor(targets.values)

            # Calculate the acceptable deviation
            deviation = percentage * targets
            lower_bounds = targets - deviation
            upper_bounds = targets + deviation

            # Check if predictions fall within the specified range
            correct = torch.logical_and(outputs >= lower_bounds, outputs <= upper_bounds)
            accuracy = torch.mean(correct.float())  # Calculate the mean of correct predictions
        return accuracy.item() * 100  # Convert to percentage

    accuracy_linear = within_percentage_accuracy(y_pred, y_test, percentage=0.2)
    accuracy_ridge = within_percentage_accuracy(y_pred_ridge, y_test, percentage=0.2)
    accuracy_lasso = within_percentage_accuracy(y_pred_lasso, y_test, percentage=0.2)

    print(f"Linear Regression - Accuracy within 20%: {accuracy_linear}%")
    print(f"Ridge Regression - Accuracy within 20%: {accuracy_ridge}%")
    print(f"Lasso Regression - Accuracy within 20%: {accuracy_lasso}%")


if st.button('Generate Movie Synopsis'):
    client = OpenAI()

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a film writer, skilled in writing movie stories with creative flair."},
        {"role": "user", "content": f"""Use your creativity to create a movie story description, you can use the following information about the movie (not all of this information needs to be explicitly restated again): \n
                    Movie Title: {movie_title} \n 
                    Budget (in millions): {budget} \n 
                    Director Name: {director_name} \n 
                    Actor Names: {actor_names} \n
                    Genre: {genre} \n
                    IMDb Score: {imdb_score} \n
                    Language: {language} \n
                    Country: {country}"""}
        ]
    )

    st.subheader('AI Generated Movie Synopsis:')
    st.write(completion.choices[0].message.content)




if st.button('Generate Movie Poster'):
    client = OpenAI()

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a film writer, skilled in writing movie stories with creative flair."},
        {"role": "user", "content": f"""Use your creativity to create a movie story description, you can use the following information about the movie (not all of this information needs to be explicitly restated again): \n
                    Movie Title: {movie_title} \n 
                    Budget (in millions): {budget} \n 
                    Director Name: {director_name} \n 
                    Actor Names: {actor_names} \n
                    Genre: {genre} \n
                    IMDb Score: {imdb_score} \n
                    Language: {language} \n
                    Country: {country}"""}
        ]
    )

    synopsis = completion.choices[0].message

    image_prompt = f"""Generate a movie poster given the following information about the movie:\n
                    Movie Title: {movie_title} \n 
                    Budget (in millions): {budget} \n 
                    Director Name: {director_name} \n 
                    Actor Names: {actor_names} \n
                    Genre: {genre} \n
                    IMDb Score: {imdb_score} \n
                    Language: {language} \n
                    Country: {country} \n 
                    Synopsis: {synopsis}"""


    response = client.images.generate(
        model="dall-e-3",
        prompt=image_prompt,
        size="1024x1792",
        quality="hd",
        n=1,
    )

    image_url = response.data[0].url

    st.subheader('AI Generated Image Link:')
    st.write(f"{image_url}")
