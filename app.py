# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Load your dataset and preprocess it
# Define your model and training as you have done

# Function to recommend places for a user
@st.cache(allow_output_mutation=True)
def recommend_places(user_id, num_recommendations=5):
    # Prepare data for the selected user
    user_encoder = user_to_user_encoded.get(user_id)
    place_df = place[['Place_Id', 'Place_Name', 'Category', 'Rating', 'Price']]
    place_df.columns = ['id', 'place_name', 'category', 'rating', 'price']
    place_visited_by_user = df[df.User_Id == user_id]
    place_not_visited = place_df[~place_df['id'].isin(place_visited_by_user.Place_Id.values)]['id']
    place_not_visited = list(set(place_not_visited).intersection(set(place_to_place_encoded.keys())))
    place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
    user_place_array = np.hstack(([[user_encoder]] * len(place_not_visited), place_not_visited))
    
    # Predict top N recommendations
    ratings = model.predict(user_place_array).flatten()
    top_ratings_indices = ratings.argsort()[-num_recommendations:][::-1]
    recommended_place_ids = [place_encoded_to_place.get(place_not_visited[x][0]) for x in top_ratings_indices]
    
    return recommended_place_ids

# Main Streamlit application
def main():
    st.title("Rekomendasi Tempat Wisata di Indonesia")

    # Sidebar to select user and number of recommendations
    user_id = st.selectbox("Pilih User ID", user['User_Id'].unique())
    num_recommendations = st.slider('Pilih jumlah rekomendasi', min_value=2, max_value=10, value=5)

    # Display top places visited by the user
    st.subheader(f"Top places visited by User {user_id}")
    top_place_user = place_visited_by_user.sort_values(by='Place_Ratings', ascending=False).head(5).Place_Id.values
    place_df_rows = place_df[place_df['id'].isin(top_place_user)]
    for row in place_df_rows.itertuples():
        st.write(f"{row.place_name} : {row.category}")

    # Display recommended places
    recommended_places = recommend_places(user_id, num_recommendations)
    st.subheader(f"Top {num_recommendations} recommended places")
    for i, place_id in enumerate(recommended_places, start=1):
        place_info = place_df[place_df['id'] == place_id].iloc[0]
        st.write(f"{i}. {place_info.place_name}\n   Category: {place_info.category}, Price: {place_info.price}, Rating: {place_info.rating}\n")

if __name__ == '__main__':
    main()
