from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import ast

app = Flask(__name__, static_folder='D:\\Recipe Recommendation-api-main\\static', template_folder='D:\\Recipe Recommendation-api-main\\templates')

# Load vocabulary and DataFrame
with open('D:\\Recipe Recommendation-api-main\\models\\recipe_recommendation_model.pkl', 'rb') as f:
    vocab = pickle.load(f)

df = pd.read_csv('D:\\Recipe Recommendation-api-main\\data\\processed\\recipes_processed.csv')

# Convert the 'Ingredient_Vector' column from string to list (if it was saved as a string)
df['Ingredient_Vector'] = df['Ingredient_Vector'].apply(ast.literal_eval)

def vectorize_ingredients(ingredient_str, vocab):
    vector = [0] * len(vocab)
    ingredients = [ingredient.strip() for ingredient in re.split(r',|\s', ingredient_str) if ingredient]
    for ingredient in ingredients:
        if ingredient in vocab:
            vector[vocab.index(ingredient)] = 1
    return vector

def recommend_recipes(user_vector, df, top_n=5):
    recipe_vectors = np.array(df['Ingredient_Vector'].tolist())
    user_vector = vectorize_ingredients(user_vector, vocab)
    user_vector = np.array(user_vector).reshape(1, -1)
    similarity_scores = cosine_similarity(user_vector, recipe_vectors)[0]
    
    recipe_similarity = list(enumerate(similarity_scores))
    top_indices = sorted(recipe_similarity, key=lambda x: x[1], reverse=True)[:top_n]
    
    top_recipe_indices = [index for index, score in top_indices]
    
    return top_recipe_indices

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None  # Initialize results variable
    if request.method == 'POST':
        user_ingredients = request.form['ingredients']
        top_recipes = recommend_recipes(user_ingredients, df, top_n=5)
        
        results = []
        for recipe in top_recipes:
            name = df.loc[recipe, 'Title']
            ingredients = df.loc[recipe, 'Core_Ingredients']
            image_url = df.loc[recipe, 'Image Link']  # Use 'Image Link' column
            
            # Format ingredients with commas
            formatted_ingredients = ', '.join(ingredients.split())  # If ingredients are space-separated
            
            results.append({'name': name, 'ingredients': formatted_ingredients, 'image_url': image_url})

    return render_template('index.html', results=results, ingredients=user_ingredients if results else None)

if __name__ == '__main__':
    app.run(debug=True)
