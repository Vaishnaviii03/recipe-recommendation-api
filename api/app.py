from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import ast

app = Flask(__name__)

# Load vocabulary and DataFrame
MODEL_PATH = 'D:\\Recipe Recommendation - Copy\\models\\recipe_recommendation_model.pkl'
DATA_PATH = 'D:\\Recipe Recommendation - Copy\\data\\processed\\recipes_processed.csv'

with open(MODEL_PATH, 'rb') as f:
    vocab = pickle.load(f)

df = pd.read_csv(DATA_PATH)
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

@app.route('/api/recommend', methods=['POST'])
def recommend():
    user_ingredients = request.json.get('ingredients')
    if not user_ingredients:
        return jsonify({'error': 'No ingredients provided.'}), 400
    
    top_recipes = recommend_recipes(user_ingredients, df, top_n=5)
    results = []
    for recipe in top_recipes:
        name = df.loc[recipe, 'Title']
        ingredients = df.loc[recipe, 'Core_Ingredients']
        image_url = df.loc[recipe, 'Image Link']
        formatted_ingredients = ', '.join(ingredients.split())
        
        results.append({'name': name, 'ingredients': formatted_ingredients, 'image_url': image_url})

    return jsonify({'recommended_recipes': results})

if __name__ == '__main__':
    app.run(debug=True)
