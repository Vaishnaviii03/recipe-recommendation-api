from flask import Blueprint, jsonify, request
import pandas as pd

# Create a Blueprint for the API
api = Blueprint('api', __name__)

# Load the recipes data (ensure the path is correct)
recipes_df = pd.read_csv('data/processed/recipes_processed.csv')

@api.route('/recipes', methods=['GET'])
def get_recipes():
    """Get all recipes."""
    recipes = recipes_df.to_dict(orient='records')
    return jsonify(recipes)

@api.route('/recipes/<int:recipe_id>', methods=['GET'])
def get_recipe(recipe_id):
    """Get a specific recipe by ID."""
    recipe = recipes_df[recipes_df['id'] == recipe_id]
    if not recipe.empty:
        return jsonify(recipe.to_dict(orient='records')[0])
    return jsonify({"error": "Recipe not found"}), 404

@api.route('/recommend', methods=['POST'])
def recommend_recipe():
    """Recommend recipes based on provided ingredients."""
    data = request.get_json()
    ingredients = data.get("ingredients", [])

    # Filter recipes based on the ingredients (basic matching logic)
    recommended_recipes = recipes_df[recipes_df['ingredients'].apply(lambda x: all(ingredient in x for ingredient in ingredients))]
    
    if not recommended_recipes.empty:
        return jsonify(recommended_recipes.to_dict(orient='records'))
    return jsonify({"message": "No recipes found for the provided ingredients."}), 404

@api.route('/recipes', methods=['POST'])
def add_recipe():
    """Add a new recipe."""
    new_recipe = request.get_json()
    recipes_df.loc[len(recipes_df)] = new_recipe  # Append new recipe
    return jsonify(new_recipe), 201

@api.route('/recipes/<int:recipe_id>', methods=['PUT'])
def update_recipe(recipe_id):
    """Update an existing recipe."""
    recipe = recipes_df[recipes_df['id'] == recipe_id]
    if not recipe.empty:
        data = request.get_json()
        recipes_df.loc[recipes_df['id'] == recipe_id, data.keys()] = data.values()
        return jsonify(data)
    return jsonify({"error": "Recipe not found"}), 404

@api.route('/recipes/<int:recipe_id>', methods=['DELETE'])
def delete_recipe(recipe_id):
    """Delete a recipe."""
    global recipes_df
    recipes_df = recipes_df[recipes_df['id'] != recipe_id]
    return '', 204