import requests

# Ask user for ingredients
user_input = input("Please enter the ingredients separated by commas: ")
url = 'http://127.0.0.1:5000/api/recommend'

# Send the POST request with the user's ingredients
response = requests.post(url, json={'ingredients': user_input})

if response.status_code == 200:
    recommendations = response.json().get('recommended_recipes', [])
    if recommendations:
        for recipe in recommendations:
            print(f"Name: {recipe['name']}")
            print(f"Ingredients: {recipe['ingredients']}")
            print(f"Image URL: {recipe['image_url']}\n")
    else:
        print("No recipes found based on the provided ingredients.")
else:
    print(f"Error: {response.status_code} - {response.text}")
