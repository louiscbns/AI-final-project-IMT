import json
import os
import time
from tqdm import tqdm
from projet import load_metadata
from mistralai import Mistral

def create_ingredient_description(ingredients):
    """
    Crée une description textuelle pour un plat à partir de ses ingrédients via l'API Mistral.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Mistral API key is not defined in .env file")
    
    client = Mistral(api_key=api_key)
    model = "mistral-large-latest"
    
    ingredients_text = "\n".join([
        f"- {ing['name']} ({ing['grams']}g) : {ing['calories']:.1f} kcal, {ing['fat']:.1f}g fat, {ing['carb']:.1f}g carbs, {ing['protein']:.1f}g protein"
        for ing in ingredients
    ])
    
    prompt = f"""As a nutritionist, analyze this dish using these ingredients and their nutritional information:

{ingredients_text}

Balanced meal criteria:
- Total calories: 500-800 kcal
- Protein: 20-30g (25-30% of calories)
- Carbs: 45-65g (45-55% of calories)
- Fat: 15-25g (25-35% of calories)

Generate an EXTREMELY short description that:
1. Compares the nutritional values with these balance criteria
2. Suggests ONE simple adjustment to improve balance
3. Avoids unnecessary details and repetitions

IMPORTANT: 
- Your response must be VERY short (maximum 50 tokens)
- Use short and direct sentences
- Avoid detailed explanations
- Mention only one adjustment at a time
- Don't repeat the same information"""
    
    chat_response = client.chat.complete(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert nutritionist analyzing dishes. You must provide VERY short and direct nutritional descriptions. Avoid unnecessary details and repetitions. IMPORTANT: Your response must be extremely short (maximum 50 tokens)."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return chat_response.choices[0].message.content

# Chemins
METADATA_PATH = 'food-nutrients/metadata.jsonl'
DESCRIPTIONS_PATH = 'descriptions.json'

# Charger toutes les données
print('Chargement des données...')
data = load_metadata(METADATA_PATH, num_samples=None)

# Charger les descriptions déjà existantes si le fichier existe
if os.path.exists(DESCRIPTIONS_PATH):
    with open(DESCRIPTIONS_PATH, 'r') as f:
        descriptions = json.load(f)
    print(f"{len(descriptions)} descriptions déjà présentes, reprise du script...")
else:
    descriptions = {}

# Générer les descriptions pour chaque plat manquant
print('Génération des descriptions pour chaque plat...')
nb_total = len(data)
nb_to_generate = 0
for item in tqdm(data):
    plat_id = item['id']
    if plat_id in descriptions and descriptions[plat_id]:
        continue  # Déjà généré
    try:
        desc = create_ingredient_description(item['ingredients'])
        time.sleep(1)  # Pause pour éviter le rate limit
    except Exception as e:
        print(f"Erreur pour le plat {plat_id} : {e}")
        desc = ""
    descriptions[plat_id] = desc
    nb_to_generate += 1
    # Sauvegarde intermédiaire toutes les 10 descriptions
    if nb_to_generate % 10 == 0:
        with open(DESCRIPTIONS_PATH, 'w') as f:
            json.dump(descriptions, f, ensure_ascii=False, indent=2)

# Sauvegarde finale
with open(DESCRIPTIONS_PATH, 'w') as f:
    json.dump(descriptions, f, ensure_ascii=False, indent=2)

print(f"Descriptions générées et sauvegardées dans {DESCRIPTIONS_PATH}")
print(f"Nombre total de descriptions : {len(descriptions)} / {nb_total}")

# Création d'un fichier enrichi avec les descriptions intégrées
OUTPUT_PATH = 'food-nutrients/metadata_with_descriptions.jsonl'
print(f"Création du fichier {OUTPUT_PATH} avec descriptions intégrées...")
with open(OUTPUT_PATH, 'w') as f_out:
    for item in data:
        plat_id = item['id']
        item_with_desc = item.copy()
        item_with_desc['description'] = descriptions.get(plat_id, "")
        f_out.write(json.dumps(item_with_desc, ensure_ascii=False) + '\n')
print(f"Fichier enrichi sauvegardé sous : {OUTPUT_PATH}")