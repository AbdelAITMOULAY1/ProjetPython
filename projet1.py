import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error

# Chemin vers le fichier CSV
path = r"C:\Users\pret\.cache\kagglehub\datasets\kusumakar\gold-prices-for-5-years-financial-predictions\versions\1"
file_name = 'Gold  Prices.csv'.strip()
data_path = os.path.join(path, file_name)

# Charger les données
df = pd.read_csv(data_path)

# Convertir la colonne 'Date' en format datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)

# Supprimer les lignes avec des valeurs manquantes
df_clean = df.dropna(subset=['Date', 'Open'])

# Extraire l'année, le mois et le jour de la colonne 'Date'
df_clean['Year'] = df_clean['Date'].dt.year
df_clean['Month'] = df_clean['Date'].dt.month
df_clean['Day'] = df_clean['Date'].dt.day

# Définir les variables d'entrée et la cible
X = df_clean[['Year', 'Month', 'Day']]
y = df_clean['Open']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer le modèle d'arbre de décision avec une profondeur limitée
model = DecisionTreeRegressor(random_state=42, max_depth=5)
model.fit(X_train, y_train)

# Prédictions et évaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Visualiser l'arbre de décision
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=['Year', 'Month', 'Day'], fontsize=10)
plt.title("Arbre de décision (profondeur limitée)")
plt.show()

# Exemple de prédiction pour une nouvelle date
new_data = pd.DataFrame({
    'Year': [2024],
    'Month': [1],
    'Day': [1]
})

predicted_price = model.predict(new_data)
print(f"Prix prédit pour 2024-01-01 : {predicted_price[0]}")
