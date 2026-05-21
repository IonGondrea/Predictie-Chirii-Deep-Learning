import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# --- PARTEA 1: Încărcarea și pregătirea datelor ---
# Citim fișierul CSV și ștergem 3 coloane care ne complică la început.
df = pd.read_csv('House_Rent_Dataset.csv')
df = df.drop(columns=['Posted On', 'Area Locality', 'Floor'])

# X = datele apartamentului (camere, suprafață etc.)
# y = prețul pe care vrem să îl ghicim
X = df.drop(columns=['Rent'])
y = df['Rent']

# --- PARTEA 2: Preprocesarea (Traducerea pentru calculator) ---
numeric_features = ['BHK', 'Size', 'Bathroom']
categorical_features = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# Aici aplicăm traducerea pe datele noastre
X_processed = preprocessor.fit_transform(X)

# Împărțim datele: 80% din ele merg la "învățat", 20% la "testat"
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# --- PARTEA 3: Crearea "Creierului" (Rețeaua Neuronală) ---
model = Sequential([
    Input(shape=(X_train.shape[1],)), # Aici am corectat avertismentul
    Dense(64, activation='relu'),     # Strat ascuns 1 (64 neuroni)
    Dropout(0.2),                     # Previne memorarea mecanică (overfitting)
    Dense(32, activation='relu'),     # Strat ascuns 2 (32 neuroni)
    Dense(16, activation='relu'),     # Strat ascuns 3 (16 neuroni)
    Dense(1, activation='linear')     # Ieșirea: 1 singur număr (PREȚUL)
])

# Spunem modelului cum să își corecteze greșelile
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# --- PARTEA 4: Antrenarea (Învățarea efectivă) ---
print("Începem antrenarea modelului...")
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# --- PARTEA 5: ---
# Testăm pe cele 20% din date pe care nu le-a văzut niciodată


from sklearn.metrics import r2_score, mean_absolute_percentage_error

test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"\nEroarea medie absolută pe setul de test (MAE): {test_mae:.2f} INR")

# 1. Generăm predicțiile pentru setul de test (avem nevoie de ele pentru calcule)
y_pred = model.predict(X_test)

# 2. Calculăm Scorul R2 ("Acuratețea" pentru regresie)
scor_r2 = r2_score(y_test, y_pred)
print(f"Acuratețea modelului (Scorul R2): {scor_r2 * 100:.2f}%")

# 3. Calculăm eroarea procentuală (MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"Eroarea medie procentuală: {mape * 100:.2f}%")


# --- PARTEA 6: DEMONSTRAȚIA ---
print("\n--- TESTARE APARTAMENT NOU ---")
# Inventăm datele unui apartament (ex: 2 camere, 1000 mp, 2 băi, situat în Mumbai, complet mobilat)
apartament_nou = pd.DataFrame([{
    'BHK': 4,
    'Size': 200,
    'Bathroom': 2,
    'Area Type': 'Super Area',
    'City': 'Mumbai',
    'Furnishing Status': 'Furnished',
    'Tenant Preferred': 'Bachelors/Family',
    'Point of Contact': 'Contact Owner'
}])

# Trecem apartamentul inventat prin același "traducător" de cuvinte în numere
date_procesate = preprocessor.transform(apartament_nou)

# Cerem rețelei să ne zică la ce preț s-ar închiria
pret_estimat = model.predict(date_procesate)
print(f"Prețul chiriei estimat de inteligența artificială este: {pret_estimat[0][0]:.2f} INR")
