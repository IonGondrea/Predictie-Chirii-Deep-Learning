import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_absolute_percentage_error


df = pd.read_csv('House_Rent_Dataset.csv')
df = df.drop(columns=['Posted On', 'Area Locality', 'Floor'])


X = df.drop(columns=['Rent'])
y = df['Rent']


numeric_features = ['BHK', 'Size', 'Bathroom']
categorical_features = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])


X_processed = preprocessor.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)


model_1 = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),     
    Dropout(0.2),                     
    Dense(32, activation='relu'),     
    Dense(16, activation='relu'),     
    Dense(1, activation='linear')     
])
model_1.compile(optimizer='adam', loss='mse', metrics=['mae'])


model_2 = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),    
    Dense(1, activation='linear')     
])
model_2.compile(optimizer='adam', loss='mse', metrics=['mae'])


early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

print("--- Începem antrenarea MODELULUI 1 (Complex) ---")
history_1 = model_1.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

print("\n--- Începem antrenarea MODELULUI 2 (Simplu) ---")
history_2 = model_2.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


print("\n" + "="*50)
print(" REZULTATE ȘI COMPARAȚIE MODELE")
print("="*50)


_, test_mae_1 = model_1.evaluate(X_test, y_test, verbose=0)
y_pred_1 = model_1.predict(X_test, verbose=0)
scor_r2_1 = r2_score(y_test, y_pred_1)
mape_1 = mean_absolute_percentage_error(y_test, y_pred_1)

_, test_mae_2 = model_2.evaluate(X_test, y_test, verbose=0)
y_pred_2 = model_2.predict(X_test, verbose=0)
scor_r2_2 = r2_score(y_test, y_pred_2)
mape_2 = mean_absolute_percentage_error(y_test, y_pred_2)


print("MODEL 1 (Complex):")
print(f" - Eroare medie absolută (MAE): {test_mae_1:.2f} INR")
print(f" - Acuratețe (Scorul R2):       {scor_r2_1 * 100:.2f}%")
print(f" - Eroare Procentuală (MAPE):   {mape_1 * 100:.2f}%\n")

print("MODEL 2 (Simplu):")
print(f" - Eroare medie absolută (MAE): {test_mae_2:.2f} INR")
print(f" - Acuratețe (Scorul R2):       {scor_r2_2 * 100:.2f}%")
print(f" - Eroare Procentuală (MAPE):   {mape_2 * 100:.2f}%")
print("="*50)


print("\n--- TESTARE APARTAMENT NOU ---")

apartament_nou = pd.DataFrame([{
    'BHK': 3,
    'Size': 400,
    'Bathroom': 4,
    'Area Type': 'Super Area',
    'City': 'Mumbai',
    'Furnishing Status': 'Furnished',
    'Tenant Preferred': 'Bachelors/Family',
    'Point of Contact': 'Contact Owner'
}])


date_procesate = preprocessor.transform(apartament_nou)


pret_estimat = model_1.predict(date_procesate, verbose=0)
print(f"Prețul chiriei estimat de Inteligența Artificială (Modelul 1) este: {pret_estimat[0][0]:.2f} INR")
