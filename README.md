
# Sistem Inteligent de Estimare a Chiriilor (Deep Learning)

**Student:** Gondrea Ion  
**Disciplina:** Calcul Neuronal  
**Tema:** Regresie pentru predicția prețurilor chiriilor

## 1. Descrierea Proiectului
Acest proiect utilizează o rețea neuronală artificială pentru a estima prețul chiriei unei proprietăți pe baza caracteristicilor sale. În contextul activității imobiliare la IMO Expert, un astfel de model oferă o bază matematică obiectivă pentru evaluarea proprietăților, ajutând la fundamentarea comisionului standard de 50% de la proprietar și 50% de la chiriaș.

## 2. Setul de Date
Am utilizat `House_Rent_Dataset.csv`, care include informații despre:
- Suprafață (Size)
- Număr de camere (BHK) și băi
- Oraș și nivel de mobilare (Furnishing Status)

## 3. Arhitectura Modelului
Modelul este o rețea de tip **Multi-Layer Perceptron (MLP)** construită cu TensorFlow/Keras:
- **Input Layer:** Preluarea datelor preprocesate (StandardScaler & OneHotEncoder).
- **Hidden Layers:** 3 straturi Dense (64, 32, 16 neuroni) cu funcție de activare ReLU.
- **Regularizare:** Strat Dropout (20%) pentru a preveni overfitting-ul.
- **Output Layer:** 1 neuron cu funcție de activare lineară pentru predicția prețului.

## 4. Rezultate
- **Metrică:** Mean Absolute Error (MAE) pe setul de test: ~12,726 INR.
- Modelul demonstrează capacitatea de a învăța corelațiile dintre suprafață/locație și prețul final.

## 5. Instrucțiuni de Rulare
1. Instalați dependințele: `pip install pandas scikit-learn tensorflow`
2. Rulați scriptul: `python proiect_calcul_neuronal.py`
