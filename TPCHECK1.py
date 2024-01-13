import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_regression
st.set_option("deprecation.showPyplotGlobalUse", False)
# Votre code Streamlit ici

# Charger les données (remplacez 'votre_dataset.csv' par le nom de votre fichier CSV)
data = pd.read_csv("dataexpresso.csv")

# Options de filtrage
selected_columns = st.multiselect("Colonnes à afficher", data.columns)
filtered_df = data[selected_columns]

# Afficher le dataframe filtré
st.dataframe(filtered_df)

# Diviser les données en ensemble d'entraînement et de test
#extract x and y from our data
x=data[["REGULARITY","FREQ_TOP_PACK"]]
y=data["CHURN"].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=40)

# Entraîner le modèle de prédiction
model=LinearRegression()
poly=PolynomialFeatures(degree=1)
x_train_fit = poly.fit_transform(x_train) #transforming our input data, entrainement du modèle
model.fit(x_train_fit,y_train)  #fitting the training data

# Prédire les abonnés d'Expresso sur l'ensemble de test


x_test_ = poly.fit_transform(x_test)
y_pred = model.predict(x_test_)
# Afficher la précision du modèle
threshold = 0.5
y_pred_binary = np.where(y_pred >= threshold, 1, 0)
accuracy = accuracy_score(y_test, y_pred_binary)
st.subheader('Précision du modèle:')
st.write(f'{accuracy:.2f}')

#Interface utilisateur Streamlit
st.title('Prédiction des abonnés Expresso')

# Ajouter des champs pour les critères de sélection
st.sidebar.header('Critères de sélection')
FREQUENCE_RECH = st.sidebar.slider("Frequence de recharge du Client", min_value=0, max_value=100, value=50)
REGULARITY = st.sidebar.slider('Nombre de fois que le client a emis un appel/90', min_value=1, max_value=90, value=40)
FREQ_TOP_PACK = st.sidebar.slider("Frequence d'utilisation du top_pack par le Client", min_value=0, max_value=50, value=20)


# Faire la prédiction avec le modèle entraîné
prediction = model.predict([[REGULARITY,FREQ_TOP_PACK,FREQUENCE_RECH]])
# Afficher le résultat de la prédiction
st.subheader('Résultat de la prédiction:')
st.write(f'Prédiction d\'abonnés Expresso : {prediction[0]}')
