import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.title("PENAMBANGAN DATA")
st.write("Nama  : Nabila Atira Qurratul Aini ")
st.write("Nim   : 210411100066 ")
st.write("Kelas : Penambangan Data B ")

data_set_description, upload_data, preprocessing, modeling, implementation = st.tabs(["Data Set Description", "Upload Data", "Preprocessing", "Modeling", "Implementation"])

with data_set_description:
    st.write("""# Data Set Description """)
    st.write("Data Set Ini Adalah : Harumanis Mango Physical Measurements (Pengukuran Fisik Mangga Harumanis) Dataset ini berisi 67 data pengukuran fisik tabular Mangga Harumanis.")
    st.write("Mangga atau mempelam adalah nama sejenis buah, demikian pula nama pohonnya. Mangga termasuk ke dalam genus Mangifera, yang terdiri dari 35-40 anggota dari famili Anacardiaceae.mangga arumanis (Mangifera indica L.) karena rasanya manis dan harum (arum) baunya. Daging buah tebal, lunak berwarna kuning, dan tidak berserat (serat sedikit). Aroma harum, tak begitu berair. Rasanya manis, tapi bagian ujung kadang kadang masih ada rasa asam.")
    st.write("# Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/mohdnazuan/harumanis-mango-physical-measurement")
    st.write("""# Penjelasan setiap kolom : """)
    st.write("""1.Weight(Berat):Berat mangga dalam gram (g)""")
    st.write("""2.Length (Panjang):Panjang buah mangga dalam centimeter (cm). Sebuah mangga arumanis panjangnya sekitar 15 cm dengan berat per buah 450 gram (rata-rata).  """)
    st.write("""3.Circumference (Lingkar): Lingkar mangga dalam sentimeter (cm)""")
    st.write("# Aplikasi ini untuk : Pengukuran Fisik Mangga Harumanis")
    st.write("# Source Code Aplikasi ada di Github anda bisa acces di link :  https://github.com/NabilaAtiraQurratulAini/coba-coba/tree/main")

with upload_data:
    df = pd.read_csv('https://raw.githubusercontent.com/NabilaAtiraQurratulAini/coba-coba/main/data-pendat.csv')
    st.dataframe(df)

with preprocessing:
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    df = df.drop(columns=["No"])
    
    #Mendefinisikan Varible X dan Y
    X = df.drop(columns=['Grade'])
    y = df['Grade'].values
    df
    X
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.Grade).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '0' : [dumies[0]],
        '1' : [dumies[1]],
    })

    st.write(labels)

with modeling:
    training, test = train_test_split(scaled_features, test_size=0.2, random_state=1)  # Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)  # Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        annbp = st.checkbox('Artificial Neural Network with Backpropagation')
        submitted = st.form_submit_button("Submit")

        # Gaussian Naive Bayes
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)
        y_pred = gaussian.predict(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))

        # K-NN
        K = 10
        knn = KNeighborsClassifier(n_neighbors=K)
        knn.fit(training, training_label)
        knn_predict = knn.predict(test)
        knn_akurasi = round(100 * accuracy_score(test_label, knn_predict))

        # Decision Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        dt_pred = dt.predict(test)
        dt_akurasi = round(100 * accuracy_score(test_label, dt_pred))

        # Artificial Neural Network with Backpropagation (ANNBP)
        if annbp:
            ann = Sequential()
            ann.add(Dense(10, input_dim=len(X.columns), activation='relu'))
            ann.add(Dense(10, activation='relu'))
            ann.add(Dense(1, activation='sigmoid'))
            ann.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            ann.fit(training, training_label, epochs=50, batch_size=16, verbose=0)
            ann_pred = ann.predict(test)
            ann_pred = np.round(ann_pred)
            ann_akurasi = round(100 * accuracy_score(test_label, ann_pred))

        if submitted:
            if naive:
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'.format(gaussian_akurasi))
            if k_nn:
                st.write("Model KNN accuracy score : {0:0.2f}".format(knn_akurasi))
            if destree:
                st.write("Model Decision Tree accuracy score : {0:0.2f}".format(dt_akurasi))
            if annbp:
                st.write("Model Artificial Neural Network with Backpropagation accuracy score : {0:0.2f}".format(ann_akurasi))

        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi': [gaussian_akurasi, knn_akurasi, dt_akurasi, ann_akurasi],
                'Model': ['Gaussian Naive Bayes', 'K-NN', 'Decision Tree', 'Artificial Neural Network with Backpropagation'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        Weight = st.number_input('Masukkan weight (berat) : ')
        Lenght = st.number_input('Masukkan lenght (panjang) : ')
        Circumference = st.number_input('Masukkan Circumference(keliling) : ')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree', 'Artificial Neural Network with Backpropagation'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                Weight,
                Lenght,
                Circumference
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian
            if model == 'K-NN':
                mod = knn 
            if model == 'Decision Tree':
                mod = dt
            if model == 'Artificial Neural Network with Backpropagation':
                mod = ann

            input_pred = mod.predict(input_norm)


            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)