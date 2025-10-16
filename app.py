import streamlit as st
import pandas as pd
import pickle

# Page setup
st.set_page_config(page_title="F1 Finish Predictor", layout="centered")
st.title("🏁 F1 Car Finish Predictor")
st.write("Predict the finishing position or probability using different ML models")

models = {
    "Logistic Regression": "logistic_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl",
    "SVM": "svm_model.pkl"
}

@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Model selection
model_name = st.selectbox("Select Model", list(models.keys()))
model = load_model(models[model_name])

# Input features 
st.subheader("Enter Race Details:")
constructor = st.selectbox("Constructor", ['McLaren', 'Maserati', 'Haas F1 Team', 'Euro Brun', 'Parnelli',
       'Sauber', 'Mercedes', 'Team Lotus', 'Onyx', 'Jordan', 'Dallara',
       'Ligier', 'Red Bull', 'Toro Rosso', 'ATS', 'Penske', 'Toleman',
       'Prost', 'Renault', 'Larrousse', 'Alfa Romeo', 'Force India',
       'March', 'Bellasi', 'Cooper-Climax', 'Lesovsky', 'Ferrari', 'BRM',
       'Minardi', 'Tyrrell', 'Brabham-BRM', 'Williams', 'Benetton',
       'Lotus', 'Brabham-Repco', 'Stewart', 'Alpine F1 Team', 'Zakspeed',
       'HWM', 'Caterham', 'Arrows', 'Coloni', 'Aston Martin',
       'Behra-Porsche', 'Theodore', 'BMW Sauber', 'Surtees', 'Ensign',
       'Jaguar', 'Brabham', 'Hesketh', 'Fittipaldi', 'AlphaTauri',
       'McLaren-Alfa Romeo', 'Toyota', 'Racing Point', 'RB F1 Team',
       'BAR', 'Veritas', 'Super Aguri', 'Shadow-Ford', 'Kuzma',
       'Footwork', 'Scirocco', 'Marussia', 'Lotus F1', 'RAM', 'Gordini',
       'Vanwall', 'Lola', 'Porsche', 'Watson', 'Talbot-Lago',
       'McLaren-Ford', 'Brawn', 'Virgin', 'Pacific', 'Leyton House',
       'Brabham-Ford', 'Life', 'Lotus-Climax', 'Emeryson', 'Lambo', 'HRT',
       'Kurtis Kraft', 'Osella', 'Phillips', 'Shadow',
       'Brabham-Alfa Romeo', 'Connaught', 'Matra', 'Lotus-BRM',
       'Cooper-Maserati', 'Simtek', 'Honda', 'Wolf', 'Lotus-Ford',
       'March-Alfa Romeo', 'Brabham-Climax', 'Iso Marlboro', 'Spirit',
       'Merzario', 'March-Ford', 'Andrea Moda', 'Wetteroth', 'Cooper',
       'MF1', 'Fondmetal', 'AGS', 'Cisitalia', 'Manor Marussia',
       'Matra-Ford', 'Forti', 'Rial', 'Tecno', 'Simca', 'Spyker MF1',
       'Martini', 'Snowberger', 'Embassy Hill', 'Cooper-BRM', 'LEC',
       'Epperly', 'OSCA', 'Maki', 'Trojan', 'De Tomaso', 'McLaren-BRM',
       'Eagle-Weslake', 'LDS-Climax', 'Eagle-Climax', 'Spyker', 'Bromme',
       'Schroeder', 'ERA', 'Marchese', 'JBW', 'Christensen', 'Gilby',
       'LDS-Alfa Romeo', 'Boro', 'Cooper-Borgward', 'Milano',
       'Cooper-Ferrari', 'Amon', 'Fry', 'BRP', 'De Tomaso-Osca', 'Alta',
       'Deidt', 'Adams', 'De Tomaso-Alfa Romeo', 'Pawl',
       'Aston Butterworth', 'Stevens', 'Nichels', 'Cooper-Castellotti',
       'Lyncar', 'LDS', 'Protos', 'Lancia', 'Lotus-Maserati', 'Moore',
       'Stebro', 'Scarab', 'Cooper-OSCA', 'AFM', 'Pankratz', 'ENB',
       'Trevis', 'EMW', 'Sherman', 'Meskowski', 'Token'])
driver = st.text_input("Driver Name")
grid = st.number_input("Grid Position", 1, 34, 1)
year = st.number_input("Year", 1, 2024, 2024)

input_df = pd.DataFrame({
    'constructor': [constructor],
    'driver_name': [driver],
    'grid': [grid],
    'year': [year]
})

# 🔮 Prediction
if st.button("Predict Finish"):
    try:
        pred = model.predict(input_df)[0]
        st.success(f"🏆 Predicted finishing position: **{pred}**")
    except Exception as e:
        st.error("⚠️ Error making prediction. Check feature preprocessing or encoding.")
        st.code(str(e))

model_accuracies = {
    "Logistic Regression": acc_log,  # replace with acc_log
    "Decision Tree": dec_tree,        # acc_dec_tree
    "Random Forest": acc_rd,        # acc_rf
    "XGBoost": acc_xg,              # acc_xg
    "SVM": acc_svm                   # acc_svm
}

st.sidebar.header("Model Accuracy")
for name, acc in model_accuracies.items():
    st.sidebar.write(f"{name}: **{acc*100:.2f}%**")
