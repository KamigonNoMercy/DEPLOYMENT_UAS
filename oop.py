import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

class ObesityModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.numerical = []
        self.ordinal = []
        self.binary = []
        self.nominal = []
        self.target = None
        self.ordinal_categories = [
            ['no', 'Sometimes', 'Frequently', 'Always'],  # CAEC
            ['no', 'Sometimes', 'Frequently', 'Always']   # CALC
        ]
        self.binary_categories = [['no', 'yes']] * 4  # 4 fitur binary
        self.model = None

    def eda(self):
        self.data = pd.read_csv(self.data_path)
        # Bersihkan kolom Age dari string 'years' dan ubah ke float
        if 'Age' in self.data.columns:
            self.data['Age'] = (
                self.data['Age']
                .astype(str)
                .str.replace('years', '', regex=False)
                .str.strip()
                .astype(float)
            )
        # Gabungkan kategori kecil di MTRANS jadi 'Others'
        if 'MTRANS' in self.data.columns:
            self.data['MTRANS'] = self.data['MTRANS'].replace(
                {'Walking': 'Others', 'Bike': 'Others', 'Motorbike': 'Others'}
            )
        # Standarisasi string di semua kolom kategorikal
        for col in ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype(str).str.strip()
        # Tentukan tipe fitur
        self.numerical = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        self.ordinal = ['CAEC', 'CALC']
        self.binary = ['FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight']
        self.nominal = ['Gender', 'MTRANS']
        self.target = 'NObeyesdad'

    def preprocess(self):
        X = self.data[self.numerical + self.ordinal + self.binary + self.nominal]
        y = self.data[self.target]
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def train(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess()

        # Set up transformer per group fitur
        numeric_transformer = Pipeline([
            ("scaler", StandardScaler())
        ])
        ordinal_transformer = Pipeline([
            ("ordinal", OrdinalEncoder(categories=self.ordinal_categories))
        ])
        binary_transformer = Pipeline([
            ("binary", OrdinalEncoder(categories=self.binary_categories))
        ])
        nominal_transformer = Pipeline([
            ("nominal", OrdinalEncoder())
        ])

        # Gabungkan jadi ColumnTransformer
        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, self.numerical),
            ("ord", ordinal_transformer, self.ordinal),
            ("bin", binary_transformer, self.binary),
            ("nom", nominal_transformer, self.nominal)
        ])

        # Buat pipeline model
        self.model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42))
        ])
        # Training
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        print("Model Performance:")
        print(classification_report(self.y_test, y_pred))
        print(f"Akurasi: {accuracy_score(self.y_test, y_pred):.4f}")

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

if __name__ == "__main__":
    trainer = ObesityModel("ObesityDataSet2.csv")  
    trainer.eda()
    trainer.train()
    trainer.save("best_model_pipeline.pkl")
