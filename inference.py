import pickle
import pandas as pd

class ObesityInference:
    def __init__(self, model_path="best_model_pipeline.pkl"):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.feature_names = [
            'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
            'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC',
            'FAF', 'TUE', 'CALC', 'MTRANS'
        ]

    def predict(self, input_data: dict):
        X = pd.DataFrame([input_data])
        X = X[self.feature_names]
        pred = self.model.predict(X)[0]
        return pred

if __name__ == "__main__":
    sample_input_1 = {
        "Gender": "Male",
        "Age": 25,
        "Height": 1.75,
        "Weight": 70,
        "family_history_with_overweight": "yes",
        "FAVC": "no",
        "FCVC": 3,
        "NCP": 3,
        "CAEC": "no",
        "SMOKE": "no",
        "CH2O": 2,
        "SCC": "no",
        "FAF": 1,
        "TUE": 2,
        "CALC": "Sometimes",
        "MTRANS": "Public_Transportation"
    }

    sample_input_2 = {
        "Gender": "Female",
        "Age": 30,
        "Height": 1.60,
        "Weight": 95,
        "family_history_with_overweight": "yes",
        "FAVC": "yes",
        "FCVC": 1,
        "NCP": 4,
        "CAEC": "Frequently",
        "SMOKE": "no",
        "CH2O": 1,
        "SCC": "yes",
        "FAF": 0,
        "TUE": 5,
        "CALC": "Frequently",
        "MTRANS": "Automobile"
    }

    infer = ObesityInference()
    prediction_1 = infer.predict(sample_input_1)
    print("Prediksi kasus 1:", prediction_1)

    prediction_2 = infer.predict(sample_input_2)
    print("Prediksi kasus 2:", prediction_2)
