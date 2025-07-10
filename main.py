from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import joblib
from train import data_preparation

x_train, x_test, y_train, y_test = data_preparation('data/adult.csv')
encoders = joblib.load('encoders.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')


while True:
    print("\n========== Income Prediction Menu ==========")
    print("1 - Accuracy of Random Forest")
    print("2 - Accuracy of Logistic Regression")
    print("3 - Confusion Matrix (Random Forest)")
    print("4 - Confusion Matrix (Logistic Regression)")
    print("5 - Predict for new person")
    print("0 - Exit")

    choice = input("Enter your choice: ")

    if choice == '1':
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        acc = round(np.mean(pred == y_test), 2)
        print(f"Random Forest Accuracy: {acc * 100:.2f}%")

    elif choice == '2':
        model = LogisticRegression(max_iter=1000)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        acc = round(np.mean(pred == y_test), 2)
        print(f"Logistic Regression Accuracy: {acc * 100:.2f}%")

    elif choice == '3':
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        cm = confusion_matrix(y_test, pred)
        print("Confusion Matrix (Random Forest):")
        print(cm)

    elif choice == '4':
        model = LogisticRegression(max_iter=1000)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        cm = confusion_matrix(y_test, pred)
        print("Confusion Matrix (Logistic Regression):")
        print(cm)

    elif choice == '5':
        print("Enter new person’s data for prediction:")
        try:
            input_data = {}
            for col in columns:
                if col in encoders:
                    le = encoders[col]
                    classes = list(le.classes_)
                    print(f"\n{col} tanlang ({', '.join(classes)}):")
                    val = input(">>> ")
                    if val not in classes:
                        raise ValueError(f"{val} not in valid options for {col}")
                    encoded_val = le.transform([val])[0]
                    input_data[col] = encoded_val
                else:
                    val = float(input(f"\n{col}: "))
                    input_data[col] = val
            df = pd.DataFrame([input_data])
            df_scaled = scaler.transform(df)
            sample_df = pd.DataFrame(df_scaled, columns=columns)
            model = RandomForestClassifier()
            model.fit(x_train, y_train)

            pred = model.predict(sample_df)[0]
            result = '>50K' if pred == 1 else '<=50K'
            print(f"\nPrediction: {result}")
        except Exception as e:
            print("⚠️ Error:", e)


    elif choice == '0':
        print("Program ended.")
        break

    else:
        print("Invalid choice.")
