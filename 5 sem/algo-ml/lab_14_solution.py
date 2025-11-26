import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import warnings

# Игнорируем FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

def write_to_submission_file(predicted_labels, out_file,
                             target='Delinquent90', index_label="client_id"):
    predicted_df = pd.DataFrame(predicted_labels,
                                index=np.arange(75001, 75001 + len(predicted_labels)),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)
    print(f"Прогноз сохранен в файл {out_file}")

# --- 1. Загрузка и предобработка данных ---
print("--- 1. Загрузка и предобработка данных ---")
train_df = pd.read_csv('credit_scoring_train.csv', index_col='client_id')
test_df = pd.read_csv('credit_scoring_test.csv', index_col='client_id')

y = train_df['Delinquent90']
train_df.drop('Delinquent90', axis=1, inplace=True)

train_df['NumDependents'] = train_df['NumDependents'].fillna(train_df['NumDependents'].median())
train_df['Income'] = train_df['Income'].fillna(train_df['Income'].median())
test_df['NumDependents'] = test_df['NumDependents'].fillna(train_df['NumDependents'].median())
test_df['Income'] = test_df['Income'].fillna(train_df['Income'].median())
print("Данные загружены и обработаны.")

# --- Разделение на обучающую и валидационную выборки для оценки метрик ---
X_train_split, X_valid, y_train_split, y_valid = train_test_split(
    train_df, y, test_size=0.3, random_state=17, stratify=y
)

# --- 2. Настройка параметров (GridSearchCV) на части данных ---
print("\n--- 2. Настройка параметра max_features с помощью GridSearchCV ---")
start_time_grid = time.time()
forest_params = {'max_features': np.linspace(.3, 1, 7)}

locally_best_forest = GridSearchCV(
    RandomForestClassifier(n_estimators=100, random_state=17, n_jobs=-1, class_weight='balanced'),
    forest_params, scoring='roc_auc', cv=5, verbose=1
)
locally_best_forest.fit(X_train_split, y_train_split)

print(f"Поиск лучших параметров занял: {time.time() - start_time_grid:.2f} сек.")
print("Лучшие параметры:", locally_best_forest.best_params_)
print("Лучший ROC AUC на кросс-валидации:", round(locally_best_forest.best_score_, 3))

# --- 3. Оценка лучшей модели на валидационном наборе ---
print("\n--- 3. Оценка лучшей модели на валидационном наборе ---")
valid_pred = locally_best_forest.predict(X_valid)
cm = confusion_matrix(y_valid, valid_pred)

print(f"Accuracy на валидационной выборке: {accuracy_score(y_valid, valid_pred):.4f}")
print("Матрица ошибок:")
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# --- 4. Финальное предсказание для тестового набора ---
print("\n--- 4. Обучение финальной модели на ВСЕХ обучающих данных и предсказание ---")
start_time_final = time.time()
# Обучаем модель с лучшими параметрами на всех данных для лучшего результата
final_forest = RandomForestClassifier(
    n_estimators=300,
    max_features=locally_best_forest.best_params_['max_features'],
    random_state=17,
    n_jobs=-1,
    class_weight='balanced'
)
final_forest.fit(train_df, y)
final_forest_pred = final_forest.predict_proba(test_df)[:, 1]

print(f"Обучение финальной модели заняло: {time.time() - start_time_final:.2f} сек.")
write_to_submission_file(final_forest_pred, 'credit_scoring_final_forest.csv')

print("\nВсе задания выполнены.")