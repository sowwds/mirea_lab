import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

X_clf, y_clf = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=2, random_state=42)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

scaler_clf = StandardScaler()
X_clf_train_scaled = scaler_clf.fit_transform(X_clf_train)
X_clf_test_scaled = scaler_clf.transform(X_clf_test)

param_grid = [
    {'C': [0.1, 1, 10, 100], 'kernel': ['linear']},
    {'C': [0.1, 1, 10, 100], 'kernel': ['rbf'], 'gamma': ['scale', 'auto', 0.1, 1]},
    {'C': [0.1, 1, 10, 100], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto']}
]

grid_search_clf = GridSearchCV(SVC(random_state=42, probability=True), param_grid, cv=5, scoring='accuracy', verbose=0, n_jobs=-1)
grid_search_clf.fit(X_clf_train_scaled, y_clf_train)

best_clf = grid_search_clf.best_estimator_
print(f"Лучшие параметры для SVC: {grid_search_clf.best_params_}")
print(f"Лучшая точность на кросс-валидации: {grid_search_clf.best_score_:.4f}")

y_clf_pred = best_clf.predict(X_clf_test_scaled)
print(classification_report(y_clf_test, y_clf_pred))

plt.figure(figsize=(10, 8))
plot_decision_regions(X_clf_test_scaled, y_clf_test, clf=best_clf, legend=2)
plt.title("SVM Classification - Граница решений для лучших параметров")
plt.xlabel("Признак 1 (масштабированный)")
plt.ylabel("Признак 2 (масштабированный)")
plt.show()

try:
    df_reg = pd.read_csv('Гиперспектр кукурузы.csv', sep=';', decimal=',')
    df_reg.columns = ['wavelength', 'Spectr']

    X_reg = df_reg[['wavelength']].values
    y_reg = df_reg['Spectr'].values
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

    scaler_reg_x = StandardScaler()
    scaler_reg_y = StandardScaler()

    X_reg_train_scaled = scaler_reg_x.fit_transform(X_reg_train)
    X_reg_test_scaled = scaler_reg_x.transform(X_reg_test)
    y_reg_train_scaled = scaler_reg_y.fit_transform(y_reg_train.reshape(-1, 1)).ravel()

    svr_param_grid = [
        {'C': [0.1, 1, 10, 100], 'kernel': ['linear']},
        {'C': [0.1, 1, 10, 100], 'kernel': ['rbf'], 'gamma': ['scale', 'auto', 0.01, 0.1]},
        {'C': [0.1, 1, 10, 100], 'kernel': ['poly'], 'degree': [2, 3]}
    ]

    grid_search_svr = GridSearchCV(SVR(), svr_param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
    grid_search_svr.fit(X_reg_train_scaled, y_reg_train_scaled)

    best_svr = grid_search_svr.best_estimator_
    print(f"\nЛучшие параметры для SVR: {grid_search_svr.best_params_}")

    y_reg_pred_scaled = best_svr.predict(X_reg_test_scaled)
    y_reg_pred = scaler_reg_y.inverse_transform(y_reg_pred_scaled.reshape(-1, 1))

    mse = mean_squared_error(y_reg_test, y_reg_pred)
    print(f"Mean Squared Error на тестовом наборе: {mse:.4f}")

    plt.figure(figsize=(12, 8))
    plt.scatter(X_reg_test, y_reg_test, color='black', label='Реальные значения', alpha=0.5)

    sort_axis = np.argsort(X_reg_test.ravel())
    plt.plot(X_reg_test[sort_axis], y_reg_pred[sort_axis], color='red', linewidth=2, label='Предсказания SVR')

    plt.title("SVR Regression - Гиперспектр кукурузы")
    plt.xlabel("Длина волны (wavelength)")
    plt.ylabel("Спектр (Spectr)")
    plt.legend()
    plt.show()

except FileNotFoundError:
    print("\nФайл 'Гиперспектр кукурузы.csv' не найден. Задание 2 пропущено.")

dt_clf = DecisionTreeClassifier(random_state=42)
rf_clf = RandomForestClassifier(random_state=42, n_jobs=-1)

dt_clf.fit(X_clf_train_scaled, y_clf_train)
rf_clf.fit(X_clf_train_scaled, y_clf_train)

y_pred_dt = dt_clf.predict(X_clf_test_scaled)
y_pred_rf = rf_clf.predict(X_clf_test_scaled)

print("Отчет по классификации: Best SVM")
print(classification_report(y_clf_test, y_clf_pred))

print("\nОтчет по классификации: Decision Tree")
print(classification_report(y_clf_test, y_pred_dt))

print("\nОтчет по классификации: Random Forest")
print(classification_report(y_clf_test, y_pred_rf))
