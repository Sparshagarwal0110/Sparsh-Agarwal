import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

file_path = "calories_data.csv"
df = pd.read_csv(file_path)

df = df.dropna()

df['Duration_Distance'] = df['Duration (min)'] * df['Distance (km)']
df['Running_Duration'] = df['Duration (min)'] * (df['Exercise Type'] == 'Running').astype(int)
df['Walking_Duration'] = df['Duration (min)'] * (df['Exercise Type'] == 'Walking').astype(int)

X = df[['Duration (min)', 'Distance (km)', 'Weight (kg)', 'Age', 'Duration_Distance', 'Running_Duration', 'Walking_Duration']]
X = pd.get_dummies(X.join(df['Exercise Type']), columns=['Exercise Type'], drop_first=True)
y = df['Calories Burned']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

def predict_calories(exercise, duration, distance, weight, age):
    input_data = pd.DataFrame({
        'Duration (min)': [duration],
        'Distance (km)': [distance],
        'Weight (kg)': [weight],
        'Age': [age],
        'Duration_Distance': [duration * distance],
        'Exercise Type': [exercise]
    })

    input_data = pd.get_dummies(input_data, columns=['Exercise Type'], drop_first=True)

    for col in set(X.columns) - set(input_data.columns):
        input_data[col] = 0
    input_data = input_data[X.columns]

    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)[0]
    return prediction

exercise = input("Enter Exercise Type (e.g., Running, Walking, Cycling, Swimming): ")
duration = int(input("Enter Duration (in minutes): "))
distance = float(input("Enter Distance (in kilometers): "))
weight = float(input("Enter Weight (in kg): "))
age = int(input("Enter Age: "))

calories = predict_calories(exercise, duration, distance, weight, age)
print(f"\nPredicted Calories Burned: {calories:.2f}")