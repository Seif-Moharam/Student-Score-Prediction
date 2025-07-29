import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping


def string_to_int(df):
    mapping = {"Low": 0, "Medium": 1, "High": 2,
               "No": 0, "Yes": 1,
               "Public": 0, "Private": 1,
               "Negative": 0, "Neutral": 1, "Positive": 2,
               "High School": 0, "College": 1, "Postgraduate": 2,
               "Near": 0, "Moderate": 1, "Far": 2,
               "Male": 0, "Female": 1}
    return df.replace(mapping)


data = pd.read_csv("StudentPerformanceFactors.csv")
data = data.dropna()
data = string_to_int(data)
x = data.drop("Exam_Score", axis=1)
y = data["Exam_Score"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss='mean_squared_error',
    metrics=['mae']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    x_train, y_train,
    epochs=200,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
print(f"Test MSE: {test_loss:.2f}, Test MAE: {test_mae:.2f}")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

y_hat = model.predict(x_test)
plt.scatter(y_test, y_hat, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("True Scores")
plt.ylabel("Predicted Scores")
plt.title("True vs Predicted Exam Scores")
plt.show()
