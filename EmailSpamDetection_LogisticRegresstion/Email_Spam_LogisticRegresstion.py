import tkinter as tk
from tkinter import Text
from tkinter import Label
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load the trained model and vectorizer
model = LogisticRegression()
vectorizer = CountVectorizer()

# Assume you have already trained the model and saved it (e.g., using joblib)
# Load the trained model
model = joblib.load('trained_model.joblib')

# Load the vectorizer
vectorizer = joblib.load('vectorizer.joblib')

def check_spam():
    # Get the text from the Text widget
    input_text = text.get("1.0", "end-1c")

    # Vectorize the input text
    input_vectorized = vectorizer.transform([input_text])

    # Make a prediction
    prediction = model.predict(input_vectorized)

    # Display the result
    result_label.config(text=f"Predicted Label: {prediction[0]}")

root = tk.Tk()
root.geometry('500x250')
root.resizable(False, False)
root.title("Spam Detector")

label = Label(root, text='CHECK EMAIL SPAM DETECTION')
label.pack(ipadx=10, ipady=10)

text = Text(root, height=8, width=50)
text.pack()

check_button = tk.Button(root, text="Check Spam", command=check_spam)
check_button.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()

# Dữ liệu test
# test_email = "Congratulations! You've won a free cruise. Click here to claim your prize now!"
