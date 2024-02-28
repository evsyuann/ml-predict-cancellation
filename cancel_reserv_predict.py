import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("Hotel_Reservations.csv")

X = df[["lead_time", "repeated_guest", "no_of_previous_cancellations", "avg_price_per_room", "no_of_adults", "no_of_children"]]
y = df["booking_status"]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Creating the Logistic Regression model
model_Log_Reg = LogisticRegression(max_iter=1000, random_state=3)
model_Log_Reg.fit(X_train, y_train)

# Creating and train the Random Forest model
model_Ran_For = RandomForestClassifier(random_state=3)
model_Ran_For.fit(X_train, y_train)

# Making predictions on the test set
predictions_Log_Reg = model_Log_Reg.predict(X_test)
predictions_Ran_For = model_Ran_For.predict(X_test)

# Evaluating the models accuracy
accuracy_Log_Reg = accuracy_score(y_test, predictions_Log_Reg)
accuracy_Ran_For = accuracy_score(y_test, predictions_Ran_For)
print("Test Accuracy Log_Reg: {:.2f}".format(accuracy_Log_Reg))
print("Test Accuracy Ran_For: {:.2f}".format(accuracy_Ran_For))
print()

# Creating example of booking
example = {'lead_time': [48], 'repeated_guest': [0], 'no_of_previous_cancellations': [0], 
 'avg_price_per_room': [94.5], 'no_of_adults': [2], 'no_of_children': [0]}
example_new_df = pd.DataFrame(example)

# Making a prediction for RandomForest model because it has better accuracy
probabilities_Ran_For = model_Ran_For.predict_proba(example_new_df)[0]

if model_Ran_For.predict(example_new_df)[0] == "Not_Canceled":
    print(f"Reservation will be not canceled with probability {probabilities_Ran_For[1]*100:.2f}%")
elif model_Ran_For.predict(example_new_df)[0] == "Canceled":
    print(f"Reservation will be canceled with probability {probabilities_Ran_For[0]*100:.2f}%")

