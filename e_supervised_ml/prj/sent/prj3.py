import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# --------------------------
# 1️ Load dataset
path = "prj/project/Dataset/anti-malware.csv"
df = pd.read_csv(path)

# Drop completely empty rows
df = df.dropna(how='all')

# Rename columns to integers
df.columns = range(df.shape[1])

# Drop first column if it's name/location
df = df.drop(columns=[0])

# Find target column (assume last column)
target_col = df.columns[-1]

# Drop rows where target is NaN
df = df[df[target_col].notna()]

# Separate target
Y = df.pop(target_col)
# Convert target to numeric 0/1 if necessary
if Y.dtype == 'object':
    Y = Y.map({'No':0, 'Yes':1})

X = df

# --------------------------
# 2️ Convert all columns to numeric (invalid entries become NaN)
X = X.apply(pd.to_numeric, errors='coerce')

# Separate numeric and categorical columns
num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Impute missing values for numeric columns
if len(num_cols) > 0:
    X[num_cols] = SimpleImputer(strategy='mean').fit_transform(X[num_cols])

# Impute missing values for categorical columns
if len(cat_cols) > 0:
    X[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(X[cat_cols])

# One-hot encode categorical columns if any
if len(cat_cols) > 0:
    X = pd.get_dummies(X, drop_first=True)

# Ensure all column names are strings
X.columns = X.columns.astype(str)

# Scale features
if not X.empty:
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-test split 80:20
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# --------------------------
# 3️ Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(x_train, y_train)
y_pred_log = log_model.predict(x_test)

# Confusion matrix for logistic regression
cm_log = confusion_matrix(y_test, y_pred_log)
print("Confusion Matrix - Logistic Regression:")
print(cm_log)
disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log)
disp_log.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Accuracy
acc_log = accuracy_score(y_test, y_pred_log)
print("Accuracy - Logistic Regression:", acc_log)

# --------------------------
# 4️ Naive Bayes
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
y_pred_nb = nb_model.predict(x_test)

# Confusion matrix for Naive Bayes
cm_nb = confusion_matrix(y_test, y_pred_nb)
print("Confusion Matrix - Naive Bayes:")
print(cm_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
disp_nb.plot()
plt.title("Naive Bayes Confusion Matrix")
plt.show()

# Accuracy
acc_nb = accuracy_score(y_test, y_pred_nb)
print("Accuracy - Naive Bayes:", acc_nb)

# --------------------------
# 5️ Compare performance
if acc_log > acc_nb:
    print("Logistic Regression performs better.")
elif acc_nb > acc_log:
    print("Naive Bayes performs better.")
else:
    print("Both models have similar performance.")