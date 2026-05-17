import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

# --------------------------
# 1 Load dataset
path = "prj/project/Dataset/BOM.csv"
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
Y = Y.map({'No':0, 'Yes':1})

X = df

# Separate numeric and categorical columns
num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Impute missing values
X[num_cols] = SimpleImputer(strategy='mean').fit_transform(X[num_cols])
X[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(X[cat_cols])

# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)
X.columns = X.columns.astype(str)

# Scale features
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# --------------------------
# 2️ Try different k for KNN
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
best_k = None
best_score = -float('inf')

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    score = r2_score(y_test, y_pred)
    print(f"k={k}, R² score={score:.4f}")
    
    if score > best_score:
        best_score = score
        best_k = k

print("\n Best k:", best_k)
print(" Best R² score:", best_score)