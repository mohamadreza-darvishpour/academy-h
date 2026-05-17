# تمرین شناسایی چهره با Olivetti Faces
#اجازه ی دانلود داده نمیشه...
# ==============================

# 1- بارگذاری داده‌ها
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

data = fetch_olivetti_faces()
X = data.images       # تصاویر 64x64
y = data.target       # برچسب‌ها (۰ تا ۳۹)

# تقسیم داده به مجموعه آموزش و آزمون (۸۰:۲۰)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"تعداد تصاویر کل: {X.shape[0]}")
print(f"تعداد افراد: {len(np.unique(y))}")

# الف) نمایش یک تصویر از هر فرد

fig, axes = plt.subplots(4, 10, figsize=(15, 6))
for i in range(40):
    ax = axes[i//10, i%10]
    ax.imshow(X[y == i][0], cmap='gray')
    ax.axis('off')
    ax.set_title(f"{i}")
plt.tight_layout()
plt.show()

# ب) آموزش مدل SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# تبدیل تصاویر 2D به وکتور 1D
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train_flat, y_train)
y_pred_svm = svm_model.predict(X_test_flat)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"دقت SVM: {accuracy_svm:.2f}")

# ج) آموزش مدل‌های تجمعی: Random Forest و AdaBoost
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_flat, y_train)
y_pred_rf = rf_model.predict(X_test_flat)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# AdaBoost
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train_flat, y_train)
y_pred_ada = ada_model.predict(X_test_flat)
accuracy_ada = accuracy_score(y_test, y_pred_ada)

print(f"دقت Random Forest: {accuracy_rf:.2f}")
print(f"دقت AdaBoost: {accuracy_ada:.2f}")

# د) پیشنهاد بهبود نتایج
print("""
برای بهبود دقت می‌توان:
1. کاهش ابعاد با PCA یا LDA
2. تنظیم هایپرپارامترهای مدل‌ها (GridSearchCV)
3. افزایش داده‌ها (Data Augmentation)
4. استفاده از شبکه‌های عصبی عمیق (CNN)
""") 