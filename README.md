# Customer Churn Prediction

## 🎯 Objective

The aim of this task is to predict whether a bank customer is likely to **churn** (i.e., leave the bank) based on their profile and account activity. This involves using a classification model to make churn predictions and analyzing feature importance.

---

## 📁 Dataset Description

- **Filename:** `Churn_Modelling.csv`
- **Source:** Local CSV file (path: `task 3/Churn_Modelling.csv`)
- **Target Variable:** `Exited` (1 = Churned, 0 = Retained)
- **Key Features Used:**
  - `Geography`
  - `Gender`
  - `Age`
  - `Balance`
  - `Tenure`
  - `NumOfProducts`
  - `EstimatedSalary`

---

## ⚙️ Tools & Libraries Used

- Python
- Pandas
- Seaborn
- Matplotlib
- scikit-learn

---

## 🧪 Approach

### 1. Data Loading & Inspection
- Loaded data using `pandas.read_csv()`
- Previewed the dataset with `.head()`

### 2. Feature Selection
- Selected relevant columns for features (`X`)
- Used `Exited` as the target variable (`y`)

### 3. Encoding Categorical Variables
- Applied **one-hot encoding** on:
  - `Geography` (converted into dummy variables)
  - `Gender` (converted into binary dummy variable)
- Used `drop_first=True` to avoid multicollinearity

### 4. Train-Test Split
- Split the dataset using `train_test_split()` (80% train, 20% test)

### 5. Model Training
- Trained a **Random Forest Classifier** on the training data

### 6. Evaluation
- Measured performance using:
  - **Accuracy Score**
  - **Classification Report** (Precision, Recall, F1-score)
- Plotted **Feature Importances** using seaborn's barplot

---

## 📊 Results & Insights

- **Accuracy:** Around ~0.86 (may vary slightly per run)
- Key features influencing churn:
  - **Age**
  - **Balance**
  - **NumOfProducts**
  - **Geography_Germany**
- Random Forest performed well in classifying churn vs non-churn customers.
- Business can focus on specific customer profiles (e.g., older customers with low activity) to reduce churn.

---

## 📂 Files Included

- `Task-03.ipynb` – Complete Jupyter Notebook with code, outputs, and plots.
- `README.md` – This documentation file.

---

## ✅ Submission Checklist

- [x] Jupyter Notebook with code and markdown cells
- [x] Dataset loaded and processed
- [x] Features selected and encoded properly
- [x] Random Forest classifier trained
- [x] Model evaluated with metrics
- [x] Feature importance visualized
- [x] Clean code with comments
- [x] README summarizing the task
- [x] Uploaded to GitHub
- [x] Link submitted on Google Classroom

---

## 🚀 How to Run This Project

1. Ensure the dataset `Churn_Modelling.csv` is placed in the correct path.
2. Clone the repository:
   ```bash
   git clone https://github.com/SayabArshad/Task-3-Customer-Churn-Prediction-Bank-Customers.git
