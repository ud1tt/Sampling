import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load dataset
data = pd.read_csv("Creditcard_data.csv")

# Handling class imbalance using SMOTE
features = data.drop(columns=["Class"])  # Adjust "Class" to match target column
labels = data["Class"]

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(features, labels)

# Standardizing the features
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Determining appropriate sample size using statistical estimation
confidence_z = 1.96  # Corresponds to 95% confidence level
population_proportion = 0.5  # Assumed proportion
error_margin = 0.05  # Acceptable margin of error

sample_count = int((confidence_z**2 * population_proportion * (1 - population_proportion)) / (error_margin**2))
print(f"Computed Sample Size: {sample_count}")

# Generating samples using different sampling techniques
sampling_methods = {
    "random": X_resampled[:sample_count],
    "stratified": X_resampled[:sample_count],  # Simplified stratified sampling with balanced data
    "systematic": X_resampled[::len(X_resampled) // sample_count],
    "bootstrap": X_resampled[np.random.randint(0, len(X_resampled), sample_count)],
    "cluster": X_resampled[:sample_count]  # Replace with an actual clustering approach if needed
}

# Model selection
classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# Evaluating models across different sampling techniques
performance_results = {}

for method_name, sample_subset in sampling_methods.items():
    X_subset = sample_subset
    y_subset = y_resampled[:len(X_subset)]
    
    model_performance = {}
    for model_name, model in classification_models.items():
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        model_performance[model_name] = round(model.score(X_test, y_test), 2)
    
    performance_results[method_name] = model_performance

# Display results
print("\nModel Performance Across Sampling Techniques:\n")
for method_name, model_scores in performance_results.items():
    print(f"{method_name.capitalize()} Sampling:")
    for model_name, score in model_scores.items():
        print(f"{model_name}: {score}")
    print()

# Save results to file
with open("model_performance_results.txt", "w") as result_file:
    result_file.write("Model Performance Across Sampling Techniques:\n\n")
    for method_name, model_scores in performance_results.items():
        result_file.write(f"{method_name.capitalize()} Sampling:\n")
        for model_name, score in model_scores.items():
            result_file.write(f"{model_name}: {score}\n")
        result_file.write("\n")
