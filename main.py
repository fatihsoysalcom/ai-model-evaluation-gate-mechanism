import random

def generate_synthetic_data(num_samples=1000):
    """Generates synthetic true labels and model predictions for a binary classification task."""
    y_true = [random.randint(0, 1) for _ in range(num_samples)]
    # Simulate a model with some errors to get varied metric results
    y_pred = []
    for true_label in y_true:
        if random.random() < 0.88: # Adjust this value to make the gate pass or fail
            y_pred.append(true_label)
        else:
            y_pred.append(1 - true_label) # Simulate an incorrect prediction
    return y_true, y_pred

def calculate_metrics(y_true, y_pred):
    """Calculates common binary classification metrics: Accuracy, Precision, Recall, F1-Score."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total_samples = len(y_true)
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

def evaluate_gate(metrics, thresholds):
    """Evaluates if all model metrics meet predefined thresholds for a CI/CD gate.
    This demonstrates the core concept of moving beyond a single metric evaluation.
    """
    print("\n--- CI/CD Gate Evaluation ---")
    gate_passed = True
    for metric_name, threshold_value in thresholds.items():
        current_value = metrics.get(metric_name, 0)
        if current_value < threshold_value:
            print(f"  ❌ FAIL: {metric_name.capitalize()} ({current_value:.2f}) is below threshold ({threshold_value:.2f})")
            gate_passed = False
        else:
            print(f"  ✅ PASS: {metric_name.capitalize()} ({current_value:.2f}) meets or exceeds threshold ({threshold_value:.2f})")
    return gate_passed

# --- Main Execution ---
if __name__ == "__main__":
    print("Simulating AI Model Evaluation for CI/CD Gate Mechanism")

    # 1. Generate synthetic data for demonstration
    y_true, y_pred = generate_synthetic_data(num_samples=1000)
    print(f"\nGenerated {len(y_true)} samples for evaluation.")

    # 2. Calculate various performance metrics for the model
    model_metrics = calculate_metrics(y_true, y_pred)
    print("\n--- Model Performance Metrics ---")
    for name, value in model_metrics.items():
        print(f"  {name.capitalize()}: {value:.2f}")

    # 3. Define the gate criteria (thresholds for each metric)
    # These thresholds represent the minimum acceptable performance for deployment.
    GATE_THRESHOLDS = {
        "accuracy": 0.85, # Example threshold for accuracy
        "precision": 0.75, # Example threshold for precision
        "recall": 0.70,    # Example threshold for recall
        "f1_score": 0.72   # Example threshold for F1-score
    }

    # 4. Apply the gate evaluation based on multiple criteria
    final_gate_result = evaluate_gate(model_metrics, GATE_THRESHOLDS)

    if final_gate_result:
        print("\n🎉 CI/CD Gate PASSED! Model is approved for deployment.")
    else:
        print("\n🛑 CI/CD Gate FAILED! Model requires further review or retraining before deployment.")
