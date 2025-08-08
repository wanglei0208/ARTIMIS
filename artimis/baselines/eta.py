import numpy as np
import copy
from get_models import get_models
class _SubstituteEnsemble:
    def __init__(self, model_names, x_train, y_train):
        self.models = [get_model(name, x_train, y_train) for name in model_names]
        self.classes_ = self.models[0].classes_
    def predict_proba(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        all_probas = [model.predict_proba(x) for model in self.models]
        return np.mean(all_probas, axis=0)

class _ZOEvasion:

    def __init__(self, substitute_models, eps, eps_step, max_iter):
        self.substitutes = substitute_models
        self.eps = eps
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.q = 15  # Number of random directions to sample for gradient estimation
        self.mu = 0.01  # Smoothing parameter for gradient estimation

    def _loss(self, x, label_index):
        """The loss is the probability of the target class."""
        proba = self.substitutes.predict_proba(x)
        return proba[0, label_index]

    def _estimate_gradient(self, x, label_index):
        """
        Core function: Estimates the gradient using Zeroth-Order (ZO) method.
        """
        grad_estimate = np.zeros_like(x, dtype=float)
        for _ in range(self.q):
            u = np.random.normal(size=x.shape)
            u /= np.linalg.norm(u)

            f_plus = self._loss(x + self.mu * u, label_index)
            f_minus = self._loss(x - self.mu * u, label_index)

            directional_derivative = (f_plus - f_minus) / (2 * self.mu)
            grad_estimate += directional_derivative * u

        return grad_estimate / self.q

    def generate(self, x_orig, y_orig_label_index):
        """
        Generates a single adversarial example for a given input.
        """
        x_adv = copy.deepcopy(x_orig)
        for _ in range(self.max_iter):
            grad = self._estimate_gradient(x_adv, y_orig_label_index)
            x_adv -= self.eps_step * np.sign(grad)

            delta = np.clip(x_adv - x_orig, -self.eps, self.eps)
            x_adv = np.clip(x_orig + delta, 0.0, 1.0)  # Assumes data is normalized [0,1]

        return x_adv


# ==============================================================================
# Main Attack Function (to be imported by your project)
# ==============================================================================
def eta_attack(model_name, X_test, y_test, X_train, y_train):
    print(f"\n[+] Running ETA (Zeroth-Order) Baseline Attack against '{model_name}'...")

    # 1. Load the black-box target model using the project's function
    target_model = get_model(model_name, X_train, y_train)

    # 2. Define and load the substitute models (all models except the target)
    substitute_names = ['DecisionTree', 'SVM', 'MLP', 'RandomForest', 'KNN']
    if model_name in substitute_names:
        substitute_names.remove(model_name)

    substitute_ensemble = _SubstituteEnsemble(substitute_names, X_train, y_train)

    # 3. Initialize the core attack agent
    attack_agent = _ZOEvasion(
        substitute_models=substitute_ensemble,
        eps=0.2,
        eps_step=0.02,
        max_iter=150
    )

    # 4. Get original predictions from the *target model*
    original_preds = target_model.predict(X_test)
    x_adv = np.zeros_like(X_test)

    print(f"[+] Generating adversarial examples for {len(X_test)} samples...")
    for i in range(len(X_test)):
        if (i + 1) % 10 == 0:
            print(f"  -> Processing sample {i + 1}/{len(X_test)}")

        x_sample = X_test[i:i + 1]
        original_pred_label = original_preds[i]

        # Convert the predicted label string to a numerical index for the loss function
        try:
            label_index = np.where(substitute_ensemble.classes_ == original_pred_label)[0][0]
        except IndexError:
            # Fallback if a label predicted by the target model doesn't exist in substitutes
            print(
                f"Warning: Label '{original_pred_label}' not found in substitute models. Skipping attack for this sample.")
            x_adv[i] = x_sample
            continue

        # Generate the adversarial example using the core logic
        x_adv_sample = attack_agent.generate(x_sample, label_index)
        x_adv[i] = x_adv_sample

    print("[+] Adversarial example generation complete.")
    return x_adv, y_test