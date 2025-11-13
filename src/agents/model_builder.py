from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib, os

class ModelBuilderAgent:
    def __init__(self, models_dir='models'):
        os.makedirs(models_dir, exist_ok=True)
        self.models_dir = models_dir
        self.models = {
            'logreg': LogisticRegression(max_iter=1000),
            'rf': RandomForestClassifier(n_estimators=100)
        }

    def train_and_select(self, X, y):
        scores = {}

        # Dynamically adjust CV folds based on dataset size
        n_samples = len(X)
        if n_samples < 6:
            # For very small datasets, skip cross-validation
            for name, model in self.models.items():
                model.fit(X, y)
                scores[name] = model.score(X, y)
        else:
            # Use safe KFold with dynamic n_splits
            cv_folds = min(3, n_samples)
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

            for name, model in self.models.items():
                scores[name] = cross_val_score(model, X, y, cv=cv).mean()

        # Select the best model and save it
        best = max(scores, key=scores.get)
        self.models[best].fit(X, y)
        joblib.dump(self.models[best], f"{self.models_dir}/{best}.joblib")

        return best, scores
