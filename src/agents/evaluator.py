from sklearn.metrics import classification_report, confusion_matrix
import json, os

class EvaluatorAgent:
    def __init__(self, out_dir='reports'):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
    def evaluate(self, model, X, y):
        preds = model.predict(X)
        report = classification_report(y, preds, output_dict=True)
        cm = confusion_matrix(y, preds).tolist()
        out = {'report': report, 'confusion_matrix': cm}
        json.dump(out, open(f"{self.out_dir}/evaluation.json", "w"), indent=2)
        return out
