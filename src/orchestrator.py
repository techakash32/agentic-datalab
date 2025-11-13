from agents.scout import DataScoutAgent
from agents.cleaner import DataCleanerAgent
from agents.model_builder import ModelBuilderAgent
from agents.evaluator import EvaluatorAgent
from agents.reasoner import ReasonerAgent
from sklearn.model_selection import train_test_split
import joblib

class Orchestrator:
    def run_pipeline(self, source, target):
        scout, cleaner = DataScoutAgent(), DataCleanerAgent()
        builder, evaluator, reasoner = ModelBuilderAgent(), EvaluatorAgent(), ReasonerAgent()
        df = scout.load_from_path(source)
        ok, msg = scout.validate(df)
        if not ok: raise ValueError(msg)
        df, _ = cleaner.run(df)
        X, y = df.drop(columns=[target]), df[target]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        best, scores = builder.train_and_select(Xtr, ytr)
        model = joblib.load(f"models/{best}.joblib")
        eval_json = evaluator.evaluate(model, Xte, yte)
        report = reasoner.generate_report(eval_json)
        return {'best_model': best, 'scores': scores, 'report': report}
