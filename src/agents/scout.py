import pandas as pd, requests, io

class DataScoutAgent:
    def load_from_path(self, path): return pd.read_csv(path)
    def load_from_url(self, url):
        resp = requests.get(url, timeout=30); resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))
    def validate(self, df):
        if df.empty: return False, "Empty dataset"
        if df.columns.duplicated().any(): return False, "Duplicate columns"
        return True, "OK"
