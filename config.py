from pathlib import Path

# --- File Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "data_science_jobs.csv"


# --- Traditional NLP Configurations ---
TFIDF_MAX_FEATURES = 1000  # Vocabulary limit for TF-IDF matrix
TOP_K_KEYWORDS = 10        # Number of top characteristic keywords to extract per job

