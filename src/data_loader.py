import sys
import re
import pandas as pd
from pathlib import Path

# Ensure root directory is in sys.path for direct module execution
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


def clean_text(text: str) -> str:
    """
    Cleans job description text by removing HTML tags, extra whitespace,
    and non-standard character artifacts.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Strip HTML tags (e.g. <p>, <br>, <li>)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 2. Normalize whitespace (tabs, newlines, multiple spaces -> single space)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def load_and_clean_data(file_path: Path) -> pd.DataFrame:
    """
    Loads job postings CSV, cleans missing values, standardizes column names,
    and applies text normalization. Auto-detects columns dynamically.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Clean column names (lowercase and strip whitespace)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # 1. Dynamically find the job description column
    desc_col = None
    possible_desc_names = [
        'descriptions (full)', 
        'description (summary)', 
        'job_description', 
        'description', 
        'summary',
        'job description'
    ]
    
    for col in possible_desc_names:
        if col in df.columns:
            desc_col = col
            break
            
    # Fallback if none match: select text column with highest average character length
    if not desc_col:
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            desc_col = max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())
        else:
            raise KeyError("Could not automatically locate a text description column in dataset.")

    # 2. Dynamically find the job title column
    title_col = None
    possible_title_names = ['job title', 'title', 'job_title', 'position']
    for col in possible_title_names:
        if col in df.columns:
            title_col = col
            break
            
    if not title_col:
        title_col = df.columns[0]  # Default to first column if title not found

    # Drop rows where job description is missing or blank
    df = df.dropna(subset=[desc_col]).reset_index(drop=True)
    
    # Create standardized clean columns
    df['clean_title'] = df[title_col].astype(str).str.strip()
    df['cleaned_description'] = df[desc_col].astype(str).apply(clean_text)
    
    print(f"✅ Data loaded successfully using '{desc_col}' column ({len(df)} postings).")
    return df


if __name__ == "__main__":
    import config
    df_test = load_and_clean_data(config.DATA_PATH)
    print("\nSample cleaned description:")
    print(df_test['cleaned_description'].iloc[0][:200] + "...")
    