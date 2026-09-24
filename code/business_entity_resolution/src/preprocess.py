import re
import unicodedata
import pandas as pd
from typing import Set

# Suffix dictionary for business names (multilingual & cross-country: US, India, France)
LEGAL_SUFFIX_MAP = {
    # English & Indian
    r'\bprivate\s+limited\b': 'pvtltd',
    r'\bpvt\.?\s*ltd\.?\b': 'pvtltd',
    r'\bp\.?\s*ltd\.?\b': 'pvtltd',
    r'\blimited\b': 'ltd',
    r'\bltd\.?\b': 'ltd',
    r'\bcorporation\b': 'corp',
    r'\bcorp\.?\b': 'corp',
    r'\bincorporated\b': 'inc',
    r'\binc\.?\b': 'inc',
    r'\bcompany\b': 'co',
    r'\bco\.?\b': 'co',
    r'\bl\.?l\.?c\.?\b': 'llc',
    r'\bl\.?l\.?p\.?\b': 'llp',
    # French (essential for Test Set generalization)
    r'\bsoci[eé]t[eé]\s+anonyme\b': 'sa',
    r'\bsoci[eé]t[eé]\s+[aà]\s+responsabilit[eé]\s+limit[eé]e\b': 'sarl',
    r'\bs\.?a\.?r\.?l\.?\b': 'sarl',
    r'\bs\.?a\.?s\.?u?\.?\b': 'sas',
    r'\be\.?u\.?r\.?l\.?\b': 'eurl',
    r'\bs\.?a\.?\b': 'sa',
}

ADDRESS_ABBREVS = {
    r'\bstreet\b': 'st', r'\bst\.?\b': 'st',
    r'\broad\b': 'rd', r'\brd\.?\b': 'rd',
    r'\bavenue\b': 'ave', r'\bave\.?\b': 'ave',
    r'\bboulevard\b': 'blvd', r'\bblvd\.?\b': 'blvd',
    r'\bbuilding\b': 'bldg', r'\bbldg\.?\b': 'bldg',
    r'\bfloor\b': 'fl', r'\bfl\.?\b': 'fl',
    r'\bopposite\b': 'opp', r'\bopp\.?\b': 'opp',
    r'\bnear\b': 'nr', r'\bnr\.?\b': 'nr',
    r'\bdrive\b': 'dr', r'\bdr\.?\b': 'dr',
    r'\blane\b': 'ln', r'\bln\.?\b': 'ln',
    r'\bsuite\b': 'ste', r'\bste\.?\b': 'ste',
    # French address terms
    r'\brue\b': 'rue',
    r'\bplace\b': 'pl',
    r'\bcedex\b': 'cedex',
}

def clean_unicode(text: str) -> str:
    """Normalize unicode characters, decompose accents (e.g., é -> e)."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text.lower().strip()

def clean_business_name(name: str) -> str:
    """Clean business name, apply suffix normalization and punctuation cleanup."""
    text = clean_unicode(name)
    # Normalize connectors
    text = re.sub(r'\s*&\s*', ' and ', text)
    text = re.sub(r'\s*\+\s*', ' and ', text)
    text = re.sub(r"[''`]", '', text)
    
    # Standardize legal suffixes
    for pattern, replacement in LEGAL_SUFFIX_MAP.items():
        text = re.sub(pattern, replacement, text)
        
    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_business_address(address: str) -> str:
    """Clean business address, standardize common abbreviations."""
    text = clean_unicode(address)
    text = re.sub(r"[''`]", '', text)
    
    for pattern, replacement in ADDRESS_ABBREVS.items():
        text = re.sub(pattern, replacement, text)
        
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_numbers(text: str) -> Set[str]:
    """Extract all numerical tokens (e.g. PIN codes, street numbers, suite numbers)."""
    if not isinstance(text, str):
        return set()
    return set(re.findall(r'\b\d+\b', text))

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply complete preprocessing pipeline to raw input DataFrame."""
    df = df.copy()
    
    # Fill NAs
    df['business_name'] = df['business_name'].fillna('')
    df['business_address'] = df['business_address'].fillna('')
    df['country'] = df['country'].fillna('')
    
    # Normalization
    df['clean_name'] = df['business_name'].apply(clean_business_name)
    df['clean_address'] = df['business_address'].apply(clean_business_address)
    df['country_clean'] = df['country'].apply(clean_unicode)
    
    # Combined representation
    df['clean_combined'] = df['clean_name'] + " " + df['clean_address']
    
    # Token extraction for rapid similarity evaluation
    df['name_tokens'] = df['clean_name'].apply(lambda x: set(x.split()))
    df['address_tokens'] = df['clean_address'].apply(lambda x: set(x.split()))
    df['address_numbers'] = df['clean_address'].apply(extract_numbers)
    
    return df
