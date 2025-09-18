def replace_synonyms(text, synonyms):
    for k, v in synonyms.items():
        pattern = r'\b' + re.escape(k) + r'\b'
        text = re.sub(pattern, v, text, flags=re.IGNORECASE)
    return text

def apply_synonym_replacement(df, column, synonym_dict):
    df[column] = df[column].apply(lambda x: replace_synonyms(x, synonym_dict))
    return df