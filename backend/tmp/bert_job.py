from backend.app.coding import get_nlp_similarity


def job(df, model):
    df['similarity'] = df.apply(lambda row: get_nlp_similarity(row['w1'], row['w2'], model), axis=1)
    return df
