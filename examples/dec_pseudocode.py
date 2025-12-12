# DEC Pseudocode Example
# Dual Embedding Cross-Check (DEC)

def dec_compare(llm_embeddings, word_vectors, threshold=0.25):

    mapped = []
    for token_embed in llm_embeddings:
        nearest = find_closest_word_vector(token_embed, word_vectors)
        mapped.append(nearest)

    llm_path = compute_path(llm_embeddings)
    static_path = compute_path(mapped)

    drift = cosine_distance(llm_path, static_path)

    if drift > threshold:
        return "Potential Hallucination", drift
    
    return "Stable", drift
