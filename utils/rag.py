import re
import faiss
import numpy as np
from utils.features import sbert_model


def split_into_sentences(text, max_words=40):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    for s in sentences:
        s = s.strip().lstrip('-').strip()
        words = s.split()
        if len(words) < 4:
            continue
        if len(words) > max_words:
            s = ' '.join(words[:max_words]) + '...'
        cleaned.append(s)
    return cleaned


def retrieve_relevant_chunks(resume_chunks, jd_text, top_k=3):
    if len(resume_chunks) == 0:
        return []

    chunk_embeddings = sbert_model.encode(resume_chunks)
    jd_embedding = sbert_model.encode([jd_text])

    chunk_embeddings = np.array(chunk_embeddings).astype('float32')
    jd_embedding = np.array(jd_embedding).astype('float32')

    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    faiss.normalize_L2(chunk_embeddings)
    faiss.normalize_L2(jd_embedding)

    index.add(chunk_embeddings)

    distances, indices = index.search(jd_embedding, min(top_k, len(resume_chunks)))

    results = [(resume_chunks[i], float(distances[0][j])) for j, i in enumerate(indices[0])]
    return results


def generate_feedback(resume_cleaned, jd_cleaned, match_result, top_k=3):
    """
    match_result: the dictionary returned by utils.matching.compute_match()
    """
    chunks = split_into_sentences(resume_cleaned)
    relevant_chunks = retrieve_relevant_chunks(chunks, jd_cleaned, top_k=top_k)

    score = match_result['final_match_score']
    common = match_result['common_skills']
    missing = match_result['missing_skills']

    if score >= 35:
        verdict = "a strong match"
    elif score >= 25:
        verdict = "a moderate match"
    else:
        verdict = "a weak match"

    common_str = ", ".join(common) if common else "no shared listed skills"
    missing_str = ", ".join(missing) if missing else "no notable gaps"
    evidence = "; ".join([c for c, s in relevant_chunks]) if relevant_chunks else "no strongly matching content found"

    feedback = (
        f"This resume is {verdict} for this role, scoring {score:.1f}/100. "
        f"Shared skills: {common_str}. Skills to highlight or develop: {missing_str}. "
        f"Most relevant resume content for this role: {evidence}"
    )
    return feedback