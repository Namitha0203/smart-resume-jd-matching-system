import json
from utils.features import (
    extract_skills, get_tfidf_similarity, get_sbert_similarity,
    get_bm25_similarity, phrase_overlap, truncate_text
)

with open('data/scaling_bounds.json') as f:
    BOUNDS = json.load(f)


def scale_clipped(value, feature_name):
    b = BOUNDS[feature_name]
    if b['max'] == b['min']:
        return 0.5
    scaled = (value - b['min']) / (b['max'] - b['min'])
    return max(0.0, min(1.0, scaled))


def compute_match(resume_cleaned, jd_cleaned):
    resume_skills = extract_skills(resume_cleaned)
    jd_skills = extract_skills(jd_cleaned)

    common_skills = list(set(resume_skills) & set(jd_skills))
    missing_skills = list(set(jd_skills) - set(resume_skills))
    common_count = len(common_skills)

    tfidf_sim = get_tfidf_similarity(resume_cleaned, jd_cleaned)

    resume_truncated = truncate_text(resume_cleaned)
    jd_truncated = truncate_text(jd_cleaned)
    sbert_sim = get_sbert_similarity(resume_truncated, jd_truncated)

    bm25_sim = get_bm25_similarity(resume_cleaned, jd_cleaned)

    combined_sim = (tfidf_sim + sbert_sim) / 2
    phrase_count = phrase_overlap(resume_cleaned, jd_cleaned)

    combined_sim_scaled = scale_clipped(combined_sim, 'tfidf_sbert_combined')
    common_count_scaled = scale_clipped(common_count, 'common_skill_count')
    phrase_count_scaled = scale_clipped(phrase_count, 'exact_phrase_overlap_count')

    final_score = (
        0.50 * combined_sim_scaled +
        0.30 * common_count_scaled +
        0.20 * phrase_count_scaled
    ) * 100

    return {
        'resume_skills': resume_skills,
        'jd_skills': jd_skills,
        'common_skills': common_skills,
        'missing_skills': missing_skills,
        'tfidf_similarity': round(tfidf_sim, 3),
        'sbert_similarity': round(sbert_sim, 3),
        'bm25_similarity': round(bm25_sim, 3),
        'combined_similarity': round(combined_sim, 3),
        'phrase_overlap_count': phrase_count,
        'common_skill_count': common_count,
        'final_match_score': round(final_score, 1),
    }