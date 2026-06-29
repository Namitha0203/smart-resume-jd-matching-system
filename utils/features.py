import re
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------------------------------------------------------------------------
# Module-level resources — loaded ONCE when this module is first imported,
# not on every request. This keeps the Flask app fast after startup.
# ---------------------------------------------------------------------------

# Skill vocabulary (197 terms, built from df_resumes + original broad-industry list)
with open('data/skill_list_checkpoint.json') as f:
    SKILL_LIST = json.load(f)

# SBERT model for semantic similarity (shared with utils/rag.py)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Pre-built BM25 index over the full training resume corpus (6,187 resumes).
# Needed because BM25's IDF weighting only makes sense relative to a real
# corpus — a brand-new resume can't be scored meaningfully in isolation.
with open('data/bm25_index.pkl', 'rb') as f:
    BM25_INDEX = pickle.load(f)


# ---------------------------------------------------------------------------
# Skill extraction
# ---------------------------------------------------------------------------

def extract_skills(text, skill_list=SKILL_LIST):
    text_lower = text.lower()
    found = []
    for skill in skill_list:
        pattern = r'\b' + re.escape(skill) + r'\b'
        for m in re.finditer(pattern, text_lower):
            if len(skill) == 1:
                before = text_lower[max(0, m.start() - 1):m.start()]
                after = text_lower[m.end():m.end() + 2]
                if before in ('.', '+') or after.startswith(('.', '#', '++')):
                    continue
            found.append(skill)
            break
    return found


# ---------------------------------------------------------------------------
# Similarity features
# ---------------------------------------------------------------------------

def get_tfidf_similarity(resume, jd):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    try:
        vectors = tfidf.fit_transform([resume, jd])
        return float(cosine_similarity(vectors[0], vectors[1])[0][0])
    except ValueError:
        return 0.0


def get_sbert_similarity(resume, jd):
    embeddings = sbert_model.encode([resume, jd])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

def tokenize_for_bm25(text):
    text = text.lower()
    text = re.sub(r'[^\w\s+#./]', ' ', text)  # remove punctuation but keep +, #, ., / for things like c++, c#, node.js
    return text.split()



def get_bm25_similarity(resume, jd):
    jd_tokens_clean = tokenize_for_bm25(jd)
    jd_skills_in_text = [t for t in jd_tokens_clean if t in SKILL_LIST]
    resume_tokens = set(tokenize_for_bm25(resume))

    raw_score = sum(
        BM25_INDEX.idf.get(token, 0) for token in jd_skills_in_text if token in resume_tokens
    )

    normalized = min(raw_score / 15.0, 1.0)
    return max(0.0, normalized)


def phrase_overlap(resume, jd):
    resume_words = resume.lower().split()
    jd_words = jd.lower().split()
    resume_phrases = set(' '.join(resume_words[i:i + 2]) for i in range(len(resume_words) - 1))
    jd_phrases = set(' '.join(jd_words[i:i + 2]) for i in range(len(jd_words) - 1))
    return len(resume_phrases & jd_phrases)


def truncate_text(text, max_words=256):
    return ' '.join(text.split()[:max_words])