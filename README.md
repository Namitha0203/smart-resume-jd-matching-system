# Smart Resume Analyzer & Job Description Matcher

A Flask web application that analyzes how well a resume matches a job description using **hybrid search** (semantic + keyword matching) and **Retrieval-Augmented Generation (RAG)**. The system computes a quantitative match score, identifies overlapping and missing skills, and generates grounded, human-readable feedback — every claim in the output is traceable to real, computed data rather than freely generated text.

## Features

- **Hybrid search** combining three independent similarity signals:
  - **SBERT** (`all-MiniLM-L6-v2`) — dense, semantic similarity
  - **TF-IDF** — sparse, general-purpose keyword similarity
  - **BM25** — sparse, curated-vocabulary keyword similarity, calibrated against a 6,187-resume corpus
- **Skill extraction** against a 197-term vocabulary, built from real extracted data (not hand-guessed), using word-boundary matching with edge-case handling for single-letter skills (e.g. correctly distinguishing `C` from `C#`/`C++`)
- **RAG-based feedback** — FAISS retrieves the most JD-relevant sentences from the resume; feedback is templated from real computed facts (score, shared/missing skills, retrieved excerpts), not open-ended LLM generation
- **Input validation** for empty, whitespace-only, and unreasonably short submissions
- A single-page web UI displaying the match score, detected keywords from both documents, shared/missing skills, all three similarity signals, and generated feedback

## Why no open-ended LLM generation?

Both `flan-t5-base` and `flan-t5-large` were tested across multiple prompt designs (direct instruction, question/answer framing, explicitly grounded fact-lists) for generating free-text match explanations. In every case the model either ignored the supplied facts and answered a generic related question, or echoed the input back without synthesis. This was treated as a genuine, tested finding rather than a prompting failure, and led directly to the templated-feedback design used here — every word in the output is grounded in real, computed data.

## Project Structure

smart_resume_analyzer/

├── app.py                       # Flask routes and input validation

├── requirements.txt

├── utils/

│   ├── cleaning.py              # Text cleaning (header spacing, address/contact stripping, etc.)

│   ├── features.py              # Skill extraction, TF-IDF, SBERT, BM25

│   ├── matching.py               # Combines features into final_match_score

│   └── rag.py                    # Sentence chunking, FAISS retrieval, templated feedback

├── data/

│   ├── skill_list_checkpoint.json   # 197-term skill vocabulary

│   ├── scaling_bounds.json          # Training-derived min/max bounds for live score scaling

│   └── bm25_index.pkl                # Pre-built BM25 index over the training resume corpus

├── templates/

│   └── index.html

└── static/


Note: the full training dataset (`df_fit_checkpoint.csv`, ~6,200 resume–JD pairs) is excluded from this repository via `.gitignore` due to size — it is not required at runtime, only during model development and validation.

## How It Works

1. **Cleaning** — raw resume/JD text is normalized: spacing fixes for glued formatting, contact info and address removal, punctuation normalization.
2. **Feature extraction** — skills are extracted from both documents; TF-IDF, SBERT, and BM25 similarity are computed.
3. **Scoring** — `final_match_score` (0–100) combines semantic+keyword similarity (50%), shared skill count (30%), and phrase overlap (20%), using fixed training-data bounds with clipping so the formula remains stable on new, unseen text.
4. **Retrieval** — the resume is split into sentence-level chunks; FAISS retrieves the chunks most semantically relevant to the job description.
5. **Feedback generation** — the score, shared/missing skills, and retrieved excerpts are assembled into a readable feedback sentence.

## Datasets Used

| Dataset | Source | Rows | Role |
|---|---|---|---|
| Resume–Job Description Fit | [`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit) | 6,241 (6,187 after cleaning) | Matching & scoring engine training/validation |
| Structured Resumes | [`datasetmaster/resumes`](https://huggingface.co/datasets/datasetmaster/resumes) | 4,817 | Skill vocabulary construction from real, nested resume data |

## Known Limitations

- **Seniority/role-level blind spot**: the model cannot reliably distinguish individual-contributor roles from management roles when vocabulary overlaps heavily (e.g. a Senior Engineer resume scoring very highly against a Senior Manager posting). A targeted keyword-based fix was attempted and itself failed validation; this remains an open, documented limitation rather than a forced patch.
- **BM25 is display-only, not blended into the final score**: correlation analysis showed BM25 and `common_skill_count` agree ~79% of the time on this dataset, since both measure skill-vocabulary overlap. Blending them would have diluted the formula's discriminative power without adding new information, so BM25 is shown to the user as a transparent, independent signal instead.
- **BM25 scoring for new (out-of-corpus) documents is a simplified approximation**: true BM25 requires the scored document to be part of the indexed corpus; new resumes are not. The implementation borrows the corpus's IDF term-rarity weights rather than performing full BM25 scoring.
- **Skill matching is word/phrase-based, not context-aware**: e.g. "sales" inside "post-sales" will register as a literal skill match even when the surrounding context isn't really about sales skills.
- **Address-stripping is pattern-based, not exhaustive**: covers common U.S. street-address formats only.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
python app.py
```

The app will be available at `http://127.0.0.1:5000`.

## Tech Stack

Python, Flask, scikit-learn, Sentence-Transformers, FAISS, rank-bm25, pandas, NumPy

## Author

Namitha