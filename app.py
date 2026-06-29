from flask import Flask, render_template, request
from utils.cleaning import clean_text
from utils.matching import compute_match
from utils.rag import generate_feedback

app = Flask(__name__)

MIN_WORDS = 15  # informed by our earlier data-cleaning threshold (we filtered JDs under 30 words;
                 # 15 is a slightly more lenient floor for live user input, not training data)


def validate_input(resume_text, jd_text):
    errors = []
    if not resume_text or len(resume_text.strip()) == 0:
        errors.append("Resume text is empty.")
    elif len(resume_text.split()) < MIN_WORDS:
        errors.append(f"Resume text is too short ({len(resume_text.split())} words). Please provide at least {MIN_WORDS} words.")

    if not jd_text or len(jd_text.strip()) == 0:
        errors.append("Job description is empty.")
    elif len(jd_text.split()) < MIN_WORDS:
        errors.append(f"Job description is too short ({len(jd_text.split())} words). Please provide at least {MIN_WORDS} words.")

    return errors


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    feedback = None
    resume_text = None
    jd_text = None
    errors = None

    if request.method == 'POST':
        resume_text = request.form.get('resume_text', '')
        jd_text = request.form.get('jd_text', '')

        errors = validate_input(resume_text, jd_text)

        if not errors:
            resume_cleaned = clean_text(resume_text)
            jd_cleaned = clean_text(jd_text)

            result = compute_match(resume_cleaned, jd_cleaned)
            feedback = generate_feedback(resume_cleaned, jd_cleaned, result)

    return render_template('index.html', result=result, feedback=feedback,
                            resume_text=resume_text, jd_text=jd_text, errors=errors)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)