import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)  # NEW: letter immediately followed by digit
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    headers = ['Summary', 'Highlights', 'Experience', 'Education', 'Skills',
               'Professional Summary', 'Executive Profile', 'Profile']
    for h in headers:
        text = re.sub(rf'\b{h}(?=[A-Z])', h + ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', ' ', text)
    text = strip_address_pattern(text)
    text = text.replace('–', '-').replace('—', '-')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def strip_address_pattern(text):
    pattern = r'\d+\s+[A-Za-z\s]+(?:Drive|Street|Avenue|Road|Lane|Blvd|Boulevard|Way|Court|Place)[A-Za-z]*,?\s*[A-Z]{2}\s*\d{5}'
    text = re.sub(pattern, ' ', text)
    return text.strip()