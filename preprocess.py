from underthesea import ner
from config import STOP_WORDS

# Xử lý văn bản trước khi đưa vào mô hình
def preprocess_text(text):
    ner_results = ner(text)
    processed_tokens = [
        word.lower() if pos_tag != 'Np' else word
        for entity in ner_results
        for word, pos_tag, _, _ in [entity]
        if pos_tag == 'Np' or (word.isalpha() and word.lower() not in STOP_WORDS)
    ]
    return ' '.join(processed_tokens)