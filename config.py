import torch

# Tập từ vựng dừng trong tiếng Việt
STOP_WORDS = set([
    "là", "và", "của", "có", "trong", "với", "lại", "thì", "mà", "đã", "sẽ", "như", "ra", "ở", "để", "khi", "vẫn",
    "này", "nên", "đó", "đây", "các", "vì", "cũng", "nào", "vừa", "nên", "đi", "hơn", "rất", "đang", "trước", "sau",
    "được", "đến", "nhiều", "một", "chỉ", "số", "đã", "những", "vào", "qua", "đi", "không", "mà", "rằng", "từ", "năm",
    "hay", "tại", "bị", "đều", "lần", "mình", "còn", "xảy", "đợt", "theo", "hiện", "tuy nhiên", "gì", "tới", "về"
])

# Thiết lập môi trường thiết bị
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')