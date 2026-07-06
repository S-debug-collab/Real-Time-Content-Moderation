
LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


CATEGORY_TO_LABELS = {
    "Hate Speech": ["identity_hate"],
    "Toxic": ["toxic", "severe_toxic"],
    "Offensive": ["obscene", "insult", "threat"],
}

CATEGORY_PRIORITY = ["Hate Speech", "Toxic", "Offensive"]


def derive_primary_category(active_labels: set) -> str:
    
    for category in CATEGORY_PRIORITY:
        if any(label in active_labels for label in CATEGORY_TO_LABELS[category]):
            return category
    return "Neutral"
