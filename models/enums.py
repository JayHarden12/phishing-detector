from enum import Enum

class PredictionType(str, Enum):
    PHISHING = "phishing"
    LEGITIMATE = "legitimate"
    REJECTED = "rejected"
