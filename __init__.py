# Make P2 boosting classes easy to import from src.boosting.
from src.boosting.adaboost import AdaBoostClassifier, DecisionStump

# These two classes are the public P2 API.
__all__ = ["AdaBoostClassifier", "DecisionStump"]
