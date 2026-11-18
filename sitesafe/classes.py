"""Class definitions shared by every script, the Streamlit app and (mirrored in TypeScript) the web app.

The order is the Construction-PPE order and must never change: the ONNX model's
output channels are indexed by it.
"""
from __future__ import annotations

CLASS_NAMES = [
    "helmet", "gloves", "vest", "boots", "goggles", "Person",
    "no_helmet", "no_goggle", "no_gloves", "no_boots", "forklift", "no_vest",
]
NUM_CLASSES = len(CLASS_NAMES)

PERSON = 5
FORKLIFT = 10
FORKLIFT_MIN_CONF = 0.5  # a weaker forklift box never raises a proximity alert or counts as a forklift

# PPE item -> (class id when worn, class id when visibly missing)
PPE_ITEMS = ["helmet", "goggles", "vest", "gloves", "boots"]
POS_CLASS = {"helmet": 0, "goggles": 4, "vest": 2, "gloves": 1, "boots": 3}
NEG_CLASS = {"helmet": 6, "goggles": 7, "vest": 11, "gloves": 8, "boots": 9}
NO_CLASSES = [6, 7, 8, 9, 11]

ITEM_OF_CLASS = {}
for _item in PPE_ITEMS:
    ITEM_OF_CLASS[POS_CLASS[_item]] = (_item, True)
    ITEM_OF_CLASS[NEG_CLASS[_item]] = (_item, False)

# Vertical band of the person box (fraction of height, 0 = top) where each item is expected.
ITEM_BAND = {
    "helmet": (0.00, 0.30),
    "goggles": (0.00, 0.30),
    "vest": (0.10, 0.75),
    "gloves": (0.15, 0.90),
    "boots": (0.75, 1.00),
}

# Site profiles: which PPE is mandatory. Goggles are only mandatory for cutting / grinding work.
SITE_PROFILES = {
    "general": ["helmet", "vest"],
    "full": ["helmet", "vest", "gloves", "boots"],
    "cutting_grinding": ["helmet", "vest", "gloves", "boots", "goggles"],
}
DEFAULT_PROFILE = "general"
