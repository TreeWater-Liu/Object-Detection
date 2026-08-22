"""Competition constants shared by data preparation and model scripts."""

CLASS_NAMES: tuple[str, ...] = (
    "person",
    "boat",
    "animal",
    "seat",
    "sign",
    "bicycle",
    "car",
    "ball",
    "light",
    "garbage can",
    "uav",
    "tricycle",
)

NUM_CLASSES = len(CLASS_NAMES)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
