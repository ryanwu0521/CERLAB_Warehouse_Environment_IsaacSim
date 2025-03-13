# =========================================
# config.py - Configuration Settings
# =========================================

# ---------- OmniVerse USD Stage ----------
USD_PATH = "omniverse://cerlabnucleus.lan.local.cmu.edu/Users/weihuanw/sim_environments/large_modular_warehouse_v1.usd"

# ---------- Feature Extraction ----------
PRIM_PATH = "/World/Warehouse/Features"  # USD Prim path for features

# ---------- Map Overlay & Partitioning ----------
OVERLAP_MARGIN = 1000.0  # Minimum distance between features in meters

# ---------- Noise Settings ----------
NOISE_STDDEV = 10.0  # Standard deviation for Gaussian noise (meters)

# ---------- Feature Matching ----------
MATCH_THRESHOLD = 0.8   # Cosine similarity threshold (0-1)
MAX_MATCH_DISTANCE = 100.0  # Maximum distance for feature matching (meters)

# ---------- Simulation Settings ----------
HEADLESS_MODE = False  # Set to True for headless execution (no UI)

# =========================================
# Notes:
# - Modify these values based on experiment needs.
# - Ensure USD_PATH is correctly set to your Omniverse environment.
# - Noise, threshold, and distance values affect map fusion accuracy.
# =========================================