import pickle
import numpy as np
print("[INFO] Loading encodings...")

with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())

print(data)
