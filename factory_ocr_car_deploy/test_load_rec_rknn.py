from rknnlite.api import RKNNLite
from pathlib import Path

MODEL = "./factory_rec_student.rknn"

print("--> Check model")
if not Path(MODEL).exists():
    raise FileNotFoundError(MODEL)

print("--> Load RKNN model")
rknn = RKNNLite()

ret = rknn.load_rknn(MODEL)
if ret != 0:
    print("Load RKNN model failed:", ret)
    exit(1)

print("Load RKNN model success")

print("--> Init runtime")
ret = rknn.init_runtime()
if ret != 0:
    print("Init runtime failed:", ret)
    exit(1)

print("Init runtime success")
print("Test OK")

rknn.release()
