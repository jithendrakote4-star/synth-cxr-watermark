import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "watermark"))
import numpy as np
from dwt_qim import make_payload, embed, extract, psnr, bit_accuracy
import attacks


def synthetic_image(seed=0, size=256):
    rng = np.random.RandomState(seed)
    base = np.linspace(40, 200, size)[None, :] * np.ones((size, 1))
    return np.clip(base + rng.normal(0, 6, (size, size)), 0, 255).astype(np.uint8)


def test_clean_roundtrip():
    img = synthetic_image()
    payload = make_payload("jithendra-lab|model-v1")
    wm = embed(img, payload)
    assert psnr(img, wm) > 38
    assert bit_accuracy(payload, extract(wm)) == 1.0


def test_wrong_key_fails():
    img = synthetic_image()
    payload = make_payload("jithendra-lab|model-v1")
    wm = embed(img, payload, key=1234)
    assert bit_accuracy(payload, extract(wm, key=999)) < 0.8


if __name__ == "__main__":
    img = synthetic_image()
    payload = make_payload("jithendra-lab|model-v1")
    wm = embed(img, payload)
    print("PSNR:", round(psnr(img, wm), 2), "dB")
    tests = {
        "clean": wm,
        "jpeg90": attacks.jpeg(wm, 90),
        "jpeg70": attacks.jpeg(wm, 70),
        "jpeg50": attacks.jpeg(wm, 50),
        "noise s=2": attacks.gaussian_noise(wm, 2),
        "noise s=5": attacks.gaussian_noise(wm, 5),
        "resize 0.75": attacks.resize_back(wm, 0.75),
        "blur r=1": attacks.blur(wm, 1),
    }
    for name, a in tests.items():
        print(f"{name:12s} bit acc: {bit_accuracy(payload, extract(a)):.3f}")