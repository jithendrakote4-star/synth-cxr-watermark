import hashlib
import numpy as np
import pywt


def make_payload(text, n_bits=64):
    """Text nundi deterministic 64-bit payload (model id / institution tag)."""
    h = hashlib.sha256(text.encode()).digest()
    return np.unpackbits(np.frombuffer(h, dtype=np.uint8))[:n_bits].astype(np.uint8)


def _qim_embed(c, bits, delta):
    off = bits * (delta / 2.0)
    return delta * np.round((c - off) / delta) + off


def _qim_extract(c, delta):
    q0 = delta * np.round(c / delta)
    q1 = delta * np.round((c - delta / 2.0) / delta) + delta / 2.0
    return (np.abs(c - q1) < np.abs(c - q0)).astype(np.uint8)


def _layout(n, n_bits, key):
    # key-dependent permutation: bits ni coefficients meeda spread chestundi
    perm = np.random.RandomState(key).permutation(n)
    bit_idx = np.arange(n) % n_bits
    return perm, bit_idx


def embed(img, payload, delta=12.0, key=1234):
    """img: 2D uint8 (grayscale, even dims). Haar DWT LH+HL sub-bands lo QIM embed."""
    x = np.asarray(img, dtype=np.float64)
    assert x.ndim == 2 and x.shape[0] % 2 == 0 and x.shape[1] % 2 == 0
    LL, (LH, HL, HH) = pywt.dwt2(x, "haar")
    c = np.concatenate([LH.ravel(), HL.ravel()])
    perm, bit_idx = _layout(c.size, len(payload), key)
    bits_at = np.empty(c.size, dtype=np.float64)
    bits_at[perm] = payload[bit_idx]
    c_w = _qim_embed(c, bits_at, delta)
    n = LH.size
    LH_w = c_w[:n].reshape(LH.shape)
    HL_w = c_w[n:].reshape(HL.shape)
    y = pywt.idwt2((LL, (LH_w, HL_w, HH)), "haar")
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


def extract(img, n_bits=64, delta=12.0, key=1234):
    x = np.asarray(img, dtype=np.float64)
    LL, (LH, HL, HH) = pywt.dwt2(x, "haar")
    c = np.concatenate([LH.ravel(), HL.ravel()])
    perm, bit_idx = _layout(c.size, n_bits, key)
    d = _qim_extract(c, delta)
    votes = np.bincount(bit_idx, weights=d[perm], minlength=n_bits)
    counts = np.bincount(bit_idx, minlength=n_bits)
    return (votes / counts > 0.5).astype(np.uint8)   # majority vote


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def bit_accuracy(a, b):
    return float(np.mean(a == b))