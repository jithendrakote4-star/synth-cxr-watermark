import numpy as np
import pywt
from dwt_qim import _qim_embed, _qim_extract, _layout, make_payload, psnr, bit_accuracy  # noqa: F401


def embed_ll(img, payload, delta=64.0, level=3, key=1234):
    """Haar DWT approximation (LL) band lo QIM embed. Coarse band kabatti blur/JPEG/resize ki robust."""
    x = np.asarray(img, dtype=np.float64)
    assert x.ndim == 2 and x.shape[0] % (2 ** level) == 0 and x.shape[1] % (2 ** level) == 0
    coeffs = pywt.wavedec2(x, "haar", level=level)
    ll = coeffs[0]
    c = ll.ravel()
    perm, bit_idx = _layout(c.size, len(payload), key)
    bits_at = np.empty(c.size, dtype=np.float64)
    bits_at[perm] = payload[bit_idx]
    coeffs[0] = _qim_embed(c, bits_at, delta).reshape(ll.shape)
    y = pywt.waverec2(coeffs, "haar")
    return np.clip(np.round(y), 0, 255).astype(np.uint8)


def extract_ll(img, n_bits=64, delta=64.0, level=3, key=1234):
    x = np.asarray(img, dtype=np.float64)
    coeffs = pywt.wavedec2(x, "haar", level=level)
    c = coeffs[0].ravel()
    perm, bit_idx = _layout(c.size, n_bits, key)
    d = _qim_extract(c, delta)
    votes = np.bincount(bit_idx, weights=d[perm], minlength=n_bits)
    counts = np.bincount(bit_idx, minlength=n_bits)
    return (votes / counts > 0.5).astype(np.uint8)