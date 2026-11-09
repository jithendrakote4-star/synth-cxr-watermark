import io
import numpy as np
from PIL import Image, ImageFilter


def jpeg(img, q):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return np.array(Image.open(buf).convert("L"))


def gaussian_noise(img, sigma, seed=0):
    rng = np.random.RandomState(seed)
    out = img.astype(np.float64) + rng.normal(0, sigma, img.shape)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def resize_back(img, scale):
    h, w = img.shape
    im = Image.fromarray(img).resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return np.array(im.resize((w, h), Image.BILINEAR))


def blur(img, radius):
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius)))