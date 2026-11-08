"""Render a 'finished look' for plain zebra swatches (M16xx): colour + twill texture from the real swatch photo,
drawn as a blind on a light window in a calm room. Output 720x900 JPG. Usage: render_simple.py <photo> <out.jpg>"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def swatch_colours(path):
    im = np.asarray(Image.open(path).convert('RGB').resize((240, 320)), float)
    c = im[40:270, 30:200].reshape(-1, 3)
    bg = np.median(im[:, -8:].reshape(-1, 3), 0)          # grey table at the right edge
    # two clusters on fabric: solid band vs sheer band
    lum = c.mean(1)
    sat = c.max(1) - c.min(1)
    d_bg = np.abs(c - bg).sum(1)
    k = np.argsort(np.abs(c - np.median(c, 0)).sum(1))
    # solid = the cluster furthest from white
    w = np.array([250, 250, 250.])
    dist_w = np.abs(c - w).sum(1)
    solid = np.median(c[dist_w > np.percentile(dist_w, 60)], 0)
    sheer = np.median(c[dist_w < np.percentile(dist_w, 25)], 0)
    g = bg.mean() / np.maximum(bg, 1)                     # white-balance on the grey table
    solid = np.clip(solid * g * (222 / max(bg.mean(), 1)) ** 0.5, 0, 255)
    m = solid.mean(); solid = np.clip(m + (solid - m) * 1.12, 0, 255)
    sheer = np.clip(sheer * g, 0, 255)
    return solid, sheer

def texture_patch(path, solid):
    im = Image.open(path).convert('RGB'); w, h = im.size
    a = np.asarray(im, float)
    # find a 160px patch closest to the solid colour in the central area
    best, bp = 1e9, None
    for y in range(int(h * .15), int(h * .8), 40):
        for x in range(int(w * .1), int(w * .7), 40):
            p = a[y:y + 160, x:x + 160]
            e = np.abs(p.mean((0, 1)) - solid).sum() + p.std() * 0.5
            if e < best:
                best, bp = e, p
    t = bp - bp.mean((0, 1))
    return t  # zero-mean detail

def render(path, out, W=720, H=900):
    solid, sheer = swatch_colours(path)
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    # room: warm light wall with soft vignette
    wall = 224 - 72 * ((xx - W * .5) / W) ** 2 - 10 * (yy / H)
    img = np.stack([wall + 4, wall + 1, wall - 4], 2)
    # window opening and frame
    x0, x1, y0, y1 = 110, 610, 120, 790
    # daylight behind (soft sky/greenery blur)
    sky = np.zeros((H, W, 3))
    t = (yy - y0) / (y1 - y0)
    sky[..., 0] = 236 - 20 * t; sky[..., 1] = 240 - 14 * t; sky[..., 2] = 244 - 30 * t
    blob = Image.new('L', (W, H), 0); d = ImageDraw.Draw(blob)
    for i in range(14):
        cx, cy, r = rng.uniform(x0, x1), rng.uniform(y0 + 300, y1), rng.uniform(40, 110)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=int(rng.uniform(40, 90)))
    blob = np.asarray(blob.filter(ImageFilter.GaussianBlur(40)), float)[..., None] / 255
    sky = sky * (1 - blob) + np.array([150, 170, 140.]) * blob
    # fabric: alternating opaque / sheer bands, procedural twill (no seams)
    band = 56
    fab = np.zeros((H, W, 3)); alpha = np.zeros((H, W))
    tw = np.sin(2 * np.pi * (xx * .87 + yy * .5) / 3.6)
    lum = solid.mean()
    amp = 0.035 + 0.05 * (lum / 255)
    for k, yb in enumerate(range(y0 + 30, y1 - 36, band)):
        ye = min(yb + band, y1 - 36)
        if k % 2 == 0:
            seg = solid[None, None, :] * (1 + amp * tw[yb:ye, :, None])
            seg *= np.linspace(1.05, 0.95, ye - yb)[:, None, None]
            fab[yb:ye] = seg
        else:
            base = sky[yb:ye] * 0.45 + (sheer * 0.4 + 255 * 0.6) * 0.55
            r = np.arange(ye - yb)
            thr = (r % 5 == 2)[:, None, None]
            base = np.where(thr, base * 0.75 + solid * 0.25, base)
            base *= 1 - 0.04 * ((r[:, None, None] / (ye - yb)) - .5) ** 2
            fab[yb:ye] = base
        alpha[yb:ye] = 1
        # thin shadow under each opaque band
        if k % 2 == 0 and ye + 3 < H:
            fab[ye:ye + 3] *= 0.92
    fab = fab[:, x0 + 14:x1 - 14]
    img_win = sky.copy()
    img[y0:y1, x0:x1] = img_win[y0:y1, x0:x1]
    # frame (white PVC) and mullion
    fr = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)); d = ImageDraw.Draw(fr)
    d.rectangle((x0 - 14, y0 - 14, x1 + 14, y1 + 14), outline=(244, 243, 240), width=16)
    d.line(((x0 + x1) // 2, y0, (x0 + x1) // 2, y1), fill=(244, 243, 240), width=14)
    d.rectangle((x0 - 40, y1 + 14, x1 + 40, y1 + 34), fill=(238, 236, 232))     # sill
    img = np.asarray(fr, float)
    # blind over the window
    bx0, bx1 = x0 + 14, x1 - 14
    m = alpha[:, x0 + 14:x1 - 14][..., None]
    img[:, bx0:bx1] = img[:, bx0:bx1] * (1 - m) + fab * m
    # cassette and bottom bar
    cas = np.linspace(250, 222, 34)[:, None, None] * np.ones((1, bx1 - bx0 + 16, 3))
    img[y0 - 6:y0 + 28, bx0 - 8:bx1 + 8] = cas
    img[y1 - 40:y1 - 28, bx0:bx1] = np.linspace(246, 214, 12)[:, None, None]
    # shadow on wall under cassette ends + slight overall light
    out_im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    sh = Image.new('L', (W, H), 0); ds = ImageDraw.Draw(sh)
    ds.rectangle((bx0 - 8, y0 + 28, bx1 + 8, y0 + 40), fill=60)
    sh = sh.filter(ImageFilter.GaussianBlur(6))
    out_im = Image.composite(Image.new('RGB', (W, H), (60, 55, 50)), out_im, sh.point(lambda v: v * 0.5))
    noise = rng.normal(0, 2.2, (H, W, 1))
    out_im = Image.fromarray(np.clip(np.asarray(out_im, float) + noise, 0, 255).astype(np.uint8))
    out_im.save(out, quality=86)
    return solid, sheer

if __name__ == '__main__':
    print(render(sys.argv[1], sys.argv[2]))
