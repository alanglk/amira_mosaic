#!/usr/bin/env python3
"""
pasta_mosaic.py — Convierte una imagen en un mosaico de piezas de pasta.

1. Ajusta la imagen a la proporción del lienzo según --fit: recorta (crop),
   deforma (stretch) o reduce sin recortar rellenando con fondo (contain).
   Con --background las zonas de ese color (y el relleno) no llevan pasta:
   quedan como código 0 y se excluyen de K-Means.
2. Calcula cuántas piezas caben: columnas = canvas_width // pasta_width,
   filas = canvas_height // pasta_height.
3. Promedia el color de cada celda (una celda = una pieza de pasta).
4. Reduce la paleta con K-Means (en espacio CIELAB por defecto, más fiel a la
   percepción humana) y asigna un código 1..N a cada color, ordenado de
   oscuro a claro.
5. Genera previsualización, grid numerado en alta resolución, CSVs y
   config.json en una subcarpeta con nombre según la configuración.

Uso:
    python pasta_mosaic.py foto.jpg --colors 8 \
        --pasta-width 2.5 --pasta-height 1 \
        --canvas-width 100 --canvas-height 70 --unit cm

Dependencias:
    pip install numpy pillow matplotlib scikit-learn
"""

import argparse
import csv
import json
import math
import os
import re
import shlex
import sys

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
from scipy import ndimage
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans


# --------------------------------------------------------------------------- #
# Utilidades de color
# --------------------------------------------------------------------------- #
def rgb_to_lab(rgb):
    """Convierte colores sRGB uint8 (N,3) a CIELAB (D65)."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = (c @ m.T) / np.array([0.95047, 1.0, 1.08883])
    eps, k = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (k * xyz + 16) / 116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)


def text_color_for(rgb):
    """Negro o blanco según la luminosidad del fondo."""
    L = rgb_to_lab(np.array([rgb], dtype=np.uint8))[0, 0]
    return (0, 0, 0) if L > 55 else (255, 255, 255)


def to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*map(int, rgb))


# --------------------------------------------------------------------------- #
# Geometría del grid
# --------------------------------------------------------------------------- #
def compute_grid(canvas_w, canvas_h, pasta_w, pasta_h):
    cols = math.floor(canvas_w / pasta_w + 1e-9)
    rows = math.floor(canvas_h / pasta_h + 1e-9)
    if cols < 1 or rows < 1:
        raise ValueError("La pieza de pasta es más grande que el lienzo.")
    return cols, rows, cols * pasta_w, rows * pasta_h


def fit_image(img, target_aspect, mode, pad_color=(0, 0, 0)):
    """Iguala la proporción del lienzo útil.

    Devuelve (imagen, caja de la imagen original dentro del resultado o zona
    recortada, máscara de relleno). La máscara marca los píxeles añadidos por
    'contain', que nunca llevan pasta.
    """
    w, h = img.size
    no_pad = np.zeros((h, w), dtype=bool)
    if mode == "stretch":
        return img, (0, 0, w, h), no_pad
    if mode == "contain":
        if w / h > target_aspect:
            new_w, new_h = w, round(w / target_aspect)
        else:
            new_w, new_h = round(h * target_aspect), h
        left, top = (new_w - w) // 2, (new_h - h) // 2
        canvas = Image.new("RGB", (new_w, new_h), tuple(pad_color))
        canvas.paste(img, (left, top))
        pad = np.ones((new_h, new_w), dtype=bool)
        pad[top:top + h, left:left + w] = False
        return canvas, (0, 0, w, h), pad
    if w / h > target_aspect:
        new_w = round(h * target_aspect)
        left = (w - new_w) // 2
        box = (left, 0, left + new_w, h)
    else:
        new_h = round(w / target_aspect)
        top = (h - new_h) // 2
        box = (0, top, w, top + new_h)
    return img.crop(box), box, no_pad


def detect_background(img, border=0.02):
    """Color de fondo = mediana de una franja fina en los bordes."""
    a = np.asarray(img)
    bw = max(1, round(min(a.shape[:2]) * border))
    strip = np.concatenate([a[:bw].reshape(-1, 3), a[-bw:].reshape(-1, 3),
                            a[:, :bw].reshape(-1, 3), a[:, -bw:].reshape(-1, 3)])
    return np.median(strip, axis=0).round().astype(np.uint8)


def sample_cells(img, cols, rows, pad_mask, bg_rgb=None, bg_tol=12.0,
                 bg_coverage=0.5, bg_from_edges=False, ss=8):
    """Color medio de cada celda y máscara de celdas de fondo.

    Se sobremuestrea a ss×ss píxeles por celda para decidir qué parte de cada
    celda es fondo: una celda es fondo si su fracción de fondo >= bg_coverage,
    y si no, su color es la media solo de los píxeles que no son fondo (así
    los bordes no se oscurecen al mezclarse con el fondo).
    """
    size = (cols * ss, rows * ss)
    px = np.asarray(img.resize(size, Image.Resampling.BOX)).astype(np.float64)
    mask = np.asarray(Image.fromarray(pad_mask.astype(np.uint8) * 255)
                      .resize(size, Image.Resampling.BOX)) >= 128
    if bg_rgb is not None:
        dist = np.linalg.norm(rgb_to_lab(px.reshape(-1, 3))
                              - rgb_to_lab(np.array([bg_rgb])), axis=1)
        mask = mask | (dist.reshape(mask.shape) <= bg_tol)
    if bg_from_edges:
        # Solo cuenta como fondo lo conectado al borde (respeta pupilas, etc.)
        lbl, _ = ndimage.label(mask) # type: ignore
        edge = np.unique(np.concatenate([lbl[0], lbl[-1], lbl[:, 0], lbl[:, -1]]))
        mask = np.isin(lbl, edge[edge > 0])

    px = px.reshape(rows, ss, cols, ss, 3)
    fg = ~mask.reshape(rows, ss, cols, ss, 1)
    n_fg = fg.sum(axis=(1, 3))
    cells = np.where(n_fg > 0, (px * fg).sum(axis=(1, 3)) / np.maximum(n_fg, 1),
                     px.mean(axis=(1, 3)))
    is_bg = 1 - n_fg[..., 0] / (ss * ss) >= bg_coverage
    return cells.round().astype(np.uint8), is_bg


# --------------------------------------------------------------------------- #
# Cuantificación
# --------------------------------------------------------------------------- #
def quantize(cells, is_bg, n_colors, space="lab", seed=0):
    """K-Means solo sobre las celdas que llevan pasta. Fondo = código 0."""
    pixels = cells[~is_bg]
    if len(pixels) == 0:
        raise ValueError("Todas las celdas son fondo; revisa --background/--bg-tolerance.")
    n_unique = len(np.unique(pixels, axis=0))
    k = min(n_colors, n_unique)
    if k < n_colors:
        print(f"Aviso: solo hay {n_unique} colores distintos; se usarán {k}.")

    feats = rgb_to_lab(pixels) if space == "lab" else pixels.astype(np.float64)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(feats)
    labels = km.labels_

    # Color de cada cluster = media RGB real de sus celdas
    palette = np.array([pixels[labels == i].mean(axis=0) for i in range(k)])
    palette = palette.round().astype(np.uint8)

    # Ordenar de oscuro a claro para que los códigos sean estables
    order = np.argsort(rgb_to_lab(palette)[:, 0])
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    palette = palette[order]
    codes = np.zeros(cells.shape[:2], dtype=int)
    codes[~is_bg] = remap[labels] + 1  # códigos 1..k
    return palette, codes


def codes_to_rgb(codes, palette, bg_rgb):
    """Imagen RGB del mosaico; el código 0 se pinta con el color de fondo."""
    return np.vstack([np.array([bg_rgb], dtype=np.uint8), palette])[codes]


# --------------------------------------------------------------------------- #
# Salidas
# --------------------------------------------------------------------------- #
def load_font(size):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf",
                 "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def render_grid_image(codes, palette, bg_rgb, pasta_w, pasta_h, min_cell_px=32,
                      numbers=True):
    """Grid en alta resolución respetando la proporción real de cada pieza."""
    scale = min_cell_px / min(pasta_w, pasta_h)
    cw, ch = max(1, round(pasta_w * scale)), max(1, round(pasta_h * scale))
    rows, cols = codes.shape
    img = Image.new("RGB", (cols * cw + 1, rows * ch + 1), (128, 128, 128))
    draw = ImageDraw.Draw(img)
    digits = len(str(len(palette)))
    font = load_font(max(6, int(min(ch * 0.55, cw * 0.9 / (0.6 * digits)))))
    for r in range(rows):
        for c in range(cols):
            code = int(codes[r, c])
            color = tuple(int(v) for v in (palette[code - 1] if code else bg_rgb))
            x0, y0 = c * cw, r * ch
            draw.rectangle([x0, y0, x0 + cw, y0 + ch], fill=color,
                           outline=(90, 90, 90))
            if numbers and code:
                draw.text((x0 + cw / 2, y0 + ch / 2), str(code),
                          fill=text_color_for(color), font=font, anchor="mm")
    return img


def save_csvs(out_dir, codes, palette, bg_rgb):
    with open(os.path.join(out_dir, "grid_codes.csv"), "w", newline="") as f:
        csv.writer(f).writerows(codes.tolist())
    all_counts = np.bincount(codes.ravel(), minlength=len(palette) + 1)
    counts = all_counts[1:]
    with open(os.path.join(out_dir, "palette.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["codigo", "R", "G", "B", "hex", "piezas"])
        if all_counts[0]:
            w.writerow([0, *map(int, bg_rgb), to_hex(bg_rgb),
                        f"{int(all_counts[0])} (fondo, sin pasta)"])
        for i, (rgb, n) in enumerate(zip(palette, counts), start=1):
            w.writerow([i, *map(int, rgb), to_hex(rgb), int(n)])
    return counts


def render_preview(path, original, crop_box, codes, palette, counts, bg_rgb,
                   used_w, used_h, args):
    rows, cols = codes.shape
    n_pasta = int((codes > 0).sum())
    quant = codes_to_rgb(codes, palette, bg_rgb)
    extent = [0, used_w, used_h, 0]  # proporción física real

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    (ax_o, ax_q), (ax_p, ax_g) = axes

    # 1. Original (con zona de recorte marcada)
    ax_o.imshow(original)
    x0, y0, x1, y1 = crop_box
    if (x1 - x0, y1 - y0) != original.size:
        ax_o.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                 ec="red", lw=2, ls="--"))
    ax_o.set_title(f"Original ({original.width}×{original.height}px)")
    ax_o.axis("off")

    # 2. Paleta reducida a resolución de piezas
    ax_q.imshow(quant, extent=extent, interpolation="nearest")
    ax_q.set_title(f"Paleta reducida: {len(palette)} colores · "
                   f"{n_pasta} piezas de {cols}×{rows} celdas")
    ax_q.set_xlabel(args.unit)
    ax_q.set_ylabel(args.unit)

    # 3. Paleta de colores
    n = len(palette)
    ax_p.set_xlim(0, 10)
    ax_p.set_ylim(n, 0)
    for i, (rgb, cnt) in enumerate(zip(palette, counts)):
        ax_p.add_patch(Rectangle((0, i + 0.08), 2.2, 0.84,
                                 color=np.array(rgb) / 255, ec="black"))
        ax_p.text(1.1, i + 0.5, str(i + 1), ha="center", va="center",
                  fontsize=12, fontweight="bold",
                  color=np.array(text_color_for(rgb)) / 255)
        ax_p.text(2.6, i + 0.5,
                  f"{to_hex(rgb)}   RGB{tuple(map(int, rgb))}   "
                  f"{int(cnt)} piezas ({100 * cnt / n_pasta:.1f}%)",
                  va="center", fontsize=11, family="monospace")
    ax_p.set_title("Paleta")
    ax_p.axis("off")

    # 4. Grid con códigos
    ax_g.imshow(quant, extent=extent, interpolation="nearest")
    for c in range(cols + 1):
        ax_g.axvline(c * args.pasta_width, color="#555", lw=0.4)
    for r in range(rows + 1):
        ax_g.axhline(r * args.pasta_height, color="#555", lw=0.4)
    if cols * rows <= 3000:
        # Tamaño de celda en pantalla (≈6.5" de caja por subplot)
        box = 6.5
        disp_w = box * min(1, used_w / used_h)
        disp_h = box * min(1, used_h / used_w)
        cell_w_pt, cell_h_pt = disp_w / cols * 72, disp_h / rows * 72
        digits = len(str(len(palette)))
        fs = max(2, min(11, cell_h_pt * 0.75, cell_w_pt * 0.85 / (0.6 * digits)))
        for r in range(rows):
            for c in range(cols):
                code = int(codes[r, c])
                if not code:
                    continue
                ax_g.text((c + 0.5) * args.pasta_width,
                          (r + 0.5) * args.pasta_height, str(code),
                          ha="center", va="center", fontsize=fs,
                          color=np.array(text_color_for(palette[code - 1])) / 255)
        ax_g.set_title("Grid con códigos")
    else:
        ax_g.set_title("Grid (demasiadas celdas para numerar aquí: "
                       "ver grid_codes.png)")
    ax_g.set_xlabel(args.unit)
    ax_g.set_ylabel(args.unit)

    fig.suptitle(
        f"Lienzo {args.canvas_width}×{args.canvas_height} {args.unit} · "
        f"pieza {args.pasta_width}×{args.pasta_height} {args.unit} · "
        f"área usada {used_w:g}×{used_h:g} {args.unit}", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_name(args):
    """Nombre de carpeta a partir de la imagen y la configuración usada.

    Siempre incluye lienzo, pieza, colores y ajuste; el resto solo si difiere
    del valor por defecto, para que el nombre no crezca sin necesidad.
    """
    g = "{:g}".format
    parts = [os.path.splitext(os.path.basename(args.image))[0],
             f"c{g(args.canvas_width)}x{g(args.canvas_height)}{args.unit}",
             f"p{g(args.pasta_width)}x{g(args.pasta_height)}",
             f"n{args.colors}", args.fit]
    if args.crop:
        parts.append("crop" + args.crop.replace(",", "-"))
    if args.background:
        parts.append("bg-" + args.background.lstrip("#"))
        if args.bg_tolerance != 12.0:
            parts.append(f"tol{g(args.bg_tolerance)}")
        if args.bg_coverage != 0.5:
            parts.append(f"cov{g(args.bg_coverage)}")
        if args.bg_from_edges:
            parts.append("edges")
    if args.space != "lab":
        parts.append(args.space)
    if args.seed != 0:
        parts.append(f"seed{args.seed}")
    return re.sub(r"[^\w.+-]", "-", "_".join(parts))


def save_config(out_dir, args, bg_rgb):
    """Guarda los parámetros y el comando exacto para poder repetir la salida."""
    cfg = dict(vars(args))
    cfg["background_rgb"] = None if bg_rgb is None else to_hex(bg_rgb)
    cfg["command"] = shlex.join([sys.executable, *sys.argv])
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Mosaico de pasta a partir de una imagen.")
    p.add_argument("image", help="Imagen de entrada")
    p.add_argument("-n", "--colors", type=int, default=8, help="Nº de colores de la paleta (def. 8)")
    p.add_argument("--pasta-width", type=float, required=True, help="Ancho real de la pieza")
    p.add_argument("--pasta-height", type=float, required=True, help="Alto real de la pieza")
    p.add_argument("--canvas-width", type=float, required=True, help="Ancho del lienzo")
    p.add_argument("--canvas-height", type=float, required=True, help="Alto del lienzo")
    p.add_argument("--unit", default="cm", help="Unidad de medida (solo etiqueta, def. cm)")
    p.add_argument("--fit", choices=["crop", "stretch", "contain"], default="crop",
                   help="crop: recorta al centro para no deformar (def.); stretch: deforma; "
                        "contain: reduce la imagen entera y rellena el resto con fondo")
    p.add_argument("--crop", metavar="L,T,R,B",
                   help="Recorte previo en píxeles de la imagen original")
    p.add_argument("--background", metavar="COLOR",
                   help="Color de fondo pintado en el lienzo: 'auto' (bordes de la imagen), "
                        "'#RRGGBB' o nombre. Las celdas de ese color no llevan pasta")
    p.add_argument("--bg-tolerance", type=float, default=12.0,
                   help="Distancia CIELAB máxima al color de fondo (def. 12)")
    p.add_argument("--bg-coverage", type=float, default=0.5,
                   help="Fracción mínima de fondo para que una celda sea fondo (def. 0.5)")
    p.add_argument("--bg-from-edges", action="store_true",
                   help="Solo es fondo lo conectado al borde de la imagen "
                        "(evita quitar zonas oscuras interiores)")
    p.add_argument("--space", choices=["lab", "rgb"], default="lab",
                   help="Espacio de color para el clustering (def. lab)")
    p.add_argument("--seed", type=int, default=0, help="Semilla de K-Means")
    p.add_argument("--cell-px", type=int, default=32, help="Píxeles mínimos por celda en grid_codes.png")
    p.add_argument("-o", "--out", default="pasta_output",
                   help="Carpeta base de salida (def. pasta_output)")
    p.add_argument("--name",
                   help="Nombre de la subcarpeta de esta ejecución "
                        "(def. generado a partir de la imagen y la configuración)")
    args = p.parse_args()

    if args.colors < 1:
        p.error("--colors debe ser >= 1")
    out_dir = os.path.join(args.out, args.name or run_name(args))
    os.makedirs(out_dir, exist_ok=True)

    original = Image.open(args.image).convert("RGB")
    if args.crop:
        try:
            original = original.crop(tuple(int(v) for v in args.crop.split(","))) # type: ignore
        except ValueError:
            p.error("--crop debe ser L,T,R,B en píxeles")
    if args.background is None:
        bg_rgb = None
    elif args.background == "auto":
        bg_rgb = detect_background(original)
    else:
        try:
            bg_rgb = np.array(ImageColor.getrgb(args.background)[:3], dtype=np.uint8)
        except ValueError:
            p.error(f"Color de fondo no válido: {args.background}")
    # Color con el que se dibuja el relleno/fondo en las salidas
    draw_bg = bg_rgb if bg_rgb is not None else np.array([255, 255, 255], dtype=np.uint8)

    cols, rows, used_w, used_h = compute_grid(args.canvas_width, args.canvas_height,
                                              args.pasta_width, args.pasta_height)
    fitted, crop_box, pad_mask = fit_image(original, used_w / used_h, args.fit, draw_bg)
    cells, is_bg = sample_cells(fitted, cols, rows, pad_mask, bg_rgb, args.bg_tolerance,
                                args.bg_coverage, args.bg_from_edges)
    palette, codes = quantize(cells, is_bg, args.colors, args.space, args.seed)

    save_config(out_dir, args, bg_rgb)
    counts = save_csvs(out_dir, codes, palette, draw_bg)
    render_grid_image(codes, palette, draw_bg, args.pasta_width, args.pasta_height,
                      args.cell_px, numbers=True).save(os.path.join(out_dir, "grid_codes.png"))
    render_grid_image(codes, palette, draw_bg, args.pasta_width, args.pasta_height,
                      args.cell_px, numbers=False).save(os.path.join(out_dir, "mosaic.png"))
    render_preview(os.path.join(out_dir, "preview.png"), original, crop_box,
                   codes, palette, counts, draw_bg, used_w, used_h, args)

    n_bg = int(is_bg.sum())
    print(f"Resolución: {cols} columnas × {rows} filas = {cols * rows} celdas")
    if n_bg:
        print(f"Fondo {to_hex(draw_bg)} (código 0, sin pasta): {n_bg} celdas · "
              f"piezas de pasta: {cols * rows - n_bg}")
    print(f"Área usada: {used_w:g} × {used_h:g} {args.unit} "
          f"(sobrante: {args.canvas_width - used_w:g} × {args.canvas_height - used_h:g})")
    for i, (rgb, n) in enumerate(zip(palette, counts), start=1):
        print(f"  {i:>2}  {to_hex(rgb)}  {int(n)} piezas")
    print(f"Archivos guardados en: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()