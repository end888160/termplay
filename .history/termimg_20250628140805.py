#!/usr/bin/env python3.8
import argparse
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import shutil
import sys
import colorsys
import humanize
from tqdm import tqdm
import numpy as np
from PIL import Image
import zlib
import random

def image_to_array(img):
    return np.asarray(img).astype(np.float32)

def array_to_image(arr):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


ASCII_CHARS = "@%#*+=-:. "
BLOCK_CHAR = "█"

RESAMPLING_MAP = {
    "none": None,
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "box": Image.BOX,
    "hamming": Image.HAMMING
}

# Standard ANSI 16 colors (foreground codes)
ANSI_16_COLORS = [
        (0, 0, 0),       # 30 black
        (128, 0, 0),     # 31 red
        (0, 128, 0),     # 32 green
        (128, 128, 0),   # 33 yellow
        (0, 0, 128),     # 34 blue
        (128, 0, 128),   # 35 magenta
        (0, 128, 128),   # 36 cyan
        (192, 192, 192), # 37 white (light gray)

        (128, 128, 128), # 90 bright black (dark gray)
        (255, 0, 0),     # 91 bright red
        (0, 255, 0),     # 92 bright green
        (255, 255, 0),   # 93 bright yellow
        (0, 0, 255),     # 94 bright blue
        (255, 0, 255),   # 95 bright magenta
        (0, 255, 255),   # 96 bright cyan
        (255, 255, 255)  # 97 bright white
    ]

ANSI_CODES = [30, 31, 32, 33, 34, 35, 36, 37,
                  90, 91, 92, 93, 94, 95, 96, 97]


def apply_sepia(img, value=1.0):
    """
    Apply sepia tone to an image with adjustable intensity.
    `value` should be between 0.0 (no effect) and 1.0 (full sepia).
    """
    if value == 0.0:
        return img.copy()  # no change

    # Create a sepia version
    sepia_matrix = (
        0.393, 0.769, 0.189, 0,
        0.349, 0.686, 0.168, 0,
        0.272, 0.534, 0.131, 0
    )
    sepia = img.convert("RGB", sepia_matrix)

    if value == 1.0:
        return sepia

    # Blend original and sepia
    return Image.blend(img, sepia, value)

def apply_edge_detection(img):
    """ Apply edge detection to an image. """

    gray = img.convert("L")
    arr = np.asarray(gray).astype(np.float32)

    dx = scipy.ndimage.sobel(arr, axis=1)
    dy = scipy.ndimage.sobel(arr, axis=0)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)

    edge_img = Image.fromarray(mag.astype(np.uint8))
    return edge_img.convert("RGB")

def apply_bloom(img, threshold=200, blur_radius=8, intensity=0.7):
    """ Apply bloom effect to an image. `threshold` and `intensity` should be between 0.0 and 1.0. """
    img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32)

    # Convert to grayscale to find bright areas
    gray = np.mean(arr, axis=2)
    mask = gray > threshold  # bright pixels mask

    # Create glow layer
    glow = np.zeros_like(arr)
    glow[mask] = arr[mask]

    # Convert glow to image and blur it
    glow_img = Image.fromarray(np.clip(glow, 0, 255).astype(np.uint8))
    glow_blurred = glow_img.filter(ImageFilter.GaussianBlur(blur_radius))

    # Blend glow with original
    glow_arr = np.asarray(glow_blurred).astype(np.float32)
    result = np.clip(arr + glow_arr * intensity, 0, 255).astype(np.uint8)

    return Image.fromarray(result)

def apply_grain_noise(img, strength=0.1):
    """ Add grainy noise to an image. `strength` should be between 0.0 and 1.0. """
    arr = image_to_array(img)
    noise = np.random.normal(0, 255 * strength, arr.shape)
    noisy = arr + noise
    return array_to_image(noisy)

def apply_chromatic_aberration(img, shift_x=2, shift_y=1):
    """ Apply chromatic aberration to an image. """
    img = img.convert("RGB")
    arr = np.asarray(img)

    # Split channels
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    def shift_channel(channel, dx, dy):
        return np.roll(channel, shift=(dx, dy), axis=(0, 1))

    # Shift each channel slightly differently
    r_shifted = shift_channel(r, -shift_x, -shift_y)
    g_shifted = shift_channel(g, 0, 0)
    b_shifted = shift_channel(b, shift_x, shift_y)

    # Stack back together
    result = np.stack([r_shifted, g_shifted, b_shifted], axis=-1)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)



def apply_filters(img, filters):
    for f in tqdm(filters, desc="Applying filters", delay=2, unit="filter"):
        try:
            if '=' in f:
                key, value = f.split('=', 1)
                key = key.lower().strip()
                value = value.strip()
                if not value.isdigit() and not (value.replace('.', '', 1).isdigit() and value.count('.') < 2):
                    # If value is not numeric, skip this filter
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
                        # If value is not a valid number, skip this filter
                        if value:
                            value = float(value)
                    else:
                        print(f"⚠️ Invalid value ({value}) for filter '{key}': must be numeric. Skipping...")
                        continue
                value = float(value)
            else:
                key = f.lower()
                value = None

            key = key.lower()

            if key == "brightness":
                img = ImageEnhance.Brightness(img).enhance(value)
            elif key == "contrast":
                img = ImageEnhance.Contrast(img).enhance(value)
            elif key == "grayscale":
                img = img.convert("L").convert("RGB")
            elif key == "invert":
                img = ImageOps.invert(img)
            elif key == "saturate":
                img = ImageEnhance.Color(img).enhance(value)
            elif key == "sharpness":
                img = ImageEnhance.Sharpness(img).enhance(value)
            elif key == "sepia":
                img = apply_sepia(img, value)
            elif key == "hue":
                img = hue_rotate(img, value)
            elif key == "blur":
                img = img.filter(ImageFilter.GaussianBlur(value))
            elif key == "posterize":
                img = ImageOps.posterize(img, int(value))
            elif key == "rotate":
                img = img.rotate(value)
            elif key == "edge":
                img = apply_edge_detection(img)
            elif key == "noise":
                img = apply_grain_noise(img, value)
            elif key == "bloom":
                img = apply_bloom(img, intensity= value / 200, blur_radius=value)
            elif key == "chromatic":
                img = apply_chromatic_aberration(img, value)
            else:
                print(f"⚠️ Unknown filter '{key}'. Skipping...")
        except Exception as e:
            print(f"⚠️ Error applying filter '{key}' ({value}): {e}")
    return img

def hue_rotate(img, degrees):
    img = img.convert("RGB")
    pixels = img.load()
    for y in range(img.height):
        for x in range(img.width):
            r, g, b = [v / 255.0 for v in pixels[x, y]]
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            h = (h + degrees / 360.0) % 1.0
            r, g, b = [int(c * 255) for c in colorsys.hls_to_rgb(h, l, s)]
            pixels[x, y] = (r, g, b)
    return img

def pixel_to_ascii(r, g, b):
    gray = int((r + g + b) / 3)
    return ASCII_CHARS[gray * len(ASCII_CHARS) // 256]

def rgb_to_ansi_fg_bg(top, bottom, char='▀'):
    tr, tg, tb = top
    br, bg, bb = bottom
    return f"\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m{char}\x1b[0m"

def rgb_to_ansi(r, g, b, char='█'):
    return f"\x1b[38;2;{r};{g};{b}m{char}\x1b[0m"

def rgb_to_ansi256(r, g, b):
    # Map RGB to xterm 256-color palette
    # 0-15: basic colors, 16-231: color cube, 232–255: grayscale
    def rgb_to_cube(v):
        return 0 if v < 48 else 1 if v < 114 else (v - 35) // 40

    r_idx = rgb_to_cube(r)
    g_idx = rgb_to_cube(g)
    b_idx = rgb_to_cube(b)
    color_code = 16 + 36 * r_idx + 6 * g_idx + b_idx
    return f"\x1b[38;5;{color_code}m"

def rgb_to_ansi16_grey(r, g, b):
    # Map to 16 ANSI color codes
    gray = int((r + g + b) / 3)
    if gray > 240:
        return "\x1b[97m"  # bright white
    if gray > 200:
        return "\x1b[37m"  # white
    if gray > 160:
        return "\x1b[37m"
    if gray > 120:
        return "\x1b[90m"  # bright black (gray)
    if gray > 80:
        return "\x1b[90m"
    if gray > 40:
        return "\x1b[30m"  # black
    return "\x1b[30m"



def rgb_to_ansi16(r, g, b, char=BLOCK_CHAR):
    """ Convert RGB to ANSI 16 color escape code. """
    # Find the closest ANSI 16 color using Euclidean distance
    min_dist = float('inf')
    best_index = 0
    for i, (cr, cg, cb) in enumerate(ANSI_16_COLORS):
        dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if dist < min_dist:
            min_dist = dist
            best_index = i

    if best_index < 8:
        return f"\x1b[{30 + best_index}m"
    else:
        return f"\x1b[{90 + (best_index - 8)}m"



def braille_char(block):
    # block: list of 8 0/1 values for dots 1–8
    dot_order = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80]
    value = sum(dot for on, dot in zip(block, dot_order) if on)
    return chr(0x2800 + value)

def rgb_to_ansi_8(r, g, b, char=BLOCK_CHAR):
    """ Convert RGB to ANSI 8 color escape code. """
    # Only standard ANSI colors (30-37)
    base_colors = ANSI_16_COLORS[:8]  # first 8 are standard
    min_dist = float('inf')
    best_index = 0
    for i, (cr, cg, cb) in enumerate(base_colors):
        dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if dist < min_dist:
            min_dist = dist
            best_index = i
    return f"\x1b[{30 + best_index}m"

def rgb_to_ansi_16(r, g, b, char=BLOCK_CHAR):
    """ Convert RGB to ANSI 16 color escape code. """
    # Find the closest ANSI 16 color using Euclidean distance
    min_dist = float('inf')
    best_index = 0
    for i, (cr, cg, cb) in enumerate(ANSI_16_COLORS):
        dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if dist < min_dist:
            min_dist = dist
            best_index = i

    if best_index < 8:
        return f"\x1b[{30 + best_index}m{char}\x1b[0m"
    else:
        return f"\x1b[{90 + (best_index - 8)}m{char}\x1b[0m"

import random

def apply_dither(
        img, 
        dither_levels=4, 
        dither_diffusion='floyd', 
        dither_mode='gray'
    ):
    """
    General dithering function.
    - levels: number of intensity levels
    - diffusion: 'floyd', 'atkinson', 'random', 'none', etc.
    - mode: 'gray' (1 channel), 'rgb' (3 channels)
    """
    img = img.convert("RGB")
    arr = img.load()
    w, h = img.size
    step = 256 // dither_levels

    # choose error matrix
    if dither_diffusion == 'floyd':
        pattern = [(1, 0, 7/16), (-1,1,3/16), (0,1,5/16), (1,1,1/16)]
    elif dither_diffusion == 'atkinson':
        pattern = [(1,0,1/8), (2,0,1/8), (-1,1,1/8), (0,1,1/8), (1,1,1/8), (0,2,1/8)]
    elif dither_diffusion == 'bayer':
        pattern = [(1, 0, 0.25), (0, 1, 0.25), (1, 1, 0.25), (0, 0, 0.25)]
    elif dither_diffusion == 'stucki':
        pattern = [(1, 0, 8/42), (-2, 1, 4/42), (1, 1, 2/42), (2, 1, 1/42), (0, 1, 4/42), (0, 2, 2/42)]
    elif dither_diffusion == 'jarvis':
        pattern = [
            (1, 0, 7/48), (2, 0, 5/48),
            (-2,1,3/48), (-1,1,5/48), (0,1,7/48), (1,1,5/48), (2,1,3/48),
            (-2,2,1/48), (-1,2,3/48), (0,2,5/48), (1,2,3/48), (2,2,1/48)
        ]
    elif dither_diffusion == 'bayer4x4':
        pattern = [(2, 0, 1/16), (3, 0, 1/16), (0, 1, 1/16), (1, 1, 1/16),
                   (2, 2, 1/16), (3, 2, 1/16), (0, 3, 1/16), (1, 3, 1/16)]
    elif dither_diffusion == 'sierra':
        pattern = [
            (1,0,5/32),(2,0,3/32),(-2,1,2/32),(-1,1,4/32),(0,1,5/32),
            (1,1,4/32),(2,1,2/32),(-1,2,2/32),(0,2,3/32),(1,2,2/32)
        ]

    elif dither_diffusion == 'random':
        pattern = []  # stochastic, handled separately
    else:
        pattern = []

    for y in range(h):
        for x in range(w):
            if dither_mode == 'gray':
                r, g, b = arr[x,y]
                old = int((r+g+b)/3)
                if dither_diffusion == 'random':
                    jitter = random.uniform(-step/2, step/2)
                    new = ((old + jitter) // step) * step
                    new = max(0, min(255, int(new)))
                    arr[x,y] = (new, new, new)
                else:
                    new = (old // step) * step
                    quant_error = old - new
                    arr[x,y] = (new, new, new)
                    for dx, dy, factor in pattern:
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < w and 0 <= ny < h:
                            nr, ng, nb = arr[nx, ny]
                            avg = int((nr+ng+nb)/3 + quant_error*factor)
                            avg = max(0, min(255, avg))
                            arr[nx, ny] = (avg, avg, avg)

            elif dither_mode == 'rgb':
                old_r, old_g, old_b = arr[x,y]
                if dither_diffusion == 'random':
                    new_r = ((old_r + random.uniform(-step/2, step/2)) // step) * step
                    new_g = ((old_g + random.uniform(-step/2, step/2)) // step) * step
                    new_b = ((old_b + random.uniform(-step/2, step/2)) // step) * step
                    arr[x,y] = (
                        max(0,min(255,int(new_r))),
                        max(0,min(255,int(new_g))),
                        max(0,min(255,int(new_b)))
                    )
                else:
                    new_r = (old_r // step) * step
                    new_g = (old_g // step) * step
                    new_b = (old_b // step) * step
                    arr[x,y] = (new_r, new_g, new_b)
                    err_r = old_r - new_r
                    err_g = old_g - new_g
                    err_b = old_b - new_b
                    for dx, dy, factor in pattern:
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < w and 0 <= ny < h:
                            nr, ng, nb = arr[nx, ny]
                            nr = max(0, min(255, int(nr + err_r*factor)))
                            ng = max(0, min(255, int(ng + err_g*factor)))
                            nb = max(0, min(255, int(nb + err_b*factor)))
                            arr[nx, ny] = (nr, ng, nb)

    return img


MODE_RATIO_MAP = {
    "color": 0.5,
    "ascii": 0.5,
    "grey": 0.5,
    "gray": 0.5,
    "bw": 0.5,
    "256": 0.5,
    "16": 0.5,
    "8": 0.5,
    "half_block": 1.0,
    "half": 1.0,
    "braille": 0.25,
}

def render_image(img, mode="color", char=BLOCK_CHAR, new_width=None, new_height=None):
    image_text = []
    
    if mode == "ascii":
        # ASCII grayscale
        for y in range(img.height):
            for x in range(img.width):
                r, g, b = img.getpixel((x, y))
                image_text.append(pixel_to_ascii(r, g, b))
            image_text.append('\n')

    elif mode == "braille":
        # Convert to grayscale for intensity threshold
        img_gray = img.convert('L')
        # Resize to multiple of 2x4
        new_width = (new_width // 2) * 2
        new_height = (new_height // 4) * 4
        img_gray = img_gray.resize((new_width, new_height))

        threshold = 128
        for y in range(0, img_gray.height, 4):
            for x in range(0, img_gray.width, 2):
                dots = []
                for dy in range(4):
                    for dx in range(2):
                        px = img_gray.getpixel((x + dx, y + dy))
                        dots.append(1 if px < threshold else 0)
                image_text.append(chr(0x2800 + (dots[0] << 6) + (dots[1] << 4) + (dots[2] << 2) + dots[3]))
            image_text.append('\n')

    elif mode == "half":
        new_height = (new_height // 2) * 4
        img = img.resize((new_width, new_height))
        for y in range(0, img.height - 1, 2):
            for x in range(img.width - (img.width % 1)):  # or just leave as is
                top = img.getpixel((x, y))
                bottom = img.getpixel((x, y + 1))
                image_text.append(rgb_to_ansi_fg_bg(top, bottom))
            image_text.append('\n')

    elif mode == "grey":
        for y in range(img.height):
            for x in range(img.width):
                r, g, b = img.getpixel((x, y))
                image_text.append(f"{rgb_to_ansi16_grey(r, g, b)}{char}\x1b[0m")
            image_text.append('\n')

    elif mode == "256":
        for y in range(img.height):
            for x in range(img.width):
                r, g, b = img.getpixel((x, y))
                image_text.append(f"{rgb_to_ansi256(r, g, b)}{char}\x1b[0m")
            image_text.append('\n')

    elif mode == "16":
        for y in range(img.height):
            for x in range(img.width):
                r, g, b = img.getpixel((x, y))
                image_text.append(f"{rgb_to_ansi16(r, g, b)}{char}\x1b[0m")
            image_text.append('\n')

    elif mode == "8":
        for y in range(img.height):
            for x in range(img.width):
                r, g, b = img.getpixel((x, y))
                image_text.append(f"{rgb_to_ansi_8(r, g, b)}{char}\x1b[0m")
            image_text.append('\n')
    elif mode == "bw":
        # Image is already dithered to 1-bit in preprocessing above
        for y in range(img.height):
            for x in range(img.width):
                pixel = img.getpixel((x, y))
                is_black = pixel == 0 if isinstance(pixel, int) else sum(pixel) < 128 * 3
                char_out = f"\x1b[30m{char}\x1b[0m" if is_black else f"\x1b[37m{char}\x1b[0m"
                image_text.append(char_out)
            image_text.append('\n')


    else:  # default: truecolor "color" mode
        for y in range(img.height):
            for x in range(img.width):
                r, g, b = img.getpixel((x, y))
                image_text.append(f"\x1b[38;2;{r};{g};{b}m{char}\x1b[0m")
            image_text.append('\n')

    return "".join(image_text)

def show_image(image_path, width=None, height_ratio=0.5, char='█',
                 dither=False, fit_to='both', filters=None, mode='color', resample=Image.NEAREST, dither_method='none'):
    resample = RESAMPLING_MAP.get(resample, Image.NEAREST)
    total_byte_count = 0
    compressed_byte_count = 0

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    if filters:
        img = apply_filters(img, filters)

    term_size = shutil.get_terminal_size(fallback=(80, 24))
    term_width, term_height = term_size.columns, term_size.lines

    eff_ratio = MODE_RATIO_MAP.get(mode, height_ratio)



    if fit_to == "height":
        max_height = term_height - 2
        aspect_ratio = img.height / img.width
        eff_ratio = 1.0 if mode == "half_block" else height_ratio
        new_height = max_height
        new_width = int(new_height / (aspect_ratio * eff_ratio))
        if new_width > term_width:
            new_width = term_width
            new_height = int(new_width * aspect_ratio * eff_ratio)
    elif fit_to == "width":
        if width is None:
            width = term_width
        new_width = width
        eff_ratio = 1.0 if mode == "half_block" else height_ratio
        new_height = int(img.height * (new_width / img.width) * eff_ratio)
    elif fit_to == "both":
        max_height = term_height - 2
        aspect_ratio = img.height / img.width
        eff_ratio = 1.0 if mode == "half_block" else height_ratio
        new_height = max_height
        new_width = int(new_height / (aspect_ratio * eff_ratio))
        if new_width > term_width:
            new_width = term_width
            new_height = int(new_width * aspect_ratio * eff_ratio)



    img = img.resize((new_width, new_height), resample=resample)

    if dither:
        if mode in {"ascii", "gray", "grey", "bw"}:
            dither_mode = 'gray'
        elif mode in {"color", "half", "256", "16", "8"}:
            dither_mode = 'rgb'
        img = apply_dither(img, dither_diffusion=dither_method, dither_mode=dither_mode)

    
    # get image
    image_text = render_image(
        img, 
        mode=mode, 
        char=char,
        new_height=new_height,
        new_width=new_width
    )

    # calculate byte counts
    total_byte_count = len(image_text.encode('utf-8'))
    compressed_image_text = zlib.compress(image_text.encode('utf-8'))
    compressed_byte_count = len(compressed_image_text)
    print(f"Total bytes: {humanize.naturalsize(total_byte_count)}, Compressed bytes: {humanize.naturalsize(compressed_byte_count)}")

    # print image
    sys.stdout.write(
        image_text
    )

    
def strip_quotes(s):
    """ Strip quotes from a string. """
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    elif s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render an image in the terminal using ANSI or ASCII.", epilog="Use Exiftool to extract metadata from images: https://exiftool.org/quickref.html")
    parser.add_argument("image", nargs="?", default=None, help="Path to the image")
    parser.add_argument("-w", "--width", type=int, help="Max width in characters (default: terminal width)")
    parser.add_argument("-r", "--ratio", type=float, default=0.5, help="Height ratio for normal mode (default: 0.5)")
    parser.add_argument("-c", "--char", default="█", help="Character to use for color rendering (default: █)")
    parser.add_argument("-ac", "--ascii-chars", type=str, default=ASCII_CHARS, help="Characters to use for ascii rendering, Black to White (default: '█▓▒░ ')")
    parser.add_argument("-ra", "--reverse-ascii", action="store_true", help="Use reverse ASCII characters for rendering (default: False)")
    parser.add_argument("-f", "--filter", nargs='*', help="Apply filters: brightness=1.2 contrast=1.5 hue=180 sepia grayscale invert")
    parser.add_argument("-d", "--dither", action="store_true", help="Enable dithering (black and white)")
    parser.add_argument(
        "-dm", "--dither-method",
        choices=["none", "floyd", "bayer", "atkinson", "stucki", "jarvis", "sierra", "bayer4x4", "random"],
        default="floyd",
        help="Dithering algorithm: none, floyd, bayer, atkinson, stucki, jarvis, sierra, bayer4x4"
    )
    parser.add_argument("-fw", "--fit-width", action="store_true", help="Fit to terminal width (default: scale to height)")
    parser.add_argument("-fh", "--fit-height", action="store_true", help="Fit to terminal height (default: scale to width)")
    parser.add_argument(
        "-m", "--mode", choices=["color", "ascii", "half", "256", "16", "8", "braille", "bw", "grey"],
        default="color", help="Rendering mode: color (default), ascii, half, 256, 16, 8, braille, grey"
    )
    parser.add_argument("-s", "--resampling", choices=RESAMPLING_MAP.keys(), default="bicubic",
                        help="Resampling method: nearest, bilinear, bicubic, box, hamming, hermite, mitchell, lanczos")


    args = parser.parse_args()

    ASCII_CHARS = args.ascii_chars or "█▓▒░ "
    BLOCK_CHAR = args.char or '█'

    if args.reverse_ascii:
        ASCII_CHARS = ASCII_CHARS[::-1]

    if not args.image:
        try:
            args.image = strip_quotes(input("🖼  Enter image path: ").strip())
            if not args.image:
                exit(1)
        except (EOFError, KeyboardInterrupt):
            exit(1)

    fit_to = "both"
    if args.fit_width:
        fit_to = "width"
    elif args.fit_height:
        fit_to = "height"

    show_image(
        args.image,
        width=args.width,
        height_ratio=args.ratio,
        char=args.char,
        dither=args.dither,
        dither_method=args.dither_method,
        filters=args.filter,
        mode=args.mode,
        fit_to=fit_to,
        resample=args.resampling
    )


