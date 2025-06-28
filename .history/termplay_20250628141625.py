#!/usr/bin/env python3.8
import argparse
import cv2
import os
import sys
import shutil
import time
from PIL import Image
from tqdm import tqdm
import signal
import os
import humanize
import shutil
import subprocess
import pysrt
import webvtt
import re
from html import unescape
import threading
import termios
import tty
import select
import zlib
import json
import textwrap
import pygame
import tempfile
import random

key_pressed = None
audio_start_time = 0.0
audio_paused_at = 0.0
is_audio_paused = False


def get_key():
    """ Continuously read a single key press from stdin. """
    global key_pressed
    while True:
        dr, dw, de = select.select([sys.stdin], [], [], 0.01)
        if dr:
            key = sys.stdin.read(1)
            key_pressed = key


os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

ANSI_16_COLORS = [
    (0, 0, 0),         # 0: black
    (128, 0, 0),       # 1: red
    (0, 128, 0),       # 2: green
    (128, 128, 0),     # 3: yellow
    (0, 0, 128),       # 4: blue
    (128, 0, 128),     # 5: magenta
    (0, 128, 128),     # 6: cyan
    (192, 192, 192),   # 7: white/gray

    (128, 128, 128),   # 8: bright black (dark gray)
    (255, 0, 0),       # 9: bright red
    (0, 255, 0),       # 10: bright green
    (255, 255, 0),     # 11: bright yellow
    (0, 0, 255),       # 12: bright blue
    (255, 0, 255),     # 13: bright magenta
    (0, 255, 255),     # 14: bright cyan
    (255, 255, 255),   # 15: bright white
]


RESAMPLING_MAP = {
    "none": None,
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "box": Image.BOX,
    "hamming": Image.HAMMING
}

ASCII_CHARS = "@%#*+=-:. "
BLOCK_CHAR = '█'

def naturalsize_bits(bits, precision: int = 2, binary: bool = False, gnu: bool = False, format = None) -> str:
	"""
	Converts a given number of bits into a human-readable string with appropriate units.

	This function formats the input `bits` into a string representation with a specified
	level of precision and unit system. It supports both decimal (base 1000) and binary
	(base 1024) unit systems, and can also output GNU-style units.

	Parameters:
		bits (int|float|str): The number of bits to be converted.
		precision (int): The number of decimal places to include in the output. Default is 2.
		binary (bool): Whether to use binary (base 1024) units. Default is False.
		gnu (bool): Whether to use GNU-style units. Default is False.
		format (str): The format string to use for the output. Default is None.

	Returns:
		str: A string representing the formatted size in the appropriate unit.
	"""

	prefixes = ['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y', 'R', 'Q']
	units = [p + ('i' if binary else '') + ('b' if gnu else 'bits') for p in prefixes]
	base = 1024 if binary else 1000
	if isinstance(bits, str):
		bits = float(bits)
	elif bits == None:
		bits = 0

	if bits == 0:
		return '0 ' + units[0]

	i = 0
	while abs(bits) >= base and i < len(units) - 1:
		bits /= base
		i += 1

	if format is None:
		return f"{bits:.{precision}f} {units[i]}"
	else:
		return format % bits + ' ' + units[i]


def supports_truecolor():
    return os.environ.get("COLORTERM") in ("truecolor", "24bit")


term_width = shutil.get_terminal_size().columns
def handle_resize(signum, frame):
    global term_width
    term_width = shutil.get_terminal_size().columns

if os.name != 'nt':
    signal.signal(signal.SIGWINCH, handle_resize)
else:
    # Windows does not support SIGWINCH, so we can't handle terminal resize events
    # We will just use the initial terminal size
    term_width = shutil.get_terminal_size().columns

def format_time(seconds):
    """ Format seconds into MM:SS format. """
    minutes = int(seconds) // 60
    sec = int(seconds) % 60
    return f"{minutes:02}:{sec:02}"

def console_title(title):
    """ Set the terminal title to the given string. """
    if os.name == 'nt':
        os.system(f"title {title}")
    else:
        sys.stdout.write(f"{chr(27)}]0;{title}{chr(7)}")
        sys.stdout.flush()

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
    return f"\x1b[{30 + best_index}m{char}\x1b[0m"


def rgb_to_ansi_256(r, g, b, char=BLOCK_CHAR):
    """ Convert RGB to ANSI 256 color escape code. """
    def rgb_to_ansi_index(r, g, b):
        if r == g == b:
            if r < 8:
                return 16
            if r > 248:
                return 231
            return int(((r - 8) / 247) * 24) + 232
        return 16 + (36 * int(r / 51)) + (6 * int(g / 51)) + int(b / 51)
    idx = rgb_to_ansi_index(r, g, b)
    return f"\x1b[38;5;{idx}m{char}\x1b[0m"

def rgb_to_ansi16_grey(r, g, b, char=BLOCK_CHAR):
    # Map average brightness to nearest ANSI 16 grayscale step
    gray = int((r + g + b) / 3)
    if gray < 32:
        code = 30  # black
    elif gray < 96:
        code = 90  # bright black / dark gray
    elif gray < 160:
        code = 37  # white-ish grey
    elif gray < 224:
        code = 97  # bright white
    else:
        code = 97  # also bright white
    return f"\x1b[{code}m{char}\x1b[0m"


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


def rgb_to_bw(r, g, b, char=BLOCK_CHAR):
    """ Convert RGB to black/white based on luminance. """
    gray = (r + g + b) / 3
    return char if gray > 128 else ' '

def rgb_to_ansi(r, g, b, char=BLOCK_CHAR):
    """ Convert RGB to ANSI escape code for true color. """
    return f"\x1b[38;2;{r};{g};{b}m{char}\x1b[0m"

def rgb_to_ansi256_index(r, g, b):
    """ Convert RGB to ANSI 256 color index (0-255). """
    if r == g == b:
        if r < 8: return 16
        if r > 248: return 231
        return int(((r - 8) / 247) * 24) + 232
    return 16 + (36 * int(r / 51)) + (6 * int(g / 51)) + int(b / 51)

def rgb_to_ansi_general(r, g, b, mode='color', char='█', legacy=False):
    """
    Map an RGB triplet to ANSI escape sequence based on selected mode.
    mode: 'true', '256', '16', '8', 'grey'
    """
    if mode == 'color':
        return f"\x1b[38;2;{r};{g};{b}m{char}\x1b[0m"

    elif mode == '256':
        idx = rgb_to_ansi256_index(r, g, b)
        return f"\x1b[38;5;{idx}m{char}\x1b[0m"

    elif mode == '16':
        return rgb_to_ansi_16(r, g, b, char)

    elif mode == '8':
        return rgb_to_ansi_8(r, g, b, char)

    elif mode == 'grey':
        return rgb_to_ansi16_grey(r, g, b, char)

    else:
        return char  # fallback


def pixel_to_ascii(r, g, b):
    """ Convert RGB pixel to ASCII character based on grayscale value. """
    gray = int((r + g + b) / 3)
    return ASCII_CHARS[gray * len(ASCII_CHARS) // 256]

def dither_image(
        img, 
        dither_levels=4, 
        dither_diffusion='floyd', 
        dither_mode='gray'
    ):
    """
    General dithering function.
    - levels: number of intensity levels
    - diffusion: 'floyd', 'atkinson', or None
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
        # Bayer matrix for 2x2 dithering
        pattern = [(1, 0, 0.25), (0, 1, 0.25), (1, 1, 0.25), (0, 0, 0.25)]
        # Scale to larger matrices as needed
        if dither_levels > 2:
            pattern = [(dx * (dither_levels // 2), dy * (dither_levels // 2), factor) for dx, dy, factor in pattern]
    elif dither_diffusion == 'stucki':
        # Stucki dithering pattern
        pattern = [(1, 0, 8/42), (-2, 1, 4/42), (1, 1, 2/42), (2, 1, 1/42), (0, 1, 4/42), (0, 2, 2/42)]
        # Scale to larger matrices as needed
        if dither_levels > 2:
            pattern = [(dx * (dither_levels // 2), dy * (dither_levels // 2), factor) for dx, dy, factor in pattern]
    elif dither_diffusion == 'jarvis':
        # Jarvis dithering pattern
        pattern = [
            (1, 0, 7/48), (2, 0, 5/48),
            (-2,1,3/48), (-1,1,5/48), (0,1,7/48), (1,1,5/48), (2,1,3/48),
            (-2,2,1/48), (-1,2,3/48), (0,2,5/48), (1,2,3/48), (2,2,1/48)
        ]

        # Scale to larger matrices as needed
        if dither_levels > 2:
            pattern = [(dx * (dither_levels // 2), dy * (dither_levels // 2), factor) for dx, dy, factor in pattern]
    elif dither_diffusion == 'bayer4x4':
        # Bayer 4x4 dithering pattern
        bayer4x4 = [
            [ 0/16,  8/16,  2/16, 10/16],
            [12/16,  4/16, 14/16,  6/16],
            [ 3/16, 11/16,  1/16,  9/16],
            [15/16,  7/16, 13/16,  5/16],
        ]

        # Scale to larger matrices as needed

    elif dither_diffusion == 'sierra':
        # Sierra dithering pattern
        pattern = [
            (1,0,5/32),(2,0,3/32),(-2,1,2/32),(-1,1,4/32),(0,1,5/32),
            (1,1,4/32),(2,1,2/32),(-1,2,2/32),(0,2,3/32),(1,2,2/32)
        ]
        # Scale to larger matrices as needed
        if dither_levels > 2:
            pattern = [(dx * (dither_levels // 2), dy * (dither_levels // 2), factor) for dx, dy, factor in pattern]
    elif dither_diffusion == 'none':
        pattern = []
    elif dither_diffusion == 'random':
        pattern = []  # no error diffusion, just stochastic threshold
    else:
        pattern = []

    for y in range(h):
        for x in range(w):
            if dither_mode == 'gray':
                r, g, b = arr[x,y]
                old = int((r+g+b)/3)
                if dither_diffusion == 'bayer4x4':
                    threshold = bayer4x4[y % 4][x % 4]
                    r, g, b = arr[x,y]
                    old = int((r+g+b)/3)
                    normalized = old / 255
                    base = int((normalized + threshold / dither_levels) * dither_levels)
                    new = max(0, min(255, base * step))
                    arr[x,y] = (new, new, new)
                elif dither_diffusion == 'random':
                    jitter = random.uniform(-step/2, step/2)
                    new = ((old + jitter) // step) * step
                    arr[x,y] = (int(new), int(new), int(new))
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
                if dither_diffusion == 'bayer4x4':
                    threshold = bayer4x4[y % 4][x % 4]
                    old_r, old_g, old_b = arr[x,y]
                    nr = int((old_r / 255 + threshold / dither_levels) * dither_levels)
                    ng = int((old_g / 255 + threshold / dither_levels) * dither_levels)
                    nb = int((old_b / 255 + threshold / dither_levels) * dither_levels)
                    new_r = max(0, min(255, nr * step))
                    new_g = max(0, min(255, ng * step))
                    new_b = max(0, min(255, nb * step))
                    arr[x,y] = (new_r, new_g, new_b)
                elif dither_diffusion == 'random':
                    jitter_r = random.uniform(-step/2, step/2)
                    jitter_g = random.uniform(-step/2, step/2)
                    jitter_b = random.uniform(-step/2, step/2)
                    new_r = ((old_r + jitter_r) // step) * step
                    new_g = ((old_g + jitter_g) // step) * step
                    new_b = ((old_b + jitter_b) // step) * step
                    arr[x,y] = (int(new_r), int(new_g), int(new_b))
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



def render_braille(img, dither=False):
    """ Render an image using Braille characters, where each character represents a 2x4 pixel block. """
    if dither:
        img = dither_image(img, levels=2, diffusion='floyd', mode='gray')


    lines = []
    for y in range(0, img.height - img.height % 4, 4):
        line = ''
        for x in range(0, img.width - img.width % 2, 2):
            dots = 0
            for dy in range(4):
                for dx in range(2):
                    r, g, b = img.getpixel((x + dx, y + dy))
                    lum = (r + g + b) / 3
                    if lum > 127:
                        braille_map = {
                            (0, 0): 0, (1, 0): 1, (2, 0): 2, (3, 0): 6,
                            (0, 1): 3, (1, 1): 4, (2, 1): 5, (3, 1): 7,
                        }
                        dots |= 1 << braille_map[(dy, dx)]
            char = chr(0x2800 + dots)
            line += char
        lines.append(line)
    return "\n".join(lines)



def extract_audio(video_path):
    temp_audio = tempfile.NamedTemporaryFile(delete=True, suffix=".wav")
    audio_path = temp_audio.name
    temp_audio.close()
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_u8", "-ar", "24000", "-ac", "1",
        "-loglevel", "quiet", # suppress output
        audio_path
    ], check=True)
    return audio_path

def init_audio(audio_path):
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)

def play_audio(start_sec=0.0):
    pygame.mixer.music.play(start=start_sec)

def pause_audio():
    global is_audio_paused, audio_paused_at
    audio_paused_at = get_audio_pos()
    pygame.mixer.music.pause()
    is_audio_paused = True

def resume_audio(current_frame_index, input_fps):
    global is_audio_paused
    if is_audio_paused:
        resume_time = current_frame_index / input_fps
        pygame.mixer.music.stop()
        pygame.mixer.music.play(start=resume_time)
        is_audio_paused = False

def stop_audio():
    pygame.mixer.music.stop()

def get_audio_pos():
    return pygame.mixer.music.get_pos() / 1000.0  # milliseconds → seconds



def resize_frame(frame, width=None, height_ratio=0.5, double_row=False, resample=Image.BICUBIC):
    """ Resize a frame to fit terminal width, maintaining aspect ratio."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(frame)

    if resample is None:
        return pil  # skip resizing

    w = width or term_width
    w = max(20, min(w, term_width - 1))  # clamp width between 20 and usable terminal width

    ratio = height_ratio * (2.0 if double_row else 1.0)
    h = int(pil.height * (w / pil.width) * ratio)

    if double_row and h % 2 != 0:
        h += 1  # make sure height is even for clean top/bottom split

    pil = pil.resize((w, h), resample=resample)

    if double_row and pil.height % 2 != 0:
        pil = pil.crop((0, 0, pil.width, pil.height - 1))  # drop last row to make height even

    return pil



def render_half_block(img, char="▀"):
    """ Render an image using half-block characters, where each pixel is represented by two rows of pixels."""
    lines = []
    for y in range(0, img.height - 1, 2):
        line = ''
        for x in range(img.width - (img.width % 1)):  # or just leave as is

            top = img.getpixel((x, y))
            bottom = img.getpixel((x, y + 1))

            r1, g1, b1 = top
            r2, g2, b2 = bottom

            line += f"\x1b[48;2;{r2};{g2};{b2}m\x1b[38;2;{r1};{g1};{b1}m{char}\x1b[0m"
        lines.append(line)
    return "\n".join(lines)



def render_frame(
        img, 
        mode="color", 
        char=BLOCK_CHAR, 
        dither=False, 
        legacy=False, 
        dither_levels=4, 
        dither_diffusion="gray",
        dither_mode="floyd"
    ):

    """ Render a single frame of an image in the specified mode. """
    if dither:
        if mode in {"ascii", "gray", "grey"}:
            img = dither_image(img, dither_levels=4, dither_diffusion=args.dither_method, dither_mode='gray')
        elif mode in {"color", "256", "16", "8"}:
            img = dither_image(img, dither_levels=8, dither_diffusion=args.dither_method, dither_mode='rgb')
        elif mode == "bw" or mode == "braille":
            img = dither_image(img, dither_levels=2, dither_diffusion=args.dither_method, dither_mode='gray')

    elif mode == "half":
        return render_half_block(img)

    lines = []
    for y in range(img.height):
        line = ''
        for x in range(img.width):
            r, g, b = img.getpixel((x, y))
            if mode == "ascii":
                line += pixel_to_ascii(r, g, b)
            elif mode == "gray":
                gval = int((r + g + b) / 3)
                line += f"\x1b[38;2;{gval};{gval};{gval}m{char}\x1b[0m"
            elif mode == "bw":
                line += rgb_to_bw(r, g, b, char)
            else:
                line += rgb_to_ansi_general(r, g, b, mode=mode, char=char, legacy=args.legacy)
        lines.append(line)

    return "\n".join(lines)

def get_subtitle_at(subtitles, current_time):
    """ Get the subtitle text for the current time. """
    for start, end, text in subtitles:
        if start <= current_time <= end:
            return text
    return ""

def parse_subtitle_html(text):
    """
    Convert some HTML tags to terminal escape sequences, and remove others
    """
    text = unescape(text)

    # Convert <u>...</u> to terminal underline
    def underline_replacer(match):
        return f"\x1b[4m{match.group(1)}\x1b[0m"

    # Convert underline tags
    text = re.sub(r'<u>(.*?)</u>', underline_replacer, text, flags=re.IGNORECASE)

    # Convert <i>...</i> to terminal italic
    def italic_replacer(match):
        return f"\x1b[3m{match.group(1)}\x1b[0m"

    # Convert italic tags
    text = re.sub(r'<i>(.*?)</i>', italic_replacer, text, flags=re.IGNORECASE)

    # Convert <b>...</b> to terminal bold
    def bold_replacer(match):
        return f"\x1b[1m{match.group(1)}\x1b[0m"

    # Convert bold tags
    text = re.sub(r'<b>(.*?)</b>', bold_replacer, text, flags=re.IGNORECASE)

    # Remove all other tags
    text = re.sub(r'</?[^u][^>]*>', '', text)

    return text.strip()

ansi_escape_re = re.compile(r'\x1b\[[0-9;]*[mK]')

def strip_ansi(text):
    """ Strip ANSI escape codes from a string. """
    return ansi_escape_re.sub('', text)

def ansi_center(line, width):
    """ Center a line of text in a given width, stripping ANSI codes. """
    stripped_len = len(strip_ansi(line))
    total_padding = max(0, width - stripped_len)
    left_pad = total_padding // 2
    right_pad = total_padding - left_pad
    return " " * left_pad + line + " " * right_pad

def render_subtitle_line(text, width, max_lines=1, ascii=False):
    """ Render a subtitle line with a frame around it. """

    if ascii:
        box_chars = ['+', '-', '+', '|', '+', '+']
    else:
        box_chars = ['┌', '─', '┐', '│', '└', '┘']
        

    max_content_width = width - 4
    lines = textwrap.wrap(text.strip(), max_content_width) if text else []
    lines = lines[:max_lines]

    # Pad to always have `max_lines`
    while len(lines) < max_lines:
        lines.append("")

    top = box_chars[0] + box_chars[1] * (max_content_width + 2) + box_chars[2]
    framed_lines = [top]

    for line in lines:
        framed_line = box_chars[3] + " " + ansi_center(line, max_content_width) + " " + box_chars[3]
        framed_lines.append(framed_line)

    bottom = box_chars[4] + box_chars[1] * (max_content_width + 2) + box_chars[5]
    framed_lines.append(bottom)

    return "\n".join(framed_lines)


def play_video(
        path, 
        width=None, 
        ratio=0.5, 
        char=BLOCK_CHAR, 
        mode='color', 
        fps_limit=None, 
        dither=False, 
        resample_filter='bicubic', 
        audio=False, 
        no_progress=False, 
        no_cursor=False, 
        subtitle=None,
        dither_levels=4,
        dither_diffusion="gray",
        dither_mode="floyd"

    ):
    """ Play a video in the terminal with various rendering modes. """
    # Open video
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        if not os.path.exists(path):
            print(f"❌ Error: Video not found: {path}")
        elif not os.access(path, os.R_OK):
            print(f"❌ Error: Unable to read video: {path}")
        else:
            print("❌ Error: Unable to open video.")
        return

    # Reset audio variables
    global audio_start_time, audio_paused_at, is_audio_paused
    paused = False
    audio_start_time = 0.0
    audio_paused_at = 0.0
    is_audio_paused = False


    # Get video properties
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    target_fps = fps_limit or input_fps
    frame_skip = max(int(round(input_fps / target_fps)), 1)

    if shutil.which("ffmpeg") is None:
        print("❌ FFmpeg not found. Please install ffmpeg to enable audio.")
        audio = False


    video_name = os.path.basename(path)

    # Load subtitles
    subtitles = []
    if subtitle:
        if os.path.isfile(subtitle):
            if subtitle.endswith(".srt"):
                subs = pysrt.open(subtitle)
                subtitles = [(s.start.ordinal / 1000, s.end.ordinal / 1000, parse_subtitle_html(s.text)) for s in subs]
            elif subtitle.endswith(".vtt"):
                vtt = webvtt.read(subtitle)
                def ts_to_sec(ts):  # 00:00:01.100
                    h, m, s = map(float, ts.replace(',', '.').split(':'))
                    return h * 3600 + m * 60 + s
                subtitles = [(ts_to_sec(c.start), ts_to_sec(c.end), parse_subtitle_html(c.text)) for c in vtt]

            else:
                print("❌ Unsupported subtitle format.")
            # save to json file (for debugging)
            with open("debug_subtitles.json", "w") as f:
                json.dump(subtitles, f, indent=2)
        else:
            print("❌ Subtitle file not found.")
            subtitle = False

    if mode == "half":
        ratio *= 1.0  # each terminal row is 2 pixels high
    elif mode == "braille":
        ratio *= 2.0  # each row represents 4 pixels

    # Progress bar
    if not no_progress:
        progress = tqdm(total=total, desc="📽 Rendering", unit="f", dynamic_ncols=True, position=0, ascii=(mode in {"ascii", "bw"}), leave=True)

    bitrate_mode = "uncompressed"
    bitrate_str = "🔌 0 b"

    # Progress bar variables - calculate bytes per second
    total_bytes = 0
    last_report_time = time.time()
    last_bytes = 0
    rate_bps = 0

    # Compressed bytes per second
    total_compressed_bytes = 0
    compressed_bytes = 0
    last_compressed = 0
    comp_rate_bps = 0


    # Enable raw mode
    orig_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)
    key_thread = threading.Thread(target=get_key, daemon=True)
    key_thread.start()


    try:
        frame_index = 0
        video_name = os.path.basename(path)
        if no_cursor:
            sys.stdout.write("\x1b[?25l")  # hide cursor

        if audio:
            print("🔊 Extracting audio...")
            audio_path = extract_audio(path)
            init_audio(audio_path)
            audio_start_time = 0.0
            play_audio(start_sec=audio_start_time)


        start_time = time.time()

        # Keyboard input handling
        paused = False
        while True:
            global key_pressed
            if key_pressed:
                seek_offset = 10  # seconds
                if key_pressed == ' ':
                    paused = not paused
                    if audio:
                        if paused:
                            stop_audio()

                elif key_pressed == '\x1b':  # ESC → arrow
                    seq = sys.stdin.read(2)
                    if seq == '[D':  # ← left
                        frame_index = max(0, frame_index - int(seek_offset * input_fps))
                    elif seq == '[C':  # → right
                        frame_index = min(total - 1, frame_index + int(seek_offset * input_fps))

                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

                    # Reset video timing
                    start_time = time.time() - (frame_index / input_fps)

                    # Restart/resync audio
                    if audio:
                        stop_audio()
                        play_audio(start_sec=frame_index / input_fps)

                    # Clear terminal to avoid stuck frame
                    sys.stdout.write("\x1b[2J\x1b[H")
                    sys.stdout.flush()
                elif key_pressed == 'q':
                    # end thread
                    print("\n⏹ Stopping...")
                    break
                elif key_pressed == 'b':
                    # Toggle bitrate display
                    if bitrate_mode == 'uncompressed':
                        bitrate_mode = 'compressed'
                        bitrate_str = f"📡 {naturalsize_bits(comp_rate_bps, gnu=True)}"
                    else:
                        bitrate_mode = 'uncompressed'
                        bitrate_str = f"🔌 {naturalsize_bits(rate_bps, gnu=True)}"
                elif key_pressed == 'r':
                    # Reset video playback
                    frame_index = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    start_time = time.time()
                    if audio:
                        stop_audio()
                        play_audio(start_sec=0.0)
                elif key_pressed == 'h':
                    # Show help
                    print("\nControls:")
                    print("  [<-]    Seek left")
                    print("  [->]    Seek right")
                    print("  [SPACE] Pause/Resume")
                    print("  [Q]     Exit")
                    print("  [B]     Toggle bitrate display")
                    print("  [R]     Reset video playback")
                    print("  [H]     Show this help")
                    key_pressed = None


                key_pressed = None


            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_skip != 0:
                frame_index += 1
                continue

            safe_width = shutil.get_terminal_size().columns - 2  # or -2 to be extra safe
            if width is None:
                width = safe_width




            # Terminal title
            current_time = frame_index / input_fps
            console_title(f"▶ [{current_time:.2f}s] {video_name}")
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / input_fps

            # Progress bar
            if not no_progress:
                progress.set_description(f"📽 Rendering... [⏱ {format_time(current_time)}/{format_time(duration)}] {bitrate_str}ps")



            # Cursor to line 2
            if not no_progress:
                progress.refresh()
                sys.stdout.write("\x1b[2;0H")
                sys.stdout.flush()
            else:
                sys.stdout.write("\x1b[1;0H")
                sys.stdout.flush()
            # Resize frame
            img = resize_frame(
                frame,
                width=width or safe_width,  # ✅ use the passed --width value
                height_ratio=ratio,
                double_row=(mode == "half"),
                resample=RESAMPLING_MAP[resample_filter],
            )
            # Get frame text
            frame_text = render_frame(
                img, 
                mode=mode, 
                char=char, 
                dither=args.dither, 
                legacy=args.legacy,
                dither_levels=dither_levels,
                dither_diffusion=dither_diffusion,
                dither_mode=dither_mode
            ) + "\n"

            # Render subtitle
            if subtitle:
                subtitle_text = get_subtitle_at(subtitles, current_time)
                if subtitle_text:
                    frame_text += render_subtitle_line(subtitle_text, width or term_width, ascii=mode=="ascii") + "\x1b[0m\n"
                else:
                    frame_text += "" + render_subtitle_line(" ", width or term_width, ascii=mode=="ascii") + "\n"


            # Get frame bytes
            frame_bytes = frame_text.encode("utf-8")
            sys.stdout.write(frame_bytes.decode("utf-8"))
            sys.stdout.flush()

            # Track byte size
            total_bytes += len(frame_bytes)

            # Track compressed bytes
            compressed_frame = zlib.compress(frame_bytes)
            compressed_bytes += len(compressed_frame)

            total_compressed_bytes += len(compressed_frame)

            if not no_progress:
                if bitrate_mode == 'uncompressed':
                    progress.set_postfix({"size": humanize.naturalsize(total_bytes)})
                elif bitrate_mode == 'compressed':
                    progress.set_postfix({"size": humanize.naturalsize(total_compressed_bytes)})

            while paused:
                console_title(f"⏸ [{current_time:.2f}s] {video_name}")
                time.sleep(0.1)
                progress.set_description(f"📽 Paused... [⏱ {format_time(current_time)}/{format_time(duration)}] {bitrate_str}ps")
                if key_pressed == ' ':
                    if audio:
                        play_audio(start_sec=frame_index / input_fps)
                    paused = False
                    key_pressed = None


            now = time.time()
            if now - last_report_time >= 1.0:
                # Report uncompressed bytes/sec
                delta_bytes = total_bytes - last_bytes
                rate_bps = delta_bytes * 8  # convert to bits/sec
                last_bytes = total_bytes
                last_report_time = now

                # Report compressed bytes/sec
                delta_compressed = compressed_bytes - last_compressed
                comp_rate_bps = delta_compressed * 8
                last_compressed = compressed_bytes

                if bitrate_mode == 'uncompressed':
                    bitrate_str = f"🔌 {naturalsize_bits(rate_bps, gnu=True)}"
                elif bitrate_mode == 'compressed':
                    bitrate_str = f"📡 {naturalsize_bits(comp_rate_bps, gnu=True)}"


            elapsed = time.time() - start_time
            target_time = frame_index / input_fps
            delay = target_time - elapsed
            if delay > 0:
                time.sleep(delay)


            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if not no_progress:
                progress.n = frame_index
                progress.refresh()



    except KeyboardInterrupt:
        # Re-show cursor
        sys.stdout.write("\x1b[?25h")
        # Restore terminal  
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)
        print(f"\n\x1b[0m;🛑 Interrupted by user.")
        exit(0)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)

        if audio:
            stop_audio()


        cap.release()
        if not no_progress:
            progress.close()

        if args.no_cursor:
            sys.stdout.write("\x1b[?25h")  # show cursor
            sys.stdout.flush()

def strip_quotes(s):
    """ Strip quotes from a string. """
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    elif s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"🎞 {os.path.basename(sys.argv[0])}: Render video frames in terminal. When using SSH Connection, use the `-C` option to enable compression, which is recommended. (e.g. `ssh -C user@host 'python3 {os.path.split(sys.argv[0])[1]} video.mp4')",
        epilog="""\
CONTROLS:
  [<-]    Seek 10s left
  [->]    Seek 10s right
  [R]     Reset video playback
  [SPACE] Pause/Resume
  [Q]     Exit
  [B]     Toggle bitrate display (🔌 uncompressed / 📡 compressed)
"""
,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video", nargs="?", type=str, default=None, help="Path to video file")
    parser.add_argument("-w", "--width", type=int, help="Max terminal width in characters")
    parser.add_argument("-r", "--ratio", type=float, default=0.5, help="Character height ratio (default=0.5)")
    parser.add_argument("-c", "--char", default=BLOCK_CHAR, type=str, help="Character to use for rendering (default=█)")
    parser.add_argument("-m", "--mode", choices=["color", "gray", "ascii", "256", "16", "8", "bw", "braille", "half", "grey", "gray"], default="color",
                    help="Rendering mode: color, gray, ascii, 256, 16, bw, braille, half")
    parser.add_argument("-d", "--dither", action="store_true", help="Enable dithering (for ascii, gray, bw, braille modes)")
    parser.add_argument("-nc", "--no-cursor", action="store_true", help="Hide cursor during playback")
    parser.add_argument("-np", "--no-progress", action="store_true", help="Hide tqdm progress bar")
    parser.add_argument("-ac", "--ascii-chars", type=str, default=ASCII_CHARS, help="Characters to use for ascii rendering, Black to White (default: '█▓▒░ ')")
    parser.add_argument("-ra", "--reverse-ascii", action="store_true", help="Use reverse ASCII characters for rendering (default: False)")
    parser.add_argument("-l", "--legacy", action="store_true", help="Enable compatibility mode for legacy terminals (e.g. Windows XP)")
    parser.add_argument("-sub", "--subtitle",  type=str, help="Path to subtitle file (.srt or .vtt). Useful for no audio playback scenarios like SSH.")
    parser.add_argument("-dl", "--dither-levels", type=int, default=4, help="Levels for dithering (default 4)")
    parser.add_argument("-dd", "--dither-mode", choices=["gray", "rgb"], default="gray", help="Dither mode gray or rgb")
    parser.add_argument("-dm", "--dither-method", choices=["floyd", "atkinson", 'bayer', 'stucki', 'jarvis', 'bayer4x4', 'sierra', 'random', "none"], default="floyd", help="Dither diffusion")

    parser.add_argument("-s", "--resampling", choices=RESAMPLING_MAP.keys(), default="bicubic",
                    help="Resampling method for image resizing (default: bicubic)")
    parser.add_argument("-a", "--audio", action="store_true", help="Enable audio playback using ffplay")
    parser.add_argument("-fps", "--fps", type=float, help="Limit FPS to specific rate")
    parser.add_argument("--loop", action="store_true", help="Loop video")

    args = parser.parse_args()

    if not args.video:
        try:
            args.video = strip_quotes(input("🎞 Enter video path: ").strip())
            if not args.video:
                exit(1)
        except (EOFError, KeyboardInterrupt):
            exit(1)

    ASCII_CHARS = args.ascii_chars or "█▓▒░ "
    BLOCK_CHAR = args.char or '█'

    if args.reverse_ascii:
        ASCII_CHARS = ASCII_CHARS[::-1]

    while True:
        play_video(
            path=args.video,
            width=args.width,
            ratio=args.ratio,
            char=args.char,
            mode=args.mode,
            fps_limit=args.fps,
            dither=args.dither,
            resample_filter=args.resampling,
            audio=args.audio,
            no_progress=args.no_progress,
            no_cursor=args.no_cursor,
            subtitle=args.subtitle,
            dither_levels=args.dither_levels, 
            dither_diffusion=args.dither_method, 
            dither_mode=args.dither_mode
        )
        if not args.loop:
            break

    exit(0)