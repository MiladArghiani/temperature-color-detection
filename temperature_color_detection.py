"""
Temperature-Sensitive Color Detection (Colormap)
--------------------------------------------------
Splits an image into its Red, Green, and Blue channels, maps each channel
through a "temperature" colormap (black -> purple -> blue -> green ->
yellow -> red -> white), then combines the channel pairs using a custom
function based on their wavelengths to highlight temperature-sensitive
color regions in the image.

Author: Milad Arghiani
"""

import sys
import subprocess
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def install(name):
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', name])
    except Exception:
        print(f"Please install package '{name}'")


try:
    import cv2  # noqa: F811 (re-check after potential install)
except ModuleNotFoundError:
    install('opencv-python')
    import cv2

try:
    import matplotlib.pyplot as plt  # noqa: F811
except ModuleNotFoundError:
    install('matplotlib')
    import matplotlib.pyplot as plt

try:
    import numpy as np  # noqa: F811
except ModuleNotFoundError:
    install('numpy')
    import numpy as np


def temperature_shift(c2, wavelength1, wavelength2, c1, intensity1, intensity2):
    """Combine two channel intensities based on their relative wavelengths."""
    wavelength_diff = (1 / wavelength1) - (1 / wavelength2)
    numerator = wavelength_diff * c2

    ratio = intensity1 / intensity2
    log_ratio = np.log(ratio)
    denominator = c1 + log_ratio

    return numerator / denominator


# Image path: defaults to a local "sample.jpg" next to this script,
# or set the IMAGE_PATH environment variable to use a different image.
DEFAULT_IMAGE_PATH = os.environ.get(
    "IMAGE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample.jpg"),
)

try:
    image = cv2.imread(DEFAULT_IMAGE_PATH, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
except Exception:
    path = input("Please enter the image path: ")
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.title("The original image")
plt.show()

print("Converting to RGB, because OpenCV reads images in BGR...")
image = rgb_image
plt.imshow(image)
plt.title("The original image in RGB")
plt.show()

print("Splitting the channels into Red, Green, and Blue")
red_channel, green_channel, blue_channel = image[:, :, 0], image[:, :, 1], image[:, :, 2]

plt.figure(figsize=(6, 3))
plt.subplot(1, 3, 1)
plt.imshow(red_channel)
plt.title('Red channel')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(green_channel)
plt.title('Green channel')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(blue_channel)
plt.title('Blue channel')
plt.axis('off')
plt.show()

# Approximate wavelengths (in meters) used to weight each channel pair
RED_WAVELENGTH = 600e-9
BLUE_WAVELENGTH = 500e-9
GREEN_WAVELENGTH = 400e-9

C1 = 1.0
C2 = 1.0

print("Applying the temperature function, this may take a moment...")

colors = ["black", "purple", "blue", "green", "yellow", "red", "white"]
cmap = LinearSegmentedColormap.from_list("temperature_cmap", colors)

red_channel = cmap(red_channel)
blue_channel = cmap(blue_channel)
green_channel = cmap(green_channel)

red_channel = np.uint8(red_channel[:, :, :3])
blue_channel = np.uint8(blue_channel[:, :, :3])
green_channel = np.uint8(green_channel[:, :, :3])

red_hsv = cv2.cvtColor(red_channel, cv2.COLOR_RGB2HSV)
blue_hsv = cv2.cvtColor(blue_channel, cv2.COLOR_RGB2HSV)
green_hsv = cv2.cvtColor(green_channel, cv2.COLOR_RGB2HSV)

hue_red, _, _ = cv2.split(red_hsv)
_, sat_green, _ = cv2.split(green_hsv)
_, sat_blue, _ = cv2.split(blue_hsv)

# Replace zero values with a very small number to avoid division/log errors
hue_red_safe = np.where(hue_red != 0, hue_red, 1e-10)
sat_green_safe = np.where(sat_green != 0, sat_green, 1e-10)
sat_blue_safe = np.where(sat_blue != 0, sat_blue, 1e-10)

RG = temperature_shift(C2, RED_WAVELENGTH, GREEN_WAVELENGTH, C1, hue_red_safe, sat_green_safe)
RB = temperature_shift(C2, RED_WAVELENGTH, BLUE_WAVELENGTH, C1, hue_red_safe, sat_blue_safe)
BG = temperature_shift(C2, BLUE_WAVELENGTH, GREEN_WAVELENGTH, C1, sat_blue_safe, sat_green_safe)

np.set_printoptions(threshold=sys.maxsize)
print(RG)

# Save the Red-Green result next to this script (change path as needed)
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RG_output.txt")
np.savetxt(output_path, RG)
print(f"Saved Red-Green channel result to {output_path}")

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(RG)
plt.title('Red-Green channel after applying the function')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(RB)
plt.title('Red-Blue channel after applying the function')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(BG)
plt.title('Blue-Green channel after applying the function')
plt.axis('off')
plt.show()

RG = RG.reshape((red_channel.shape[0], red_channel.shape[1], 1))
RB = RB.reshape((red_channel.shape[0], red_channel.shape[1], 1))
BG = BG.reshape((red_channel.shape[0], red_channel.shape[1], 1))

merged_image = np.concatenate([RG, RB, BG], axis=2)

# Clip final values to a valid displayable range
merged_image = np.clip(merged_image, 0, 1)

plt.figure(figsize=(8, 6))
plt.imshow(merged_image)
plt.title('Final result')
plt.axis('off')
plt.show()
