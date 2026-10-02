# Temperature-Sensitive Color Detection (Colormap)

Detects temperature-sensitive colors in an image by splitting it into its
Red, Green, and Blue channels, mapping each channel through a custom
"temperature" colormap (black → purple → blue → green → yellow → red →
white), and combining channel pairs using a wavelength-based function to
highlight regions with temperature-related color shifts.

## How it works
1. Load an image and split it into R, G, B channels.
2. Apply a temperature colormap to each channel individually.
3. Convert each mapped channel to HSV and extract the relevant
   component (Hue for red, Saturation for green/blue).
4. Combine channel pairs (Red-Green, Red-Blue, Blue-Green) using a
   function based on each color's approximate wavelength.
5. Merge the results into a final image highlighting temperature-sensitive
   regions.

## Tech stack
Python, OpenCV, NumPy, Matplotlib

## Usage
Run the script and provide the path to an image when prompted
(or set the `IMAGE_PATH` environment variable).
