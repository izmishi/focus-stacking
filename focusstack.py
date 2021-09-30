from PIL import Image, ImageFilter, ImageMath, ImageOps
from math import sqrt, log
import argparse
import os
from pathlib import Path
import numpy as np
import sys

from lens_breathing_correction import exposure_normalised, shift_corrected, zoom_corrected


def open_Image_with_orientation(path):
	return ImageOps.exif_transpose(Image.open(path))


def output_dir_path_type(string):
    if os.path.isdir(string):
        return string
    else:
    	try:
    		os.mkdir(string)
    	except:
    		raise NotADirectoryError(string)


parser = argparse.ArgumentParser()

parser.add_argument("-x", "--normalise-exposure", help="normalise the exposures of the photos", action="store_true")
parser.add_argument("-z", "--correct-zoom", help="correct zooming due to lens breathing", action="store_true")
parser.add_argument("-s", "--correct-shift", help="correct any lateral shifts", action="store_true")
parser.add_argument("-a", "--correct-all", help="correct for everything (same as -xzs)", action="store_true")

parser.add_argument("-d", "--debug", help="print debug logs", action="store_true")
parser.add_argument("-i", "--debug-images", help="save debug images", action = "store_true")

parser.add_argument('inputpath', type=Path, nargs='?', default=os.getcwd())
parser.add_argument('-o', '--output-path', type=output_dir_path_type, nargs='?')
args = parser.parse_args()

# Get the folder containing images
if args.output_path:
	output_file_path = args.output_path
else:
	output_file_path = os.path.join(args.inputpath.absolute(), "Focus Stacked")

if not os.path.isdir(output_file_path):
	try:
		os.mkdir(output_file_path)
	except:
		raise NotADirectoryError(output_file_path)

# Get list of images
image_paths = []
for type in ["*.JPG", "*.jpg", "*.PNG", "*.png"]:
	image_paths.extend(args.inputpath.glob(type))
image_strings = [str(p) for p in image_paths]
image_strings.sort()

colour_images = [open_Image_with_orientation(image_string).convert("RGB") for image_string in image_strings]

# Normalise exposures
if args.normalise_exposure or args.correct_all:
	print("Normalising exposures")
	colour_images = exposure_normalised(colour_images)

# Correct for shifting
if args.correct_shift or args.correct_all:
	print("Correcting for shifting")
	colour_images = shift_corrected(colour_images)

# Correct for zoom
if args.correct_zoom or args.correct_all:
	print("Correcting for zooming")
	colour_images = zoom_corrected(colour_images)

split_images = [image.split() for image in colour_images]
np_images = [np.asarray(image) for image in colour_images]

image_size = colour_images[0].size


smooth_kernel = (
	1, 1, 1, 1, 1,
	1, 1.5,   1.5,   1.5, 1,
	1, 1.5, 4, 1.5, 1,
	1, 1.5,   1.5,   1.5, 1,
	1, 1, 1, 1, 1
)
smooth_filter = ImageFilter.Kernel((5, 5), smooth_kernel)

small_smooth_kernel = (
	1,  1.5,   1,
	1.5,  3, 1.5,
	1,  1.5,   1
	)
small_smooth_filter = ImageFilter.Kernel((3, 3), small_smooth_kernel)


gaussian_blur_filter = ImageFilter.GaussianBlur(7)

small_blur_kernel = (
	1, 2, 1,
	2, 4, 2,
	1, 2, 1
	)


a = 1
b = 1.3
c = 1.6
d = 2
e = -4*a - 8*b - 8*c - 4*d

edge_kernel = [
	a, b, c, b, a,
	b, c, d, c, b,
	c, d, e, d, c,
	b, c, d, c, b,
	a, b, c, b, a,
]
edge_filter = ImageFilter.Kernel((5, 5), edge_kernel, scale = 1)

negative_edge_kernel = [-x for x in edge_kernel]
negative_edge_filter = ImageFilter.Kernel((5, 5), negative_edge_kernel, scale = 1)

small_edge_kernel = [
	-1, -1, -1,
	-1,  8, -1,
	-1, -1, -1	
]
small_edge_filter = ImageFilter.Kernel((3, 3), small_edge_kernel, scale = 1)

negative_small_edge_kernel = [-x for x in small_edge_kernel]
negative_small_edge_filter = ImageFilter.Kernel((3, 3), negative_small_edge_kernel, scale = 1)


edge_images = []

for i in range(len(colour_images)):
	print("\rFinding edges in image {}/{}".format(i+1, len(colour_images)), end="")
	image = colour_images[i]
	smooth_image = image.filter(small_smooth_filter)#.filter(smooth_filter)#.filter(small_smooth_filter).filter(small_smooth_filter)#.filter(small_smooth_filter)
	imageWithEdges = smooth_image.filter(edge_filter).convert("L")
	negImageWithEdges = smooth_image.filter(negative_edge_filter).convert("L")
	imageWithEdges = ImageMath.eval("convert(max(a, b), 'L')", a=imageWithEdges, b=negImageWithEdges)

	temp_image = imageWithEdges
	for j in range(10):
		temp_image = temp_image.filter(ImageFilter.GaussianBlur(0 + 1 * log(1 + 2*j*j)))
		temp_image = ImageOps.autocontrast(temp_image, (1.5, 0))
		# temp_image = Image.blend(imageWithEdges, temp_image, 0.5).convert("L")
		temp_image = ImageMath.eval("convert(max(int(a / c), b), 'L')", a=imageWithEdges, b=temp_image, c=int(1 + j / 5))
	temp_image = temp_image.filter(ImageFilter.GaussianBlur(1))
	imageWithEdges = temp_image

	output = os.path.join(output_file_path, "Image {:0>3d}.jpg".format(i))
	if args.debug_images:
		imageWithEdges.save(output, quality=100, subsampling=0)
	edge_images.append(np.asarray(imageWithEdges))

print()
stacked_image = np_images[0].copy()

for x in range(image_size[0]):
	print("\rStacking images {:.2f}%".format(100 * x / image_size[0]), end="")
	for y in range(image_size[1]):
		edge_values = [(1.5 * len(edge_images) - i) * edge_images[i][y][x] for i in range(len(edge_images))]
		index_of_max = edge_values.index(max(edge_values))
		stacked_image[y][x] = np_images[index_of_max][y][x]

print("\r", end="")
sys.stdout.write("\033[K") # clear to the end of line
sys.stdout.flush()
print("\rStacking images 100.00%")
stacked_image = Image.fromarray(stacked_image)
output = os.path.join(output_file_path, "Stacked Image.jpg")
stacked_image.save(output, quality=100, subsampling=0)