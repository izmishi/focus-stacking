from PIL import Image, ImageEnhance, ImageOps
import argparse
import os
from pathlib import Path
import numpy as np



def debug_print(message):
	if args.debug:
		print(message)

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


def crop_and_resize(image, left, top, right, bottom, new_size):
	image = image.crop((left, top, right, bottom))
	image = image.resize(new_size, Image.LANCZOS)
	return image

def crop_centre_nth(image, n, offset):
	# Divides the image into an n x n grid and crops into the centre
	# It then offsets the crop by `offset`
	width, height = image.size
	left = width * (n - 1) / (2 * n) + offset[0]
	top = height * (n - 1) / (2 * n) + offset[1]
	right = width * (n + 1) / (2 * n) + offset[0]
	bottom = width * (n + 1) / (2 * n) + offset[1]
	return image.crop((left, top, right, bottom))

def zoom_image(image, scale):
	width, height = image.size
	new_width, new_height = (width * scale, height * scale)
	left = (width - new_width) / 2
	top = (height - new_height) / 2
	right = (width + new_width) / 2
	bottom = (height + new_height) / 2
	return crop_and_resize(image, left, top, right, bottom, image.size)

def difference_score(image1, image2):
	buffer1 = np.asarray(image1.convert('L')).astype('int64')
	buffer2 = np.asarray(image2.convert('L')).astype('int64')
	buffer3 = buffer1 - buffer2
	buffer3 = np.abs(buffer3).astype('uint8')
	score = np.mean(buffer3)
	return score

def exposureScore(image):
	return np.mean([x for row in np.asarray(image.convert('L')) for x in row])

def scale_between(image1, image2):
	diff = difference_score(image1, image2)
	differences = [diff]
	scales = [1]
	step_size = 0.001

	scores = [[], []]

	i = 0

	# for i in np.arange(step_size, step_size * 10, step_size):
	while True:
		if i > 0.1:
			break
		# print(i)
		# Zoom second image 
		score0 = difference_score(image1, zoom_image(image2, 1 - i))
		differences.append(score0)
		scales.append(1 - i)

		# Zoom first image
		score1 = difference_score(zoom_image(image1, 1 - i), image2)
		differences.append(score1)
		scales.append(1 / (1 - i))
		# Break loop and return if the difference increases (we have found the minimum)
		if i > step_size:
			if score0 < score1:
				if scores[0][-1] < score0:
					return 1 - (i - step_size)
			else:
				if scores[1][-1] < score1:
					return 1 / (1 - (i - step_size))


		scores[0].append(score0)
		scores[1].append(score1)
		i += step_size

	# print(differences)
	index_of_min = differences.index(min(differences))
	return scales[index_of_min]


directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
def neighbouring_directions(d):
	j = directions.index(d)
	i = (j - 1) % len(directions)
	k = (j + 1) % len(directions)
	return [directions[x] for x in [i, j, k]]

def shift_between(image1, image2):
	steps = [] # a list containing (difference score, offset vector)

	cropped_image1 = crop_centre_nth(image1, 2, (0,0))
	cropped_image2 = crop_centre_nth(image2, 2, (0,0))
	steps.append((difference_score(cropped_image1, cropped_image2), (0,0)))

	step_candidate_directions = directions
	for i in range(100):
		step_candidates = []
		for direction in step_candidate_directions:
			shift = tuple((sum(x) for x in zip(*[x[1] for x in steps], direction)))
			cropped_image1 = crop_centre_nth(image1, 2, (0,0))
			cropped_image2 = crop_centre_nth(image2, 2, shift)
			s = difference_score(cropped_image1, cropped_image2)
			step_candidates.append((s, direction))

		min_difference = min([x[0] for x in step_candidates])
		if min_difference > steps[-1][0]:
			return [sum(x) for x in zip(*[x[1] for x in steps])]

		index_of_min = [x[0] for x in step_candidates].index(min_difference)
		step_direction = step_candidates[index_of_min][1]

		steps.append(step_candidates[index_of_min])
		step_candidate_directions = neighbouring_directions(step_direction)

	index_of_min = [x[0] for x in steps].index(min([x[0] for x in steps]))
	return tuple((sum(x) for x in zip(*[x[1] for x in steps[0:index_of_min]])))


def difference_vector(point0, point1):
	return (point0[0] - point1[0], point0[1] - point1[1])

def exposure_normalised(images):
	images = images.copy()
	exposures = []
	for image in images:
		exposures.append(exposureScore(image))
	brightest = max(exposures)
	images = [ImageEnhance.Brightness(images[i]).enhance(brightest / exposures[i]) for i in range(len(images))]
	return images

def shift_corrected(images):
	images = images.copy()
	image_size = images[0].size
	shifts_relative_to_first = [(0, 0)]
	for i in range(len(images) - 1):
		shift = shift_between(images[i], images[i + 1])
		cumulative_shift = (shifts_relative_to_first[-1][0] - shift[0], shifts_relative_to_first[-1][1] - shift[1])
		shifts_relative_to_first.append(cumulative_shift)

	crop_min_x = max([x[0] for x in shifts_relative_to_first])
	crop_max_x = image_size[0] + min([x[0] for x in shifts_relative_to_first])
	crop_min_y = max([x[1] for x in shifts_relative_to_first])
	crop_max_y = image_size[1] + min([x[1] for x in shifts_relative_to_first])

	crop_size = (int(crop_max_x - crop_min_x), int(crop_max_y - crop_min_y))

	for i in range(len(images)):
		image = images[i]
		shift = shifts_relative_to_first[i]
		left = crop_min_x - shift[0]
		top = crop_min_y - shift[1]
		right = crop_max_x - shift[0]
		bottom = crop_max_y - shift[1]
		shift_corrected_image = crop_and_resize(image, left, top, right, bottom, crop_size)
		images[i] = shift_corrected_image
	return images

def zoom_corrected(images):
	images = images.copy()
	cumulative_scales = [1]

	for i in range(len(images) - 1):
		scale = scale_between(images[i], images[i + 1])
		cumulative_scales.append(scale * cumulative_scales[-1])

	largest_scale = max(cumulative_scales)
	cumulative_scales = [s / largest_scale for s in cumulative_scales]

	for i in range(len(images)):
		image = images[i]
		scale = cumulative_scales[i]
		zoomed_image = zoom_image(image, scale)
		images[i] = zoomed_image
	return images





if __name__ == "__main__":
	# Run only if not imported
	parser = argparse.ArgumentParser()

	parser.add_argument("-x", "--normalise-exposure", help="normalise the exposures of the photos", action="store_true")
	parser.add_argument("-z", "--correct-zoom", help="correct zooming due to lens breathing", action="store_true")
	parser.add_argument("-s", "--correct-shift", help="correct any lateral shifts", action="store_true")
	parser.add_argument("-a", "--correct-all", help="correct for everything (same as -xzs)", action="store_true")

	parser.add_argument("-d", "--debug", help="print debug logs", action="store_true")

	parser.add_argument('inputpath', metavar="logs-dir", help="the path to the folder containing the images to correct", type=Path, nargs='?', default=os.getcwd())
	parser.add_argument('-o', '--output-path', help="the output folder path", type=output_dir_path_type, nargs='?')

	args = parser.parse_args()

	if not (args.normalise_exposure or args.correct_zoom or args.correct_shift or args.correct_all):
		parser.error('No correction requested. Please provide at least one of -x, -z, -s or -a')

	# Get the folder containing images
	if args.output_path:
		output_file_path = args.output_path
	else:
		output_file_path = os.path.join(args.inputpath.absolute(), "Breathing Corrected")

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

	images = [open_Image_with_orientation(x) for x in image_strings]


	# Normalise exposures
	if args.normalise_exposure or args.correct_all:
		print("Normalising exposures")
		images = exposure_normalised(images)

	# Correct for shifting
	if args.correct_shift or args.correct_all:
		print("Correcting for shifting")
		images = shift_corrected(images)

	# Correct for zoom
	if args.correct_zoom or args.correct_all:
		print("Correcting for zooming")
		images = zoom_corrected(images)


	for i in range(len(images)):
		output = os.path.join(output_file_path, "Corrected Image {:0>3d}.jpg".format(i))
		print("Saving " + output)
		images[i].save(output, quality=100, subsampling=0)