"""
Functions for extracting features and simplifying images, 
reading and writing audio files,
and creating/normalizing/exporting spectrograms.
"""
import numpy as np
from pydub import AudioSegment
import audiosegment
from glob import glob
import cv2

import copy

# generator that yields music files, instead of 
def read_songs(dir, recursive = True, extensions = ["m4a", "mp3"]):
    extensions_str = str(tuple("." + s for s in extensions))
    # load paths to music files recursively
    paths = glob(dir + "/**/*%s"%extensions_str, recursive = True)
    # return a generator for reading the audio files
    return (audiosegment.from_file(f) for f in paths)

# convert a signal (as a numpy array) to an AudioSegment object
def toAudioSegment(arr,framerate):
    seg = AudioSegment(
    arr.tobytes(), 
    frame_rate=framerate,
    sample_width=arr.dtype.itemsize, 
    channels=1
    )
    return seg

# save a numpy array as an audio (mp3) file
def np_to_mp3(arr,framerate,filename):
	# convert to AudioSegment object
	seg = toAudioSegment(arr,framerate)
	seg.export(filename, "mp3")

# extract a spectrogram from an audiosegment object
def get_spect(seg):
    freqs, times, amplitudes = seg.spectrogram(window_length_s=0.03, overlap=.9)
    return freqs, times, amplitudes

# normalize spectrogram so that values are within a certain range
def normalize_spectogram(Sxx):
    Sxx = np.log10(Sxx + 1e-9)
    smin = Sxx.min()
    srange = Sxx.max() - smin
    Sxx = (Sxx - smin)/srange
    return Sxx

def black_overlay(arr,black = 255, depth = 2):
	heigh,width = arr.shape
	tmp = np.repeat(black,height*width*depth).reshape(height,width,depth)
	tmp[:,:,0] = arr
	return tmp

# edge detection
def edge_detect(arr,lowerbound = 0, upperbound = 200,scale = None):
	if scale:
		arr *= scale
	if len(arr.shape) <= 2:
		arr = black_overlay(arr,black = 255, depth = 2)
	arr = arr.astype('uint8')
	# extract edges 
	edges = cv2.Canny(arr,lowerbound, upperbound)
	return edges

# 'all-or-nothing' an image 
def img_binarize(arr,threshold = .5, low = 0, high = 1):
	minval,maxval = arr.min(),arr.max()
	thresh_val = min_value + threshold(maxval - minval)
	# copy for no destructive assignment
	arr_copy = copy.copy(arr)
	# is it black or white?
	black = arr > thresh_val
	white = np.logical_not(black)
	# binarize
	arr_copy[black] = low
	arr_copy[white] = high
	return arr_copy

# apply a Gaussian blur. Especially useful for spectrograms
def gaussian_blur(arr, window = (9,9),depth =0):
	return cv2.GaussianBlur(arr, window, depth)







