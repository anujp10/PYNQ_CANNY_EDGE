#!/usr/bin/env python
# coding: utf-8

# # Canny Edge
# 


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from pynq import Xlnk, Overlay


canny = Overlay("canny.xclbin")

image_path = "input_image.jpg"
original_image = Image.open(image_path)

canny_edge = canny.canny_1
canny_edge.signature

low_threshold = 0
high_threshold = 255
height = 1920
width = 1080

in_buffer = allocate((height, width), np.uint8)
out_buffer = allocate((height, width), np.uint8)

start_time = time.time()

in_buffer.sync_to_device()
canny_edge.call(in_buffer, out_buffer, height, width, low_threshold, high_threshold)
out_buffer.sync_from_device()

stop_time = time.time()
print("--- %s seconds ---" % (stop_time - start_time))

get_ipython().run_line_magic('xdel', 'in_buffer')
get_ipython().run_line_magic('xdel', 'out_buffer')
canny.free()