from PIL import Image, ImageDraw
import math
import os
import random
import numpy

directory = os.path.dirname(os.path.realpath(__file__))

def combine(im1, im2):
    r = numpy.asarray(im1)
    g = numpy.asarray(im2)
    b = numpy.zeros((180, 320), dtype=numpy.uint8)
    x = numpy.dstack((r, g, b))
    im = Image.fromarray(x, mode='RGB')
    return im

def process_dir(in_dir, out_prefix):
    listing = os.listdir(in_dir)
    listing.sort()
    for i in range(len(listing) - 1):
        print(listing[i], listing[i + 1])
        with Image.open(in_dir + listing[i]) as im1:
            with Image.open(in_dir + listing[i + 1]) as im2:
                im = combine(im1, im2)
                im.save(os.path.join(directory, "../experiment3/training/0/") + out_prefix + ("%06da.png" % i))
                im = combine(im2, im1)
                im.save(os.path.join(directory, "../experiment3/training/0/") + out_prefix + ("%06db.png" % i))

                draw = ImageDraw.Draw(im2)
                [num_ellipses] = random.choices([1, 2, 3], [0.85, 0.13, 0.02])
                for c in range(num_ellipses):
                    color = random.randrange(0, 255)
                    area = random.uniform(50 * 50, (320 * 180) / 2)
                    ratio = random.uniform(0.3, 1.7)
                    height = math.sqrt(area / ratio)
                    width = height * ratio
                    x = random.randrange(0, 320)
                    y = random.randrange(0, 180)
                    draw.ellipse((x - width/2, y - height/2, x + width/2, y + height/2), fill=color)
                im = combine(im1, im2)
                im.save(os.path.join(directory, "../experiment3/training/1/") + out_prefix + ("%06da.png" % i))
                im = combine(im2, im1)
                im.save(os.path.join(directory, "../experiment3/training/1/") + out_prefix + ("%06db.png" % i))


process_dir(os.path.join(directory, "../out_cam2_long/"), "S_")
process_dir(os.path.join(directory, "../out_cam2_long2/"), "T_")

def process_test(in_dir, out_dir, out_prefix):
    listing = os.listdir(in_dir)
    listing.sort()
    for i in range(len(listing) - 1):
        print(listing[i], listing[i + 1])
        with Image.open(in_dir + listing[i]) as im1:
            with Image.open(in_dir + listing[i + 1]) as im2:
                im = combine(im1, im2)
                im.save(os.path.join(directory, "../experiment3/" + out_dir) + out_prefix + ("%06da.png" % i))
                im = combine(im2, im1)
                im.save(os.path.join(directory, "../experiment3/" + out_dir) + out_prefix + ("%06db.png" % i))


process_test(os.path.join(directory, "../out1/"), "test1/0/", "S_")
process_test(os.path.join(directory, "../out3/"), "test1/0/", "T_")
process_test(os.path.join(directory, "../out5/"), "test1/0/", "U_")
process_test(os.path.join(directory, "../out2/"), "test2/1/", "S_")
process_test(os.path.join(directory, "../out4/"), "test2/1/", "T_")
process_test(os.path.join(directory, "../out6/"), "test2/1/", "U_")
