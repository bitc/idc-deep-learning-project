from PIL import Image, ImageDraw
import math
import os
import random

directory = os.path.dirname(os.path.realpath(__file__))

# static: 4048 total
# human: 1895 total

for filename in os.listdir("../experiment2/training_1_raw/"):
    print(filename)
    with Image.open("../experiment2/training_1_raw/" + filename) as im:
        draw = ImageDraw.Draw(im)
        [num_ellipses] = random.choices([1, 2, 3], [0.85, 0.13, 0.02])
        for i in range(num_ellipses):
            color = random.randrange(0, 255)
            area = random.uniform(50 * 50, (320 * 180) / 2)
            ratio = random.uniform(0.3, 1.7)
            height = math.sqrt(area / ratio)
            width = height * ratio
            x = random.randrange(0, 320)
            y = random.randrange(0, 180)
            draw.ellipse((x - width/2, y - height/2, x + width/2, y + height/2), fill=color)
        im.save("../experiment2/training/1/" + filename)
