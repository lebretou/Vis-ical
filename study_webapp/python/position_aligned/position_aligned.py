import svgwrite
import math
import random
from svgwrite import cm, mm

RATIOS = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
START_A = 3.5
START_B = 6.5


def positions(name, ratio, index):
    random_offset_a = random.uniform(0,2)

    # draw canvas
    img = svgwrite.Drawing(filename=name, size = (12*cm, 6*cm))
    
    # create the rectangles as the axes 
    img.add(img.rect(insert=(3 * cm, random_offset_a * cm), size = (6 *cm, 3*cm), fill="white", stroke="black", stroke_width=3))

    # genrate random numbers and scaled them to the correct values
    num1 = 80
    num2 = num1 * ratio

    coin_flip = random.randint(0, 1) 

    if coin_flip == 0:
        tmp = num1
        num1 = num2
        num2 = tmp

    scaled_a = 3.0 - (num1 * 3.0 / 100)
    scaled_b = 3.0 - (num2 * 3.0 / 100)

    # add the points 
    img.add(img.circle(center=(START_A * cm, (scaled_a + random_offset_a) *cm), r=0.1*cm,fill="black"))
    img.add(img.circle(center=(START_B* cm, (scaled_b + random_offset_a) *cm), r=0.1*cm,fill="black"))

    # add markers
    # img.add(img.text('A', insert=(3.5 * 37.6, (4 + random_offset_a) * 37.6)))
    # img.add(img.text('B', insert=(6.5 * 37.6, (4 + random_offset_a) * 37.6)))

    # add labels 
    img.add(img.text("PA"+str(index), insert=(0.1*cm, 0.5*cm), fill='lightgray'))

    img.save()




if __name__ == '__main__':
    index = 0

    for ratio in RATIOS:
        positions(('position_aligned_' + str(index) + '.svg'), ratio, index)
        index += 1

    # positions('instruction.svg', 0.5, 0)