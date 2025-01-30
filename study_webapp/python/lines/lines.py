import svgwrite
import random
from svgwrite import cm, mm


MAX_LENGTH = 6
BASE_LENGTH = 4.4
RATIOS = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]

def lines(name, ratio, index):
    # create canvas
    img = svgwrite.Drawing(filename=name, size = (12*cm, 6*cm))

    length_a = BASE_LENGTH
    length_b = BASE_LENGTH * ratio

    # 50% chance to swap the position of line a and line b
    coin_flip = random.randint(0, 1)
    if (coin_flip == 0):
        tmp = length_b
        length_b = length_a
        length_a = tmp

    # define loc for lines
    line_a_start, line_a_end = line_coordinates(length_a, (3, 3))
    line_b_start, line_b_end = line_coordinates(length_b, (9, 3))
    line_a_center = (3,3)
    line_b_center = (9,3)



    # draw and rotate line a
    line_a = img.line(start=(line_a_start[0]*cm, line_a_start[1]*cm), end=(line_a_end[0]*cm,line_a_end[1]*cm), stroke='black', stroke_width=4)
    rotate_angle_a = random.randint(-90, 90)

    # draw and rotate line b
    line_b = img.line(start=(line_b_start[0]*cm, line_b_start[1]*cm), end=(line_b_end[0]*cm,line_b_end[1]*cm), stroke='black', stroke_width=4)
    rotate_angle_b = random.randint(-90, 90)
    
    # add markers
    # marker_a = img.text('A', insert=(5.8 * 37.6, 5.8 * 37.6))
    # marker_b = img.text('B', insert=(11.8 * 37.6, 5.8 * 37.6))

    # add labels 
    img.add(img.text('L'+str(index), insert=(0.1*cm, 0.5*cm), fill='lightgray'))

    img.add(line_a)
    img.add(line_b)
    # img.add(marker_a)
    # img.add(marker_b)
    img.save()

# length and center in the unit of cm
def line_coordinates(length, center):
    half = length / 2.0
    start = (center[0], center[1] - half)
    end = (center[0], center[1] + half)
    return start, end



if __name__ == '__main__':
    
    index = 0

    for ratio in RATIOS:
        lines(('lines_' + str(index) + '.svg'), ratio, index)
        index += 1

    # lines("instruction.svg", 0.5, 0)
