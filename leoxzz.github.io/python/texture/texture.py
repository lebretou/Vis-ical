import svgwrite
import math
import random
from svgwrite import cm, mm

RATIOS = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]

def textures(name, ratio, index):
    # draw canvas
    img = svgwrite.Drawing(filename=name, size = (12*cm, 6*cm))


    pattern = img.pattern(id='pattern', patternUnits='userSpaceOnUse',
                                        size=(10,10), patternTransform="rotate(45 0 0)")
    pattern.add(img.line(start=(0, 0), end=(0, 10), stroke='black', stroke_width=1))

    pattern_2 = img.pattern(id='pattern-2', patternUnits='userSpaceOnUse',
                                        size=(10,10), patternTransform="rotate(-45 0 0)")
    pattern_2.add(img.line(start=(0, 0), end=(0, 10), stroke='black', stroke_width=1))

    pattern_3 = img.pattern(id='pattern-3', patternUnits='userSpaceOnUse',
                                        size=(10 / ratio,10), patternTransform="rotate(45 0 0)")
    pattern_3.add(img.line(start=(0, 0), end=(0, 10), stroke='black', stroke_width=1))

    pattern_4 = img.pattern(id='pattern-4', patternUnits='userSpaceOnUse',
                                        size=(10 /ratio,10), patternTransform="rotate(-45 0 0)")
    pattern_4.add(img.line(start=(0, 0), end=(0, 10), stroke='black', stroke_width=1))

    coin_flip = random.randint(0,1)
    if coin_flip == 0:
        img.add(pattern)
        img.add(img.rect(insert=(1*cm, 1*cm), size = (4*cm, 4*cm), fill='url(#pattern)', stroke_width=3, stroke='black'))


        img.add(pattern_2)
        img.add(img.rect(insert=(1*cm, 1*cm), size = (4*cm, 4*cm), fill='url(#pattern-2)', stroke_width=3, stroke='black'))


        img.add(pattern_3)
        img.add(img.rect(insert=(7*cm, 1*cm), size = (4*cm, 4*cm), fill='url(#pattern-3)', stroke_width=3, stroke='black'))


        img.add(pattern_4)
        img.add(img.rect(insert=(7*cm, 1*cm), size = (4*cm, 4*cm), fill='url(#pattern-4)', stroke_width=3, stroke='black'))
    else:
        img.add(pattern)
        img.add(img.rect(insert=(1*cm, 1*cm), size = (4*cm, 4*cm), fill='url(#pattern-3)', stroke_width=3, stroke='black'))


        img.add(pattern_2)
        img.add(img.rect(insert=(1*cm, 1*cm), size = (4*cm, 4*cm), fill='url(#pattern-4)', stroke_width=3, stroke='black'))


        img.add(pattern_3)
        img.add(img.rect(insert=(7*cm, 1*cm), size = (4*cm, 4*cm), fill='url(#pattern)', stroke_width=3, stroke='black'))


        img.add(pattern_4)
        img.add(img.rect(insert=(7*cm, 1*cm), size = (4*cm, 4*cm), fill='url(#pattern-2)', stroke_width=3, stroke='black'))


    # add markers
    # img.add(img.text('A', insert=(3 * 37.6, 5.8 * 37.6)))
    # img.add(img.text('B', insert=(9 * 37.6, 5.8 * 37.6)))

    # add labels 
    img.add(img.text('T'+str(index), insert=(0.1*cm, 0.5*cm), fill='lightgray'))

    img.save()


index = 0

for ratio in RATIOS:
    textures(('textures_' + str(index) + '.svg'), ratio, index)
    index += 1

# textures("instruction.svg", 0.5, 0)