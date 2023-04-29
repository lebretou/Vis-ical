import random
RATIOS = [0.55, 0.45, 0.25, 0.65, 0.75, 0.35]

def generate_pairs(ratios):
    pairs = []
    for i in range(len(ratios)):
        larger_num = random.uniform(0, 100)
        smaller_num = larger_num * ratios[i]
        larger_num = round(larger_num, 1)
        smaller_num = round(smaller_num, 1)

        coin_flip = random.randint(0, 1)
        if coin_flip == 0:
            tmp = larger_num
            larger_num = smaller_num
            smaller_num = tmp
        pairs.append([larger_num, smaller_num])

    return pairs

print(generate_pairs(RATIOS))
