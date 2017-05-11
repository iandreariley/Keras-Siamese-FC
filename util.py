from math import sqrt

def inclusive_range(start, end):
    return range(start, end + 1)

def euclidean_distance(x1, y1, x2, y2):
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)
