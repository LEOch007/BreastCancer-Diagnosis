# -*- coding: utf-8 -*-
# @Time : 2020/5/30 4:22 下午 
# @Author : Qingduo-Feng 
# @File : FeatureCalculation.py 
# @Function:

import numpy as np
import pandas as pd
import math

def eduDis(a, b):
    dis = np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    return dis

def straightline(point2, point1):
    k = (point2[1] - point1[1]) / (point2[0] - point1[0])
    b = point2[1] - k * point2[0]
    return k, b


def pointToLineDis(point, line):
    if line[0] == 0:
        offset = round(abs(point[0]), 2)
    elif line[0] == 1:
        offset = round(abs(point[1]), 2)
    else:
        k, b = line[1], line[2]
        offset = round(abs(k * point[0] - point[1] + b) / np.sqrt(k ** 2 + 1), 2)
    return offset

def isEdge(img, point):
    x_max = len(img)
    y_max = len(img[0])
    x = point[0]
    y = point[1]
    count = 0

    if x + 1 >= x_max - 1 or y + 1 >= y_max - 1:
        return True

    if img[x + 1, y] > 0:
        count += 1
    if img[x, y+1] > 0:
        count += 1
    if img[x-1, y] > 0:
        count += 1
    if img[x, y-1] > 0:
        count += 1
    if count < 4:
        return True
    else:
        return False


def calRadius(edge_points):
    # calculate radius

    radiuses = []
    points_num = np.unique(edge_points)
    points = np.array(edge_points)
    for index in points_num:
        if index == 0:
            continue
        points_cood = np.argwhere(points == index)
        center_x = np.mean(points_cood[:, 0])
        center_y = np.mean(points_cood[:, 1])
        center = (int(center_x), int(center_y))
        tmp_radius = []
        for point in points_cood:
            # judge if the point is the edge
            if not isEdge(edge_points, point):
                continue
            radius = eduDis(center, point)
            tmp_radius.append(radius)
        radiuses.append(np.mean(tmp_radius))
    return radiuses

def calPerimeter(edge_points):
    perimeters = []
    points_num = np.unique(edge_points)
    points = np.array(edge_points)
    for index in range(1, len(points_num)):
        points_cood = np.argwhere(points == index)
        tmp_perimeter = 0
        for point in points_cood:
            # judge if the point is the edge
            if not isEdge(edge_points, point):
                continue
            tmp_perimeter += 1
        perimeters.append(tmp_perimeter)
    return perimeters

def calArea(edge_points):
    area = []
    points_num = np.unique(edge_points)
    points = np.array(edge_points)
    for index in points_num:
        if index == 0:
            continue
        points_cood = np.argwhere(points == index)
        area.append(len(points_cood))
    return area

def calCompactness(perimeter, area):
    p = np.array(perimeter)
    a = np.array(area)
    return p ** 2 / a

def calSmoothness(edge_points):
    smoothness = []
    points_num = np.unique(edge_points)
    points = np.array(edge_points)
    for index in points_num:
        if index == 0:
            continue
        points_cood = np.argwhere(points == index)
        center_x = np.mean(points_cood[:, 0])
        center_y = np.mean(points_cood[:, 1])
        center = (int(center_x), int(center_y))
        tmp_radius = []
        for point in points_cood:
            # judge if the point is the edge
            if not isEdge(edge_points, point):
                continue
            radius = eduDis(center, point)
            tmp_radius.append(radius)
        # calculate smoonthness
        radius_count = len(tmp_radius)
        differences = []
        for i in range(0, radius_count):
            if radius_count < 3:
                differences.append(0)
                break
            if i == 1:
                differences.append(abs(tmp_radius[i] - np.mean([tmp_radius[i + 1], tmp_radius[0]])))
            elif i == radius_count - 1:
                differences.append(abs(tmp_radius[i] - np.mean([tmp_radius[0], tmp_radius[i - 1]])))
            else:
                differences.append(abs(tmp_radius[i] - np.mean([tmp_radius[i+1], tmp_radius[i - 1]])))
        smoothness.append(np.mean(differences))
    return smoothness


def kb(vertex1, vertex2):
    x1 = vertex1[0]
    y1 = vertex1[1]
    x2 = vertex2[0]
    y2 = vertex2[1]

    if x1 == x2:
        return (0, x1)  # 0-垂直直线
    if y1 == y2:
        return (1, y1)  # 1-水平直线
    else:
        k = (y1 - y2) / (x1 - x2)
        b = y1 - k * x1
        return (2, k, b)  # 2-倾斜直线

def calConcavity(edge_points):
    # 首先通过每两点之间构建函数，并判断所有点距离该线的符号是否一致，来判断该点是凸还是凹
    # 对于所有凹点，计算距离周边两点直线的距离即为concavity
    concavity = []
    concavity_count = []
    points_num = np.unique(edge_points)
    points = np.array(edge_points)
    for index in points_num:
        edges = []
        if index == 0:
            continue
        points_cood = np.argwhere(points == index)
        for point in points_cood:
            # judge if the point is the edge
            if isEdge(edge_points, point):
                edges.append(point)

        concav_points = []
        tmp_count = 0
        # begin to calculate the concavity
        for i in range(len(edges)):
            pre = i
            nex = (i+1)%len(edges)
            line = kb(edges[pre], edges[nex])

            if line[0] == 0:
                offset = [vertex[0] - edges[pre][0] for vertex in edges]
            elif line[0] == 1:
                offset = [vertex[1] - edges[pre][1] for vertex in edges]
            else:
                k, b = line[1], line[2]
                offset = [k * vertex[0] + b - vertex[1] for vertex in edges]
            offset = np.array(offset)
            large_count = len(np.argwhere(offset >= 0))
            small_count = len(np.argwhere(offset <= 0))
            if large_count != len(edges) or small_count != len(edges):
                # the point is a concav point
                concav_points.append(i)
                tmp_count += 1

        # begin to calculate the value of concavity
        tmp_concav = 0
        for i in concav_points:
            pre = (i - 1) % len(edges)
            nex = (i + 1) % len(edges)
            point = edges[i]
            line = kb(edges[pre], edges[nex])

            if line[0] == 0:
                offset = point[0] - edges[pre][0]
            elif line[0] == 1:
                offset = point[1] - edges[pre][1]
            else:
                k, b = line[1], line[2]
                offset = k * point[0] + b - point[1]
            offset = abs(offset)
            tmp_concav += offset

        concavity_count.append(tmp_count)
        concavity.append(tmp_concav)

    return concavity_count, concavity

def calSymmetry(edge_points):
    symmetry = []
    points_num = np.unique(edge_points)  # get the number of nucleus
    points = np.array(edge_points)
    for index in points_num:
        if index == 0:
            continue
        points_cood = np.argwhere(points == index)
        longest_distance = 0
        distpoint1 = []
        distpoint2 = []
        # calculate the longest distance and the relevant points
        # print(points_cood)
        for i in points_cood:
            for j in points_cood:
                if (longest_distance <= eduDis(i, j)):
                    longest_distance = eduDis(i, j)
                    distpoint1 = i
                    distpoint2 = j
        # get the straight line equation of distpoint1 and dispoint2
        line = kb(distpoint2, distpoint1)

        leftside = 0
        rightside = 0
        for i in points_cood:
            if line[0] == 0:
                result = i[0]
            elif line[0] == 1:
                result = i[1]
            else:
                k, b = line[1], line[2]
                result = k * i[0] + b
            if (result < i[1]):
                leftside = leftside + pointToLineDis(i, line)
            else:
                rightside = rightside + pointToLineDis(i, line)
        symmetry.append(abs(leftside - rightside))
    return symmetry

def calFractalDim(edge_points):
    fractalDim = []
    points_num = np.unique(edge_points)  # get the number of nucleus
    points = np.array(edge_points)
    for index in points_num:
        if index == 0:
            continue
        points_cood = np.argwhere(points == index)
        slopes = []
        i = 0
        while (i < len(points_cood) - 1):
            if not isEdge(edge_points, points_cood[i]):
                i = i + 1
                continue
            j = i + 1
            while (j < len(points_cood) - 1):
                if not isEdge(edge_points, points_cood[j]):
                    j = j + 1
                    continue
                if (1 < eduDis(points_cood[i], points_cood[j]) < 10):
                    slope = (points_cood[i][1] - points_cood[j][1]) / (points_cood[i][0] - points_cood[j][0])
                    if (slope < 0):
                        if (math.isinf(slope) != True):
                            if (math.isnan(slope) != True):
                                slopes.append(slope)
                j = j + 1
            i = j + 1
        # print(slopes)
        fractalDim.append(np.mean(slopes))
    return fractalDim


def calTexture(edge_points, image):
    texture = []
    points_num = np.unique(edge_points)
    points = np.array(edge_points)
    for index in points_num:
        if index == 0:
            continue
        points_cood = np.argwhere(points == index)
        intensity_value = []
        for point in points_cood:
            x = point[0]
            y = point[1]
            intensity_value.append(image[x,y])
        intensity_value = np.array(intensity_value)
        texture.append(np.var(intensity_value))
    return texture


def feature_extract(center_points, edge_points, image):
    feature_arr = []
    radius = np.array(calRadius(edge_points))
    perimeter = np.array(calPerimeter(edge_points))
    area = np.array(calArea(edge_points))
    compactness = np.array(calCompactness(perimeter, area))
    smoothness = np.array(calSmoothness(edge_points))
    concavity_points, concavity = calConcavity(edge_points)
    concavity_points = np.array(concavity_points)
    concavity = np.array(concavity)
    symmetry = np.array(calSymmetry(edge_points))
    textture = np.array(calTexture(edge_points, image))
    fractal_dimension = calFractalDim(edge_points)
    fractal_dimension = np.array(fractal_dimension)
    fractal_dimension[np.isnan(fractal_dimension)] = 0

    # calculate mean value
    feature_arr.append(np.mean(radius))
    feature_arr.append(np.mean(perimeter))
    feature_arr.append(np.mean(area))
    feature_arr.append(np.mean(compactness))
    feature_arr.append(np.mean(smoothness))
    feature_arr.append(np.mean(concavity))
    feature_arr.append(np.mean(concavity_points))
    feature_arr.append(np.mean(symmetry))
    feature_arr.append(np.mean(fractal_dimension))
    feature_arr.append(np.mean(textture))

    feature_arr.append(np.std(radius))
    feature_arr.append(np.std(perimeter))
    feature_arr.append(np.std(area))
    feature_arr.append(np.std(compactness))
    feature_arr.append(np.std(smoothness))
    feature_arr.append(np.std(concavity))
    feature_arr.append(np.std(concavity_points))
    feature_arr.append(np.std(symmetry))
    feature_arr.append(np.std(fractal_dimension))
    feature_arr.append(np.std(textture))

    feature_arr.append(np.max(radius))
    feature_arr.append(np.max(perimeter))
    feature_arr.append(np.max(area))
    feature_arr.append(np.max(compactness))
    feature_arr.append(np.max(smoothness))
    feature_arr.append(np.max(concavity))
    feature_arr.append(np.max(concavity_points))
    feature_arr.append(np.max(symmetry))
    feature_arr.append(np.min(fractal_dimension))
    feature_arr.append(np.max(textture))

    feature_arr = np.array(feature_arr)
    feature_arr = np.around(feature_arr, decimals=4)


    return feature_arr.tolist()
