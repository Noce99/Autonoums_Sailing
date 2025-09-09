import random

import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, sin, cos, pi, atan


def spherical_coordinate(r, latitude, longitude):
    """
    :param r: 0 < r < +infinity
    :param latitude: 0 < latitude < PI
    :param longitude: 0 < longitude < 2*PI
    """
    # From https://en.wikipedia.org/wiki/Spherical_coordinate_system
    x = r*sin(latitude)*cos(longitude)
    y = r*sin(latitude)*sin(longitude)
    z = r*cos(latitude)
    return x, y, z


def inverse_spherical_coordinate(x, y, z):
    # From https://en.wikipedia.org/wiki/Spherical_coordinate_system
    r = sqrt(x**2+y**2+z**2)
    latitude = None  # theta
    longitude = None  # phi
    if z > 0:
        latitude = atan(sqrt(x**2+y**2)/z)
    elif z < 0:
        latitude = pi + atan(sqrt(x**2+y**2)/z)
    if z == 0:
        if x*y == 0:
            print(f"UNDEFINED encountered in inverse_spherical_coordinate! [{x}, {y}, {z}]")
        else:
            latitude = pi / 2
    if x > 0:
        longitude = atan(y/x)
    elif x < 0 and y >= 0:
        longitude = atan(y/x) + pi
    elif x < 0 and y < 0:
        longitude = atan(y/x) - pi
    elif y > 0:
        longitude = pi / 2
    elif y < 0:
        longitude = - pi / 2
    else:
        print(f"UNDEFINED encountered in inverse_spherical_coordinate! [{x}, {y}, {z}]")
    if longitude < 0:
        longitude += 2*pi
    elif longitude > 2*pi:
        longitude -= 2*pi
    return r, latitude, longitude


def greed_point_coordinate(boat_position, greed_size, max_greed_distance_km):
    """
    :param boat_position: (Latitude, Longitude)
    :param greed_size: (Number of points in each of the 2 directions)
    :param max_greed_distance_km: Max km distance from the center (non diagonal)
    :return: A (2s+1, 2s+1, 2) numpy array with latitude and longitude of each grid point
    """
    t, g = boat_position
    if t == 0:
        t = 0.001
    if g == 0:
        g = 0.001
    s = greed_size
    h = max_greed_distance_km
    b_p = (cos(g)*sin(t), sin(g)*sin(t), cos(t))
    bpx, bpy, bpz = b_p
    d_theta = h / (s*6314)
    output_latitude_longitude = np.zeros((s*2+1, s*2+1, 2))

    L = 1 / sqrt(1 * 1 + bpx ** 2 / bpy ** 2)
    K = -bpx / bpy / sqrt(1 * 1 + bpx ** 2 / bpy ** 2)
    for i in range(-s, s+1, 1):
        c = cos(pi / 2 - t - d_theta * i)
        my_s = sin(pi / 2 - t - d_theta * i)
        C = 1 - c
        R11 = L ** 2 * C + c
        R12 = L * K * C
        R21 = L * K * C
        R22 = K * K * C + c
        R31 = -K * my_s
        R32 = L * my_s
        for j in range(-s, s+1, 1):
            xc1, yc1 = cos(g+d_theta*j), sin(g+d_theta*j)
            _, latitude, longitude = inverse_spherical_coordinate(R11*xc1 + R12*yc1, R21*xc1 + R22*yc1,
                                                                  R31*xc1 + R32*yc1)
            if g > pi:
                latitude = pi - latitude
            if i == 0 and j == 0:
                # print(f"{latitude}, {longitude}")
                pass
            output_latitude_longitude[i + s, j + s, 0] = latitude
            output_latitude_longitude[i + s, j + s, 1] = longitude
    return output_latitude_longitude


def bilinear_interpolation(x, y, x1, x2, y1, y2, f11, f12, f22, f21):
    """
    All definition in: https://en.wikipedia.org/wiki/Bilinear_interpolation
    :param x:
    :param y:
    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :param f11:
    :param f12:
    :param f22:
    :param f21:
    :return:
    """
    f1 = (x2-x)/(x2-x1)*f11+(x-x1)/(x2-x1)*f21
    f2 = (x2-x)/(x2-x1)*f12+(x-x1)/(x2-x1)*f22
    return (y2-y)/(y2-y1)*f1+(y-y1)/(y2-y1)*f2


"""
# Usage Example:
result = greed_point_coordinate((0.785398, 0.1745329), 10, 1000, visualize=True)
"""