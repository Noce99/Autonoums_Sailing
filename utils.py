import math
import random

import torch

from Sailing_Boats_Autopilot.constants import EARTH_RADIUS


def m_to_lon(distance, position):
    actual_lat = position[0]
    earth_circumference_at_actual_lat = 2 * math.pi * EARTH_RADIUS * math.cos(math.pi/2 - actual_lat)
    """
    print("@" * 120)
    print(f"Actual Lat: {math.cos(math.pi/2 - actual_lat)}")
    print(f"earth_circumference_at_actual_lat = {earth_circumference_at_actual_lat}")
    print(f"Earth Circumference = {2 * math.pi * EARTH_RADIUS}")
    print("@"*120)
    """
    return distance / earth_circumference_at_actual_lat * 2 * math.pi


def m_to_lat(distance):
    return distance / EARTH_RADIUS


def lat_converter(degree, minutes, seconds, direction):
    degree = degree / 180 * math.pi
    minutes = (minutes / 60) / 180 * math.pi
    seconds = (seconds / 3600) / 180 * math.pi
    if direction == "N":
        return math.pi / 2 - degree - minutes - seconds
    elif direction == "S":
        return math.pi / 2 + degree + minutes + seconds
    else:
        print("Error: Latitude direction not 'N' or 'S'")


def lon_converter(degree, minutes, seconds, direction):
    degree = degree / 180 * math.pi
    minutes = (minutes / 60) / 180 * math.pi
    seconds = (seconds / 3600) / 180 * math.pi
    if direction == "E":
        return degree + minutes + seconds
    elif direction == "W":
        return 2*math.pi - degree - minutes - seconds
    else:
        print("Error: Latitude direction not 'E' or 'W'")


# Check-Point Coordinate: https://voilesetvoiliers.ouest-france.fr/course-au-large/vendee-globe/vendee-globe-decouvrez-le-parcours-et-ses-portes-des-glaces-7555327d-294a-b643-931e-3c427662d7c2
LORIENT = [lat_converter(47, 41, 23, "N"), lon_converter(3, 23, 23, "W")]
CHECK_POINT_0 = [lat_converter(43, 57, 40, "N"), lon_converter(11, 28, 11, "W")]  # Norther Spain
CHECK_POINT_1 = [lat_converter(5, 50, 4, "S"), lon_converter(31, 46, 20, "W")]    # Middle Atlantique
CHECK_POINT_2 = [lat_converter(40, 18, 50, "S"), lon_converter(9, 56, 43, "W")]   # Gough Island
CHECK_POINT_3 = [lat_converter(42, 0, 0, "S"), lon_converter(20, 0, 0, "E")]      # Porte Atlantique
CHECK_POINT_4 = [lat_converter(50, 0, 0, "S"), lon_converter(45, 0, 0, "E")]      # Porte Kerguelen
CHECK_POINT_5 = [lat_converter(53, 6, 6, "S"), lon_converter(73, 32, 0, "E")]     # Heard
CHECK_POINT_6 = [lat_converter(46, 0, 0, "S"), lon_converter(105, 0, 0, "E")]     # AMSA 1
CHECK_POINT_7 = [lat_converter(50, 0, 0, "S"), lon_converter(140, 0, 0, "E")]     # AMSA 2
CHECK_POINT_8 = [lat_converter(52, 0, 0, "S"), lon_converter(175, 0, 0, "W")]     # Porte Nouvelle Zelande
CHECK_POINT_9 = [lat_converter(52, 0, 0, "S"), lon_converter(140, 0, 0, "W")]     # Porte Ouest Pacifique
CHECK_POINT_10 = [lat_converter(56, 8, 43, "S"), lon_converter(63, 53, 47, "W")]  # Cape Hornos
CHECK_POINT_11 = [lat_converter(5, 50, 4, "S"), lon_converter(31, 46, 20, "W")]   # Middle Atlantique
CHECK_POINT_12 = [lat_converter(43, 57, 40, "N"), lon_converter(11, 28, 11, "W")]  # Norther Spain
tracks = [
          (LORIENT, CHECK_POINT_0),
          (CHECK_POINT_0, CHECK_POINT_1),
          (CHECK_POINT_1, CHECK_POINT_2),
          (CHECK_POINT_2, CHECK_POINT_3),
          (CHECK_POINT_3, CHECK_POINT_4),
          (CHECK_POINT_4, CHECK_POINT_5),
          (CHECK_POINT_5, CHECK_POINT_6),
          (CHECK_POINT_6, CHECK_POINT_7),
          (CHECK_POINT_7, CHECK_POINT_8),
          (CHECK_POINT_8, CHECK_POINT_9),
          (CHECK_POINT_9, CHECK_POINT_10),
          (CHECK_POINT_10, CHECK_POINT_11),
          (CHECK_POINT_11, CHECK_POINT_12),
          (CHECK_POINT_12, LORIENT),
          ]

LAST = -2


def random_start_end_race():
    global LAST
    LAST += 1
    if LAST >= len(tracks):
        LAST = 0
    return tracks[LAST]


def random_location_in_atlantic_ocean():
    """
    lat_degree = random.randint(15, 43)
    lon_degree = random.randint(20, 70)
    lat = lat_converter(lat_degree, random.randrange(0, 60), random.randrange(0, 60), "N")
    lon = lon_converter(lon_degree, random.randrange(0, 60), random.randrange(0, 60), "W")
    """
    lat = lat_converter(34, random.randrange(0, 60), random.randrange(0, 60), "N")
    lon = lon_converter(28, random.randrange(0, 60), random.randrange(0, 60), "W")
    return [lat, lon]


def torch_from_direction_to_ones(directions):
    """
    :param directions: torch.Tensor of direction between 0 and 2*Pi
    :return: concatenated torch tensor of ones
    """
    out = torch.zeros(directions.shape[0], directions.shape[1]*8).cuda()
    for batch in range(directions.shape[0]):
        for i in range(directions.shape[1]):
            out[batch, 8*i + int(directions[batch, i] / math.pi * 4)] = 1
    return out


def single_batch_torch_from_direction_to_ones(directions):
    out = torch.zeros(directions.shape[0] * 8).cuda()
    for i in range(directions.shape[0]):

        out[8 * i + int((directions[i]-0.001) / math.pi * 4)] = 1
    return out


def get_single_directions_embedded(boat_orientation, boat_position, target_position, wind=None):
    if wind is not None:
        wind_orientation = math.pi - math.atan2(wind[1], wind[0])
        wind_orientation -= math.pi/2
        if wind_orientation < 0:
            wind_orientation = 2*math.pi + wind_orientation
        out = single_batch_torch_from_direction_to_ones(
            torch.Tensor([boat_orientation, from_start_end_to_direction(boat_position, target_position),
                          wind_orientation]))
    else:
        out = single_batch_torch_from_direction_to_ones(
            torch.Tensor([boat_orientation, from_start_end_to_direction(boat_position, target_position)]))
    return out


def calculate_directions(boat_orientation, boat_position, target_position, wind):
    wind_orientation = math.pi - math.atan2(wind[1], wind[0])
    wind_orientation -= math.pi / 2
    if wind_orientation < 0:
        wind_orientation = 2 * math.pi + wind_orientation
    return torch.Tensor([boat_orientation, from_start_end_to_direction(boat_position, target_position),
                         wind_orientation])


def from_start_end_to_direction(start_point, end_point):
    """
    # NON FUNZIONA QUANDO SI CAMBIA QUADRANTEEEE!
    TIPO:
    START POINT = [lat_converter(45, 0, 0, "N"), lon_converter(45, 0, 0, "W")]
    END POINT = [lat_converter(45, 0, 0, "N"), lon_converter(315, 0, 0, "W")]
    :param start_point:
    :param end_point:
    :return: A 0 -> 2*PI angle
    """
    if start_point[1] > math.pi > end_point[1]:
        start_point[1] = - (2*math.pi - start_point[1])
    if start_point[1] < math.pi < end_point[1] and abs(end_point[1] - start_point[1]) > math.pi:
        end_point[1] = - (2 * math.pi - end_point[1])
    angle = math.atan2(end_point[1]-start_point[1], end_point[0]-start_point[0])
    angle = math.pi - angle
    return angle


def angular_distance_degree(start_point, end_point):
    distance_angle = math.acos(math.sin(start_point[0] - math.pi / 2) * math.sin(end_point[0] - math.pi / 2) +
                               math.cos(start_point[0] - math.pi / 2) * math.cos(end_point[0] - math.pi / 2) *
                               math.cos(start_point[1] - end_point[1]))
    return distance_angle / math.pi * 180
