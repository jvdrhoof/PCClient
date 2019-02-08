import itertools
import numpy as np
import pandas as pd

from scipy.spatial import ConvexHull


cos = np.cos
sin = np.sin
sqrt = np.sqrt
asin = np.arcsin
atan2 = np.arctan2
acos = np.arccos
pi = np.pi


def cart_to_spher(v):
    v = v / np.linalg.norm(v)
    phi = np.arctan2(v[1], v[0])
    if phi < 0:
        phi += 2 * pi
    theta = np.arccos(v[2])
    return phi, theta


def angle(v_1, v_2):
    v_1_u = v_1 / np.linalg.norm(v_1)
    v_2_u = v_2 / np.linalg.norm(v_2)
    return np.arccos(np.clip(np.dot(v_1_u, v_2_u), -1.0, 1.0))


def rotation_matrix(v):
    phi, theta = cart_to_spher(v)
    lat = pi / 2 - theta
    r_z = [[cos(phi), sin(phi), 0], [-sin(phi), cos(phi), 0], [0, 0, 1]]
    r_y = [[cos(lat), 0, sin(lat)], [0, 1, 0], [-sin(lat), 0, cos(lat)]]
    return np.matmul(r_y, r_z)


def xyz_to_xy(v, screen, crop):
    x, y, z = v
    d_y_screen = np.abs(x) * screen.scale * screen.height + screen.fov
    d_z_screen = np.abs(x) * screen.scale * screen.width + screen.fov * screen.width / screen.height
    x = -y / d_z_screen * screen.width
    y = z / d_y_screen * screen.height
    if crop:
        x = max(x, -screen.width / 2)
        x = min(x, screen.width / 2)
        y = max(y, -screen.height / 2)
        y = min(y, screen.height / 2)
    return [x, y]


def print_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


def in_scope(screen, camera, focus, box):
    r = rotation_matrix(focus - camera)
    v = box.center
    center_trans = v - camera
    center_rot = np.dot(r, center_trans)

    x, y, z = v
    c_x = [x - box.d_x / 2, x + box.d_x / 2]
    c_y = [y - box.d_y / 2, y + box.d_y / 2]
    c_z = [z - box.d_z / 2, z + box.d_z / 2]
    vertices_3d = [np.dot(r, v - camera) for v in itertools.product(c_x, c_y, c_z)]

    # Extract values
    x, y, z = zip(*vertices_3d)

    # The object is behind the camera
    if max(x) < 0:
        dist = max(x)
        area = 0

    # The camera is within the object
    elif min(x) < 0 and min(y) < 0 < max(y) and min(z) < 0 < max(z):
        dist = 0
        area = 1

    # The object is in front of the camera
    else:
        vertices_2d = [xyz_to_xy(v, screen, 1) for v in vertices_3d]
        u, v = zip(*vertices_2d)
        if min(u) == max(u) or min(v) == max(v):
            dist = min(x)
            area = 0
        else:
            hull = ConvexHull(vertices_2d)
            dist = min(x)
            area = hull.volume / screen.width / screen.height
        
    return dist, area
