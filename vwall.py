import itertools
import numpy as np
import pandas as pd

import help_functions as hf
import player
import rate_adaptation as ra

from scipy.spatial import ConvexHull


DATA_DIR = "/home/jvdhooft/Dropbox/PC/data/"
RES_DIR = "/home/jvdhooft/Dropbox/PC/images/"


cos = np.cos
sin = np.sin
sqrt = np.sqrt
asin = np.arcsin
atan2 = np.arctan2
acos = np.arccos
pi = np.pi


class Screen:
    def __init__(self, width, height, fov, scale):
        self.width = width
        self.height = height
        self.fov = fov
        self.scale = scale


class Box:
    def __init__(self, center, d_x, d_y, d_z):
        self.center = center
        self.d_x = d_x
        self.d_y = d_y
        self.d_z = d_z

        
class Frame:
    def __init__(self, camera, focus, boxes):
        self.camera = camera
        self.focus = focus
        self.boxes = boxes


# Screen characteristics
screen = Screen(800, 600, 20, 0.0005859375)

# Point cloud objects
pcs = [23, 24, 25, 26]
d = [pd.read_csv('%s/pc_%i.dat' % (DATA_DIR, pc), sep=' ', header=None) for pc in pcs]

# Number of frames
n_frames = min(len(pc.index) for pc in d)

# Rate adaptation information
f = pd.read_csv('%s/bits.dat' % DATA_DIR, sep=' ', header=None, names=['pc', 'frame', 'points', 'r1', 'r2', 'r3', 'r4', 'r5'])

# Iterate over camera paths and possible scenes
cameras = [1]
scenes = [1]
for camera, scene in itertools.product(cameras, scenes):
    
    # Read input data
    c = pd.read_csv('%s/camera_%i.dat' % (DATA_DIR, camera), sep=' ', header=None)
    q = pd.read_csv('%s/scene_%i.dat' % (DATA_DIR, scene), sep=' ', header=None)
    
    # Iterate over all frames
    h = pd.DataFrame(0.0, index=np.arange(len(pcs) * n_frames), columns=['frame', 'pc', 'A_vis', 'A_pot', 'dist'])
    frames = []
    for i in range(n_frames):
        
        # Define camera and focus location
        camera = np.array([c[1][i], c[2][i], c[3][i]])
        focus = np.array([c[4][i], c[5][i], c[6][i]])
        
        # Iterate over all point cloud objects
        boxes = []
        for j in range(len(pcs)):
            
            # Determine position and dimension
            x, y, z = d[j][0][i], d[j][1][i], d[j][2][i]
            d_x, d_y, d_z = d[j][3][i], d[j][4][i], d[j][5][i]
            
            # Determine offset and translation angles
            e = q.loc[q[0] == pcs[j]].iloc[0]
            o_x, o_y, o_z, a_x, a_y, a_z = [e[i] for i in range(1, 7)]
            
            # We only consider rotations around the y-axis for now
            x, z = cos(a_y) * x - sin(a_y) * z, \
                   sin(a_y) * x + cos(a_y) * z
            d_x, d_z = cos(a_y) * d_x - sin(a_y) * d_z, \
                       sin(a_y) * d_x + cos(a_y) * d_z
            
            # Translations
            x += o_x
            y += o_y
            z += o_z
            
            box = Box([x, y, z], d_x, d_y, d_z)
            boxes.append(box)
            
            e = h.iloc[i * len(pcs) + j]
            e['frame'] = i + 1
            e['pc'] = pcs[j]
            
            g = hf.in_scope(screen, camera, box.center, box)
            #print(g)
            e['dist'] = g[0]
            e['A_pot'] = g[1]
            
            g = hf.in_scope(screen, camera, focus, box)
            #print(g)
            e['A_vis'] = g[1]
        
        frame = Frame(camera, focus, boxes)
        frames.append(frame)
    
    data = pd.merge(f, h, on=['frame', 'pc'])


host = "10.0.0.1"
port = 8080
buffer_size = 2
gop = 30
fps = 30
n_seg = 10
rate_adapter = ra.RateAdaptation(5, 1, 0)
n_conn = 1
pc_names = {23: 'loot', 24: 'redandblack', 25: 'soldier', 26: 'longdress'}
pc_ids = [23, 24, 25, 26]

p = player.Player(host, port, buffer_size, gop, fps, n_seg, data,
                  rate_adapter, n_conn, pc_names, pc_ids)
p.run()
