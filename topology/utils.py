
# Third party imports
import numpy as np



cached_msc_path = None

def get_points_from_vertices(vertices, sampled = False, image = None):
    points = tuple()
    for vertex in vertices:
        line = vertex.points
        if sampled:
            t_points = tuple()
            if len(line) >= 3:
                p1 = line[0]
                p2 = line[len(line)//2]
                p3 = line[-1]
                #pix1 = image[p1[:, 1], p1[:, 0]]
                #pix2 = image[p2[:, 1], p2[:, 0]]
                #pix3 = image[p3[:, 1], p3[:, 0]]
                t_points = np.array([p1,p2,p3])#.flatten()
            elif len(line) == 2:
                p1 = line[0]
                p2 = line[1]
                p_middle = ((p1[1] + p2[1])/2.,(p1[0] + p2[0])/2.)
                #pix1 = image[p1[:, 1], p1[:, 0]]
                #pix2 = image[p2[:, 1], p2[:, 0]]
                t_points = np.array([p1,p_middle,p2])#.flatten()
            elif len(line) == 1:
                p1 = line[0]
                #pix1 = image[p1[:, 1], p1[:, 0]]
                t_points = np.array([p1,p1,p1])#.flatten()
            return t_points
        points += tuple(np.array(np.round(line), dtype=int))
    points = np.vstack(points)
    return points


def get_pixel_values_from_vertices(vertices, image, sampled = False):
    if sampled:
        return get_points_from_vertices(vertices, sampled=True, image=image)
    else:
        points = get_points_from_vertices(vertices,sampled = sampled, image = image)
    return image[points[:, 1], points[:, 0]].flatten()

def get_centroid(vertex):
    points = np.array(vertex.points)
    x_mean = np.mean(points[:,1])#/points.shape[0]
    y_mean = np.mean(points[:,0])#/points.shape[0]
    centroid = np.array([x_mean, y_mean])
    return centroid

def translate_points_by_centroid(vertex, centroid):
    points = get_points_from_vertices(vertex, sampled=True)
    points = points - centroid
    return points

def is_ridge_arc(arc, msc):
    return 0 not in [
        msc.nodes[arc.node_ids[0]].index,
        msc.nodes[arc.node_ids[1]].index,
    ]


def is_valley_arc(arc, msc):
    return 2 not in [
        msc.nodes[arc.node_ids[0]].index,
        msc.nodes[arc.node_ids[1]].index,
    ]

