# Standard library imports
import sys
import os

# Third party imports
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import scipy
from skimage import morphology

from matplotlib.widgets import LassoSelector, RectangleSelector
from matplotlib.path import Path


class SelectFromCollection:
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.5):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts, closed=False)
        self.ind = np.nonzero(path.contains_points(self.xys, radius=1.0))[0]
        #self.fc[:, -1] = self.alpha_other
        #self.fc[self.ind, -1] = 1
        #self.collection.set_facecolors(self.fc)
        #self.canvas.draw_idle()

    def disconnect(self, fc):
        #self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(fc)
        self.collection.set_alpha(0.5)
        #self.canvas.draw_idle()



class ArcSelector(object):
    def __init__(
            self, image=None, gid_vertex_dict=None, gid_edge_dict=None, selection_radius=4, valley=True
            , ridge=True, invert=False, kdtree=None, arc_map=None
    ):
        light_red = "#fbb4ae"
        red = "#e41a1c"

        light_blue = "#b3cde3"
        blue = "#377eb8"

        light_green = "#ccebc5"
        green = "#4daf4a"

        light_purple = "#decbe4"
        purple = "#984ea3"

        light_orange = "#fed9a6"
        orange = "#ff7f00"

        yellow = "#ffff33"

        offIvory = "#ffffa2"
        # Needed for kdTree on large point sets:
        sys.setrecursionlimit(10000)

        self.image = image
        self.gid_vertex_dict = gid_vertex_dict
        self.gid_edge_dict = gid_edge_dict

        self.invert = invert

        self.in_color = red
        self.out_color = blue
        self.bg_color = purple
        self.deselect_color = offIvory

        self.in_vertices = set()
        self.out_vertices = set()
        self.test_vertices = set()
        self.train_vertices = set()

        self.out_pixels = set()
        self.arc_drawings = {}
        self.kdtree = kdtree
        self.arc_map = arc_map

    def get_closest_arc_index(self, point):
        distance, index = self.kdtree.query(point)
        return self.arc_map[index]

    def launch_ui(self,xlims=None, ylims=None, use_inference=False, box_select=False, msc_arcs=None):


        if xlims is None or ylims is None and self.image is not None:
            X = self.image.shape[0]
            Y = self.image.shape[1]
            xlims = [0, X]
            ylims = [0, Y]

        plt.ion()

        if xlims is None or ylims is None:
            subplot_kw = dict(xlim=(0, self.image.shape[1]), ylim=(self.image.shape[0], 0), autoscale_on=False)
        if xlims is not None and ylims is not None:
            subplot_kw = dict(xlim=(xlims[0], xlims[1]), ylim=(ylims[1], ylims[0]), autoscale_on=False)
        self.fig, self.ax = plt.subplots(subplot_kw=subplot_kw)  # figure()

        arc_xpoints, arc_ypoints = [], []
        arc_points = []
        self.scatter_points = self.ax.scatter(arc_xpoints,
                                              arc_ypoints,
                                              facecolor="ivory",
                                              edgecolor="none",
                                              s=1,
                                              marker=",",
                                              alpha=0.3,
                                              zorder=1,
                                              )

        if use_inference:
            cmap = cm.get_cmap('seismic')
            cmap.set_under('black')
            cmap.set_bad('black')
            # cmap.set_over('white')
            # plt.set_cmap(cmap)

            cmap_accurate = cm.get_cmap('cool')

        plt.imshow(self.image, cmap=plt.cm.Greys_r, zorder=2)  # cmap=plt.cm.Greys_r, #cmap=plt.cm.Greys_r,
        c = 0
        for vertex in self.gid_vertex_dict:

            arc_index = vertex.gid
            points = np.array(vertex.points)

            arc_xpoints.append(points[:, 0])
            arc_ypoints.append(points[:, 1])
            for point in points:
                arc_points.append(np.asarray(point))

            old_offset = self.scatter_points.get_offsets()
            new_offset = np.concatenate([old_offset, np.array(points)])
            old_color = self.scatter_points.get_facecolors()

            if c == 0:
                new_color = np.concatenate([old_color, np.array(old_color)])
                c += 1
            else:
                new_color = np.concatenate([old_color, np.array([old_color[0, :]])])

            self.scatter_points.set_offsets(new_offset)  # np.c_[points[:,0],points[:,1]])
            self.scatter_points.set_facecolors(new_color)
            # self.fig.canvas.draw()

            color = "ivory"

            if use_inference:
                if not isinstance(vertex.prediction, (int, np.integer)):
                    if len(vertex.prediction) == 3:
                        label_color = cmap(0.5) if float(vertex.prediction[2]) > 0.5 else cmap(float(vertex.prediction[1]))
                        pred = float(vertex.prediction[2]) if float(vertex.prediction[2]) > 0.5 else float(vertex.prediction[1])
                    else:
                        label_color = cmap(float(vertex.prediction[1]))
                        pred = float(vertex.prediction[1])
                else:
                    # print("pred ", arc.prediction)
                    label_color = cmap(float(vertex.prediction))
                    pred = float(vertex.prediction)
                color = label_color
                if pred <= 0.06 or pred >= 0.94:
                    color = cmap_accurate(pred)

            self.arc_drawings[arc_index] = self.ax.scatter(  ##### plt.scatter
                points[:, 0],
                points[:, 1],
                facecolor=color,
                edgecolor="none",
                s=2,
                marker=",",
                alpha=0.3,
                zorder=3,
            )
        # np.c_[points[:, 0], points[:, 1]])
        self.scatter_points.set_visible(False)
        self.fig.canvas.draw()

        print(">>>> image shape: ", self.image.shape)
        extrema_points = [[]]  # append all 2-saddle to inner array for slicing
        for edge in self.gid_edge_dict.values():
            x, y = edge.points
            extrema_points[0].append([x, y])
        xy = np.array(extrema_points)  # [i])
        self.ax.scatter(  # plt.scatter
            xy[:, 0],
            xy[:, 1],
            facecolor=self.bg_color,
            edgecolor="none",
            s=4,
            marker=",",
            zorder=4,
        )
        self.selector = SelectFromCollection(self.ax, self.scatter_points)
        self.fig.canvas.mpl_connect("key_press_event", self.assign_class)
        # self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        plt.show(block=True)

        in_vertices = []
        out_vertices = []
        test_vertices = []
        train_vertices = []
        test_vertex_ids = []
        for vertex in self.gid_vertex_dict.values():
            index = vertex.gid
            if index in self.in_vertices:
                in_vertices.append(vertex)
            elif index in self.out_vertices:
                out_vertices.append(vertex)
            # elif index in self.train_arcs:
            #    train_arcs.append(arc)
            else:
                test_vertices.append(vertex)
                test_vertex_ids.append(index)

        # end of ui
        return self.in_vertices, in_vertices, self.out_vertices, out_vertices, np.array(list(self.out_pixels)), test_vertices, test_vertex_ids

    def toggle_arc_lasso(self, x_points, y_points, event_key='1'):
        in_class = event_key == '1'
        out_class = event_key == '2'
        remove_arcs = event_key == 'x'

        min_indices = []
        for x, y in zip(x_points,y_points): # no loop just x y event
            pt = np.array([x, y])
            min_index = self.get_closest_arc_index(pt)
            min_indices.append(min_index)
            if in_class:
                if min_index in self.in_vertices:
                    self.in_vertices.remove(min_index)
                elif min_index in self.out_vertices:
                    self.out_vertices.remove(min_index)
                elif min_index in self.test_vertices:
                    self.test_vertices.remove(min_index)
                #else:
                #    #print("added to positive labels")
                self.in_vertices.add(min_index)
            if out_class:
                if min_index in self.out_vertices:
                    self.out_vertices.remove(min_index)
                elif min_index in self.in_vertices:
                    self.in_vertices.remove(min_index)
                elif min_index in self.test_vertices:
                    self.test_vertices.remove(min_index)
                #    #print("added to negative labels")
                self.out_vertices.add(min_index)
            if remove_arcs:
                if min_index in self.out_vertices:
                    #print("removed from negative labels")
                    self.out_vertices.remove(min_index)
                elif min_index in self.in_vertices:
                    #print("removed from positive labels")
                    self.in_vertices.remove(min_index)
                elif min_index in self.test_vertices:
                    self.test_vertices.remove(min_index)
                self.test_vertices.add(min_index)
        return min_indices

    def assign_class(self, event):
        xdata = self.selector.xys[self.selector.ind][:,0]
        ydata = self.selector.xys[self.selector.ind][:, 1]
        selected_indices = self.toggle_arc_lasso(                         #           selected_index
            self.selector.xys[self.selector.ind][:,0],self.selector.xys[self.selector.ind][:,1]
            , event.key  #event.xdata
        )
        for selected_index in selected_indices:
            if event.key == '1':
                self.arc_drawings[selected_index].set_facecolor(self.in_color)
                self.arc_drawings[selected_index].set_alpha(0.3)
            elif event.key == '2':
                self.arc_drawings[selected_index].set_facecolor(self.out_color)
                self.arc_drawings[selected_index].set_alpha(0.3)
            elif event.key == 'x':
                self.arc_drawings[selected_index].set_facecolor(self.deselect_color)
                self.arc_drawings[selected_index].set_alpha(0.3)

        self.scatter_points.set_visible( False )
        self.fig.canvas.draw()