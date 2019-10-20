import math
from skimage import io, color
import numpy as np
from tqdm import trange


class Cluster(object):
    # global cluster num
    cluster_index = 0

    def __init__(self, h, w, l=0, a=0, b=0):
        """
        Args:
            h: row position
            w: column position
            l: lab l value
            a: lab a value
            b: lab b value
        """
        self.update(h, w, l, a, b)
        self.pixels = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b

    def __str__(self):
        return "{},{}:{} {} {} ".format(self.h, self.w, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()


class SLICProcessor2D(object):
    def __init__(self, img, K, M):
        """
        Args:
            img: 2D numpy array
            K: num of seeds
            M: parameter for Dc
        """
        self.K = K
        self.M = M

        self.data = self.input_rgb2lab(img)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))

        self.clusters = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width), np.inf)
        
        self.output_image = self.data.copy()
        
    def input_rgb2lab(self, img):
        lab_arr = color.rgb2lab(img)
        return lab_arr
        
    def make_cluster(self, h, w):
        return Cluster(h, w, self.data[h][w][0],
                                    self.data[h][w][1],
                                    self.data[h][w][2])

    def init_clusters(self):
        # beginning of h and w
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height: # from top to down
            while w < self.image_width: # from left to right
                self.clusters.append(self.make_cluster(int(h), int(w)))
                w += self.S
            w = self.S / 2
            h += self.S

    def get_gradient(self, h, w):
        # if a 3x3 mask will exceed the image
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        gradient = self.data[w + 1][h + 1][0] - self.data[w][h][0] + \
                   self.data[w + 1][h + 1][1] - self.data[w][h][1] + \
                   self.data[w + 1][h + 1][2] - self.data[w][h][2]
        return gradient

    def relocate_clusters_by_gradient(self):
        for cluster in self.clusters:
            # the gradient of current location
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            # gradients within a 3x3 mask
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    if new_gradient < cluster_gradient:
                        cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
                        cluster_gradient = new_gradient

    def assign(self):
        """
        assign pixels to clusters
        """
        for cluster in self.clusters:
            # pixels in the range of a cluster
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
                if h < 0 or h >= self.image_height: continue
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                    if w < 0 or w >= self.image_width: continue
                        
                    L, A, B = self.data[h][w]
                    Dc = math.sqrt(
                        math.pow(L - cluster.l, 2) +
                        math.pow(A - cluster.a, 2) +
                        math.pow(B - cluster.b, 2))
                    Ds = math.sqrt(
                        math.pow(h - cluster.h, 2) +
                        math.pow(w - cluster.w, 2))
                    D = math.sqrt(math.pow(Dc / self.M, 2) + math.pow(Ds / self.S, 2))
                    
                    # if D is smaller than the older one, reassign the pixel
                    if D < self.dis[h][w]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D

    def relocate_clusters(self):
        for cluster in self.clusters:
            sum_h = sum_w = n = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                n += 1
                _h = int(sum_h / n)
                _w = int(sum_w / n)
                cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])

    def return_output_image(self):
        img = np.copy(self.output_image)
        for cluster in self.clusters:
            for p in cluster.pixels:
                img[p[0]][p[1]][0] = cluster.l
                img[p[0]][p[1]][1] = cluster.a
                img[p[0]][p[1]][2] = cluster.b
            img[cluster.h][cluster.w][0] = 0
            img[cluster.h][cluster.w][1] = 0
            img[cluster.h][cluster.w][2] = 0
        img = color.lab2rgb(img)
        return img

    def iterate(self, iter_times):
        self.init_clusters()
        self.relocate_clusters_by_gradient()
        for i in trange(iter_times):
            self.assign()
            self.relocate_clusters()
        self.output_image =  self.return_output_image()