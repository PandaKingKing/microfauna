import numpy as np
import cv2


def calc_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def calc_angle(pt1, pt2, pt3):
    a, b, c = calc_distance(pt1, pt2), calc_distance(pt2, pt3), calc_distance(pt1, pt3)
    cosine = abs(a ** 2 + b ** 2 - c ** 2) / (2 * a * b) if a and b else 1
    return np.arccos(round(cosine, 5))


class Track:
    def __init__(self, pts, start_frame=0):
        self.pts = list(pts)
        self.start_frame = int(start_frame)

    def __len__(self):
        return len(self.pts)
    
    def __getitem__(self, item):
        assert -len(self) <= item < len(self)
        return self.pts[item]

    def draw(self, src):
        for i in range(1, len(self)):
            cv2.line(src, self[i - 1], self[i], (255, 255, 0), thickness=5)
        cv2.circle(src, self[-1], 10, (0, 0, 255), -1)

    def associate(self, pts, dist_range=(0, 100)):
        pts_dist = [calc_distance(self[-1], pt) for pt in pts]
        idx_min_dist = np.argmin(pts_dist)
        retval = dist_range[0] < pts_dist[idx_min_dist] < dist_range[1]
        if retval:
            self.pts.append(pts[idx_min_dist])
            del pts[idx_min_dist]
        return retval

    def update(self, translation):
        self.pts = [(round(pt[0] + translation[0]), round(pt[1] + translation[1]))
                    for pt in self.pts]

    def motion_analysis(self, fps):
        assert len(self) >= 3
        v_linear = [calc_distance(self[i - 1], self[i]) * fps
                    for i in range(1, len(self))]
        v_angular = [calc_angle(self[i - 2], self[i - 1], self[i]) * fps
                     for i in range(2, len(self))]
        return v_linear, v_angular

