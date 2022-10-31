import numpy as np
import cv2


def calc_translation(src1, src2, method='orb'):
    translation = ()
    try:
        MIN_MATCH_COUNT = 10
        if len(src1.shape) > 2:
            src1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
        if len(src2.shape) > 2:
            src2 = cv2.cvtColor(src2, cv2.COLOR_BGR2GRAY)
        if method == 'sift':
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(src1, None)
            kp2, des2 = sift.detectAndCompute(src2, None)
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        else:
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(src1, None)
            kp2, des2 = orb.detectAndCompute(src2, None)
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                                table_number=6,
                                key_size=12,
                                multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # print(matches)
        good = []
        for m in matches:
            if len(m) == 2:
                if m[0].distance < 0.7 * m[1].distance:
                    good.append(m[0])
            # else:
            #     print(m)
        # done_match = len(good) > MIN_MATCH_COUNT
        if len(good) > MIN_MATCH_COUNT:
            # pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            # pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            # H = np.eye(3, dtype=M.dtype)
            # H[:2, 2] = M[:2, 2]
            x_t = np.median([(kp2[m.trainIdx].pt[0] - kp1[m.queryIdx].pt[0]) for m in good])
            y_t = np.median([(kp2[m.trainIdx].pt[1] - kp1[m.queryIdx].pt[1]) for m in good])
            # print(f'X_t={x_t:.2f}, Y_t={y_t:.2f}')
            H = np.array([[1, 0, x_t],
                          [0, 1, y_t]])
            translation = (x_t, y_t)
            # x_t, y_t = np.abs(M[:2, 2]).round().astype('int').tolist()
            # src1 = src1[y_t:]
            # src2 = src2[:src2.shape[0] - y_t]
            # src1 = cv2.warpPerspective(src1, M, src1.shape[::-1])
            src1 = cv2.warpAffine(src1, H, src1.shape[::-1])
            # src1 = src1[edge:src1.shape[0] - edge, edge:src1.shape[1] - edge]
            # src2 = src2[edge:src2.shape[0] - edge, edge:src2.shape[1] - edge]
            # print(src1.shape, src2.shape)
            # vis = np.hstack([src1, src2])
            # good = [m for m, n in matches if m.distance < 0.7 * n.distance]
            # draw_params = dict(matchColor=(0, 255, 0),
            #                    singlePointColor=None,
            #                    matchesMask=None,
            #                    flags=2)
            # vis = cv2.drawMatches(src1, kp1, src2, kp2, good, None, **draw_params)
            # cv2.namedWindow('match', cv2.WINDOW_NORMAL)
            # cv2.imshow('match', vis)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # t_arr = np.array([abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) for m in good])
        else:
            print('Not Enough Matches are Found!')

    except Exception as e:
        print('My Exception: ', e)

    return translation, src1, src2


# def stitch(src1, src2, translation):
#     # assert src1.shape == src2.shape
#     x_t, y_t = translation
#     h, w = src1.shape
#     # pano = np.zeros((h + abs(y_t), w - abs(x_t)))
#     # x_slice = slice(x_t, None) if x_t >= 0 else slice(x_t)
#     if x_t >= 0 and y_t >= 0:
#         pano = np.vstack((src1))


if __name__ == "__main__":
    img = cv2.imread('samples/pic1.jpg', 0)
    t_real = 400
    img2 = img.copy()
    img2[:t_real] = 255
    img2[t_real:] = img[:-t_real]
    origin = np.hstack([img, img2])
    cv2.namedWindow('origin', cv2.WINDOW_NORMAL)
    cv2.imshow('origin', origin)
    calc_translation(img, img2)
    # res = np.hstack([img[:-t_test], img2[t_test:]])
    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    # cv2.imshow('result', res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
