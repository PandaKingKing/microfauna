import numpy as np
import cv2
import features_match
from pathlib import Path
import matplotlib.pyplot as plt
from track import calc_distance, Track
import pandas as pd


def tracks_analysis(tracks, frames_count, fps, vis=False):
    # idx = ['mean'] + [str(i) for i in range(round(frames_count))]
    df = pd.DataFrame(index=range(round(frames_count + 1)))
    count = 0
    ax = plt.axes()
    for tr in tracks:
        if len(tr) < 3:
            continue
        count += 1
        velocity, omega = tr.motion_analysis(fps)
        col_name = f'Target{count}-StartFrame{tr.start_frame:.0f}'
        nrows = len(tr)
        df.loc[1:nrows, col_name + '_track-x'] = [pt[0] for pt in tr.pts]
        df.loc[1:nrows, col_name + '_track-y'] = [pt[1] for pt in tr.pts]
        df.loc[:nrows, col_name + '_velocity'] = [np.mean(velocity), np.nan] + velocity
        df.loc[:nrows, col_name + '_omega'] = [np.mean(omega), np.nan, np.nan] + omega
        tr = np.array(tr.pts)
        ax.plot(tr[:, 0], tr[:, 1], '-')

    df.columns = df.columns.map(lambda x: tuple(x.split('_')))
    df = df.rename(index={0: 'mean'})

    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    if vis:
        plt.show()

    return df, count


def video_tracking(video_id: str, gap: int = 2, data_dir: Path = Path('data'), edge: int = 50,
                   diff_stat_th: int = 5, th: int = 20, ksize: tuple = (3, 19),
                   cnt_area_range: tuple = (500, 5000), pt_dist_range: tuple = (0, 100),
                   video_clip: tuple = (100, 1400), vis: bool = False) -> None:
    """
    This Function is to apply Tracking Method to a Video.

    :param video_id: Path of the video
    :param gap: Gap of the frames reading
    :param data_dir: Path of the xlsx files to save
    :param edge: The length of image edge to remove
    :param diff_stat_th: Threshold of difference between adjacent frames
    :param th: Threshold to segment the grayscale image into binary image
    :param ksize: The set of kernel size values used in morphology
    :param cnt_area_range: Range of contour area scale
    :param pt_dist_range: Range of distance between adjacent points in one track
    :param video_clip: The set of parameters used in video clipping
    :param vis: Whether to visualize
    """
    # edge = 10 * gap
    assert video_clip[0] > 0 and video_clip[1] > 0
    data_path = data_dir / (Path(video_id).stem + f'_{video_clip[0]}_{video_clip[1]}.xlsx')
    xlsx_wt = pd.ExcelWriter(str(data_path))
    if vis:
        cv2.namedWindow('match', cv2.WINDOW_NORMAL)
        cv2.namedWindow('diff', cv2.WINDOW_NORMAL)
        cv2.namedWindow('target', cv2.WINDOW_NORMAL)
        cv2.waitKey(0)
    cap = cv2.VideoCapture(video_id)
    if not cap.isOpened():
        print('Video not existed!')
        exit()
    if video_clip[0] < video_clip[1]:
        clip_start, clip_end = video_clip
    else:
        clip_start, clip_end = 0, cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # vid_path = vid_dir / Path(video_id).with_suffix('.avi').name
    # vid_out = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*'XVID'),
    #                           20.0, (int(width-2*edge), int(height-2*edge)))
    tracks, x_left, x_right, y_base, y_step = [], 50, 500, 100, 100
    ret, frame_pre = cap.read()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if frame_num > clip_end:
            df, tracks_count = tracks_analysis(tracks, (clip_end - clip_start) / gap, fps / gap)
            df.to_excel(xlsx_wt, )
            if vis:
                print(f'Video Clip {clip_start:.0f}-{clip_end:.0f}')
                print(f'Number of Targets Detected is {tracks_count}')
                print('---------------------------------------------------')
            break
        if frame_num % gap != 0:
            continue
        frame_roi = frame[edge:-edge, edge:-edge] if edge else frame
        panel = np.full_like(frame_roi, 255)
        if vis:
            cv2.putText(panel, 'TargetNum', (x_left, y_base),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(panel, 'LinearVelocity', (x_right, y_base),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5, cv2.LINE_AA)
        if frame_num % video_clip[0] < video_clip[1]:
            translation, img1, img2 = features_match.calc_translation(frame_pre, frame)
            frame_pre = frame.copy()
            if len(translation) == 0:
                if vis:
                    if cv2.waitKey(0) == ord('q'): break
                continue
            if edge:
                img1 = img1[edge:-edge, edge:-edge]
                img2 = img2[edge:-edge, edge:-edge]
            diff = cv2.absdiff(img1, img2)
            diff_stat = np.median(diff)
            # print(f"Difference between frames is {diff_stat}")
            mth = cv2.hconcat([img1, img2])
            if diff_stat > diff_stat_th:
                print('Bad Matching!')
                if vis:
                    if cv2.waitKey(0) == ord('q'): break
                continue
            diff = cv2.GaussianBlur(diff, (5, 5), 0)
            # th = np.sort(diff.flatten())[-2000]
            # print(f'Threshold is {th}')
            ret, binary = cv2.threshold(diff, th, 255, cv2.THRESH_BINARY)
            ksize1, ksize2 = ksize
            kernel1 = np.ones((ksize1, ksize1), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel1)
            kernel2 = np.ones((ksize2, ksize2), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel2)
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pts = []
            for cnt in cnts:
                if cnt_area_range[0] < cv2.contourArea(cnt) < cnt_area_range[1]:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx, cy = x + w / 2 + edge, y + h / 2 + edge
                    if calc_distance((cx, cy), (frame.shape[1] / 2, frame.shape[0] / 2)) > 100:
                        pts.append((round(cx), round(cy)))
                        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
                        # cv2.drawContours(frame_roi, [cnt], 0, (255, 0, 0), 5)
                elif vis:
                    cv2.drawContours(binary, [cnt], 0, 0, -1)
            if vis:
                cv2.imshow('match', mth)
                cv2.imshow('diff', binary)
            if tracks:
                new_tracks, i = [], 0
                for tr in tracks:
                    tr.update(translation)
                    ret = tr.associate(pts, dist_range=pt_dist_range) if pts else False
                    if ret or len(tr) > 10:
                        if vis: tr.draw(frame)
                        new_tracks.append(tr)
                        if 0 < tr[-1][0] < frame_roi.shape[1] and 0 < tr[-1][1] < frame_roi.shape[0] and len(tr) > 1:
                            i += 1
                            v = calc_distance(tr[-2], tr[-1]) * fps / gap
                            if vis:
                                cv2.putText(frame, f'{i}', tr[-1],
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 10, cv2.LINE_AA)
                                cv2.putText(panel, f'{i}', (x_left, y_base + i * y_step),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                                cv2.putText(panel, f'{v:.2f} pixels/s', (x_right, y_base + i * y_step),
                                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
                tracks = new_tracks.copy()
            tracks += [Track([pt], start_frame=frame_num) for pt in pts]
            if frame_num % video_clip[0] >= video_clip[1] - gap:
                df, tracks_count = tracks_analysis(tracks, video_clip[1] / gap, fps / gap)
                start = int(frame_num / video_clip[0]) * video_clip[0]
                end = start + video_clip[1]
                df.to_excel(xlsx_wt, sheet_name=f'{start:.0f}-{end:.0f}')
                if vis:
                    print(f'Video Clip {start:.0f}-{end:.0f}')
                    print(f'Number of Targets Detected is {tracks_count}')
                    print('---------------------------------------------------')
                tracks = []
        else:
            frame_pre = frame.copy()
        if vis:
            cv2.putText(frame_roi, f'{frame_num:.0f}', (0, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3, cv2.LINE_AA)
            result = np.hstack((frame_roi, panel))
            cv2.imshow('target', result)
            kk = cv2.waitKey(1)
            if kk == ord('p'):
                kk = cv2.waitKey(0)
            if kk == ord('q'):
                break
    cap.release()
    # vid_out.release()
    if vis:
        cv2.destroyAllWindows()
    try:
        xlsx_wt.save()
    except Exception as e:
        print('My Exception: ', e)


if __name__ == '__main__':
    vid = "F:\code_pz\\tracking\\samples\\video2.mp4"
    video_tracking(vid, vis=True)
