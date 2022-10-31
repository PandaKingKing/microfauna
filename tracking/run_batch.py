from pathlib import Path
from video_tracking import video_tracking

# Basic Parameters
wd = Path("Z:/ROMIDAS0.3/ROMIDAS-NAS/D")
video_dir = wd / '123/'
data_dir = wd / 'data'
video_clip = (1500, 3000)
# Further Parameters (No change needed generally)
gap = 2
edge = 50
diff_stat_th = 5
th = 20
ksize = (3, 19)
cnt_area_range = (500, 5000)
pt_dist_range = (0, 100)
vis = True
# (Reading the function docstring for more information)

for video_path in video_dir.glob('*.mp4'):
    print(f'-' * 10 + video_path.name + '-' * 10)
    video_tracking(str(video_path), gap, data_dir, edge, diff_stat_th, th,
                   ksize, cnt_area_range, pt_dist_range, video_clip, vis)
