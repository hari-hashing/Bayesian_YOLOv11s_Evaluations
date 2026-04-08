# Bayesian_YOLOv11s_Evaluations

Dataset and final trained checkpoints : https://drive.google.com/drive/folders/1yYtKpg4lmhyWATWLOF-Pa1c-5OGPoJeH?usp=sharing

# Download the image folder using the following code snippet 
```python
from pathlib import Path
import yaml
from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import download

# Download labels
segments = True  # segment or box labels
dir = "/home/kunal-pg/VLA_0/vla0/Bayesian/BNN"  # dataset root dir

# Uncomment to download COCO labels
# urls = [ASSETS_URL + ("/coco2017labels-segments.zip" if segments else "/coco2017labels.zip")]  # labels
# urls = [ASSETS_URL]
# download(urls, dir=dir)

# Download COCO dataset images
urls = [
    "http://images.cocodataset.org/zips/train2017.zip",  # 19G, 118k images
    "http://images.cocodataset.org/zips/val2017.zip",    # 1G, 5k images  
    "http://images.cocodataset.org/zips/test2017.zip",   # 7G, 41k images (optional)
]
download(urls, dir=dir, threads=3)
```
