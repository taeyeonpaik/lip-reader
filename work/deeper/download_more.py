import sys,json,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path('outputs/lipreading-studio').resolve()))
import training.download as d
d.ROOT=Path('work/overnight/data').resolve()
items=[('train','MahmoodAnaam/LRS2-Train','26dea093e50d1ee89fe717648b50922cc4b4011b',f'data/train-{i:05d}-of-00022.parquet') for i in [4,7,15,19]]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: new=list(pool.map(d.download,items))
old=json.loads((d.ROOT/'downloads.json').read_text())
Path('work/deeper/downloads.json').write_text(json.dumps(old+new,indent=2)+'\n')
