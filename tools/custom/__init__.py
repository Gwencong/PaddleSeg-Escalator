from pathlib import Path
ROOT = Path(__file__).parent.parent.parent.resolve().absolute()
dir_path = Path(__file__).parent.resolve().absolute()

import sys
sys.path.append(dir_path.__str__())
import json2labelImg_custom