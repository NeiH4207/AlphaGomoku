
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ModelConfig = json.load(open('configs/model.json'))["Model-1"]