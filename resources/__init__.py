# resources/__init__.py
# -*- coding: utf-8 -*-
"""资源目录"""

import os

# 资源路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SAVE_DATA_DIR = os.path.join(BASE_DIR, 'save_data')
TEST_FILES_DIR = os.path.join(BASE_DIR, 'TestFiles')
FONT_DIR = os.path.join(BASE_DIR, 'Font')
