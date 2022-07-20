# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from time import time
import logging
from os.path import join, dirname
import sys
import os

_log_path = join(dirname(__file__), '../logs')
def mk_save(log_dir, shell_file, net_file, main_file):  
    shutil.copy(shell_file, log_dir)
    shutil.copy(net_file, log_dir)
    shutil.copy(main_file, log_dir)

def init_log(log_dir, log_name):  
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=join(log_dir, '{}.log'.format(log_name)),
                        filemode='w')
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging