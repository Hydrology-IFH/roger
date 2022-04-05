#!/usr/bin/env python
# coding=utf-8

import os
import sys

here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(here)
if '/Users/robinschwemmle/anaconda3/envs/roger/lib/python3.8/site-packages/UNKNOWN-0.0.0-py3.8.egg' in sys.path:
    sys.path.remove('/Users/robinschwemmle/anaconda3/envs/roger/lib/python3.8/site-packages/UNKNOWN-0.0.0-py3.8.egg')
if '/Users/robinschwemmle/anaconda3/envs/roger/lib/python3.8/site-packages/roger-0+unknown-py3.8-macosx-10.9-x86_64.egg' in sys.path:
    sys.path.remove('/Users/robinschwemmle/anaconda3/envs/roger/lib/python3.8/site-packages/roger-0+unknown-py3.8-macosx-10.9-x86_64.egg')

del here
