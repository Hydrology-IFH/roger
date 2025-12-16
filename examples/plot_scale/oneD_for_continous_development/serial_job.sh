#!/bin/bash

python oneD.py -ms ihringen
python oneD.py -ms breitnau

python oneD_lbc.py -ms ihringen
python oneD_lbc.py -ms breitnau

python oneD_nosnow_noint.py -ms ihringen
python oneD_nosnow_noint.py -ms breitnau

python oneD_nosnow_noint_noet.py -ms ihringen
python oneD_nosnow_noint_noet.py -ms breitnau
