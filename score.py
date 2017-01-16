#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from sklearn import metrics
from math import sqrt


if len( sys.argv ) != 3:
		print "\nUsage: ", sys.argv[0], "<predicted scores file> <reference scores file> \n"
		exit()

predic, ref = sys.argv[1:]

predic = [ np.float32( item ) for item in open( predic ).readlines() ]
ref = [ np.float32( item ) for item in open( ref ).readlines() ]

mae = metrics.mean_absolute_error( ref, predic )
rmse = sqrt( metrics.mean_squared_error( ref, predic ) )

print "MAE:", mae
print "RMSE:", rmse
