#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from math import sqrt

if len( sys.argv ) != 5:
	print( "\nUsage: ", sys.argv[0], "<libsvm train (string)> <n fold (int)> <nb threads (int)> <output prefix>\n" )
	exit()

feat, nfold, nbthreads, out = sys.argv[1:]
nfold = int( nfold )
nbthreads = int( nbthreads )

C = np.logspace( -12.0, 6.0, num = 20, endpoint = True, base = 2 )
G = np.logspace( -10.0, -0.1, num = 15, endpoint = True, base = 2 )
E = np.logspace( -8.0, -4.0, num = 8, endpoint = True, base = 2 )

X, y = load_svmlight_file( feat )
params = [ { 'kernel': [ 'rbf' ], 'gamma': G, 'C': C, 'epsilon': E } ]

score = 'neg_mean_absolute_error'
clf = GridSearchCV( SVR( cache_size = 16384 ), params, cv = 10, n_jobs = nbthreads, scoring = score )

for i in range( nfold ):
	print( "Fold", i )
	X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.1 )
	clf.fit( X_train, y_train )
	print( clf.best_params_ )
	y_pred = clf.predict( X_test )
	with open( out + ".fold" + str( i ) + ".pred", "w" ) as outfile:
		outfile.write( "\n".join( [ str( item ) for item in y_pred ] ) )
	print( "Pearson r", metrics.pearson( y_test, y_pred ) )
