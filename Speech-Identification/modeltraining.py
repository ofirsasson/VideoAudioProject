import sys
import os
import cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GMM 
from featureextraction import extract_features

import warnings
warnings.filterwarnings("ignore")


def main(ARGS):
	source = ARGS.source+"/"   
	if not os.path.isdir(ARGS.model):
	 os.mkdir(ARGS.model)
	dest = ARGS.model+"/" #"TrySpeakers_models/"
	
	train_file = ARGS.text #"TrainingDataPath.txt"        
	file_paths = open(train_file,'r')
	
	count = 1

	# Extracting features for each speaker
	features = np.asarray(())
	for path in file_paths:    
	    path = path.strip()   
	    print path
    	
    	    # Read the audio
	    sr,audio = read(source + path)
    	
	    # Extract 40 dimensional MFCC & delta MFCC features
	    vector = extract_features(audio,sr)
	    if features.size == 0:
	        features = vector
	    else:
	        features = np.vstack((features, vector))

	    # When features of 5 files of speaker are concatenated, then do model training
	    if count == int(ARGS.number):    
	        gmm = GMM(n_components = 16, n_iter = 200, covariance_type='diag',n_init = 3)
	        gmm.fit(features)
        	
        	# Dumping the trained gaussian model
                if ARGS.name:
                 picklefile = ARGS.name+".gmm"
                else:
        	 picklefile = path.split("-")[0]+".gmm"
        	cPickle.dump(gmm,open(dest + picklefile,'w'))
        	print '+ modeling completed for speaker:',picklefile," with data point = ",features.shape    
        	features = np.asarray(())
        	count = 0
	    count = count + 1
	
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="Training GMM model from recordings")

    parser.add_argument('-s', '--source', required=True,
                        help="Path to training data")

    parser.add_argument('-m', '--model', required=True,
                        help="Path where training model will be saved")

    parser.add_argument('-t', '--text', required=True,
                        help="Path to training data path file (.txt)")

    parser.add_argument('-n', '--name', required=True,
                        help="Name for the new model")
    parser.add_argument('-num', '--number', required=True,
                        help="number of records to train with (lines in txt file)")

    ARGS = parser.parse_args()
    main(ARGS)
