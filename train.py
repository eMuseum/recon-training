import numpy as np
from os import listdir

# Try to load cPickle, fallback to pickle
try:
   import cPickle as pickle
except:
   import pickle

# Make sure that Caffe is on the python path:
caffe_root = 'CAFFE_ROOT'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe
import caffe.imagenet

# Use the imagenet classifier provided by Caffe
net = caffe.imagenet.ImageNetClassifier(caffe_root + 'examples/imagenet/imagenet_deploy.prototxt',
                                        caffe_root + 'examples/imagenet/caffe_reference_imagenet_model')

# Set to test and CPU only
net.caffenet.set_phase_test()
net.caffenet.set_mode_cpu()

# Paths to the images folders, it should be improved
# TODO Dynamically fetch from database
images_path=['/data/pis_12/Guernica/',
             '/data/pis_12/abside de San Clemente de Tahull/',
             '/data/pis_12/Bedroom in Arles/',
             '/data/pis_12/ca,talan landscape/',
             '/data/pis_12/Ciencia y Caridad al Cubierto/',
             '/data/pis_12/condensation cube/',
             '/data/pis_12/pepa/',
             '/data/pis_12/la masia/',
             '/data/pis_12/La minotauromaquia/',
             '/data/pis_12/La noche estrellada/',
             '/data/pis_12/Las meninaS/',
             '/data/pis_12/la ultima cena/',
             '/data/pis_12/peristencia de la memoria/',
             '/data/pis_12/Port Alguer/',
             '/data/pis_12/the great masturbator/',
             '/data/pis_12/the temptation of saint anthony/'
             ]

# Create a tuple for classes (y) and one for feats
y = []
label_class=1;
feat_vector = []

# Neuralnet layer from which we get features (up to fc8)
layer = 'fc6'

# Foreach image, get feats
for c in images_path:
    files=listdir(caffe_root + c)
    print "**********************"
    print c
	
    for filename in files:
        print filename 
        scores = net.predict(caffe_root + c + filename)
        feat = net.caffenet.blobs[layer].data[4]
        feat_vector.append(feat.flatten())
        y.append(label_class)
		
    label_class=label_class+1

# Learn classifier
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import confusion_matrix

# Get data
data = np.array(feat_vector)
y = np.array(y)

# Define classifier
clf = svm.SVC(C=1, kernel='linear', probability=True);
clf.fit(data, y)
    
# Save classifier
f = open(caffe_root + "/data/pis_12/clf", "wb+")
pickle.dump(clf, f)
f.flush()
f.close()
