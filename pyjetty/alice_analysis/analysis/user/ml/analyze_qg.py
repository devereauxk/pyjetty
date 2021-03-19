#!/usr/bin/env python3

"""
Example class to read quark-gluon dataset
"""

import os
import argparse
import yaml
import h5py

# Data analysis and plotting
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Energy flow package
import energyflow
import energyflow.archs

# sklearn
import sklearn
import sklearn.linear_model
import sklearn.ensemble

# Tensorflow and Keras
import tensorflow as tf
from tensorflow import keras

# Base class
from pyjetty.alice_analysis.process.base import common_base

################################################################
class AnalyzeQG(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
        super(common_base.CommonBase, self).__init__(**kwargs)
        
        self.config_file = config_file
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize config file
        self.initialize_config()
        
        print(self)
        print()
        
    #---------------------------------------------------------------
    # Initialize config file into class members
    #---------------------------------------------------------------
    def initialize_config(self):
    
        # Read config file
        with open(self.config_file, 'r') as stream:
          config = yaml.safe_load(stream)
          
        self.train = config['n_train']
        self.val = config['n_val']
        self.test = config['n_test']
        self.models = config['models']
        self.K = config['K']

    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def analyze_qg(self):
    
        # Read input variables
        with h5py.File(os.path.join(self.output_dir, 'nsubjettiness.h5'), 'r') as hf:
            y = hf['y'][:]
            X = hf['X'][:]
            X_Nsub = hf['X_Nsub'][:]

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # Split data into train and test sets (extend to include a validation set as well. See: data_split?)
        test_frac = 0.2
        (X_Nsub_train, X_Nsub_test, y_Nsub_train, y_Nsub_test) = energyflow.utils.data_split(X_Nsub, y, val=0, test=test_frac)

        # Fit ML model -- 1. SGDClassifier
        if 'linear' in self.models:
        
            print('Training SGDClassifier')
            sgd_clf = sklearn.linear_model.SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
            sgd_clf.fit(X_Nsub_train, y_Nsub_train)
            
            # Use cross validation predict (here split training set) and compute the confusion matrix from the predictions
            y_Nsub_crossval_SGD = sklearn.model_selection.cross_val_predict(sgd_clf, X_Nsub_train, y_Nsub_train, cv=3,method="decision_function")
            #confusion_SGD = sklearn.metrics.confusion_matrix(y_Nsub_train, y_Nsub_crossval_SGD)
            #print('Confusion matrix for SGD Classifier (test set): \n {}'.format(confusion_SGD))

            # Get predictions for the test set .. actually don't need this when using cross_val_predict (?)
            # preds_Nsub_SGD = sgd_clf.predict(X_Nsub_test)
            
            # Get AUC from training process
            Nsub_auc_SGD = sklearn.metrics.roc_auc_score(y_Nsub_train, y_Nsub_crossval_SGD)
            print('SGDClassifier: AUC = {} (cross validation)'.format(Nsub_auc_SGD))
            
            # Compute ROC curve: the roc_curve() function expects labels and scores
            Nsub_fpr_SGD, Nsub_tpr_SGD, thresholds = sklearn.metrics.roc_curve(y_Nsub_train, y_Nsub_crossval_SGD)
            
            # Plot ROC curve for SGDClassifier
            self.plot_roc_curve(Nsub_fpr_SGD, Nsub_tpr_SGD, label1='SGDClassifier')
            print()
            
            # Check number of threhsolds used for ROC curve
            # n_thresholds = len(np.unique(y_Nsub_scores_SGD)) + 1

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
        # Fit ML model -- 2. Random Forest Classifier
        if 'random_forest' in self.models:
            
            forest_clf = sklearn.ensemble.RandomForestClassifier(random_state=42)
            y_Nsub_probas_forest = sklearn.model_selection.cross_val_predict(forest_clf, X_Nsub_train, y_Nsub_train, cv=3,method="predict_proba")
            
            # The output here are class probabilities. We us use the positive class's probability for the ROC curve
            y_Nsub_scores_forest = y_Nsub_probas_forest[:,1]
            
            print(y_Nsub_scores_forest)
            
            # Compute AUC & ROC curve
            Nsub_auc_RFC = sklearn.metrics.roc_auc_score(y_Nsub_train,y_Nsub_scores_forest)
            print('Random Forest Classifier: AUC = {} (cross validation)'.format(Nsub_auc_RFC))
            Nsub_fpr_forest, Nsub_tpr_forest, thresholds_forest = sklearn.metrics.roc_curve(y_Nsub_train,y_Nsub_scores_forest)
            
            # Plot ROC curve
            self.plot_roc_curve(Nsub_fpr_SGD,Nsub_tpr_SGD,Nsub_fpr_forest, Nsub_tpr_forest,"SGD_Nsub","RF_Nsub")
            print()

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
        # Fit ML model -- 3. Dense Neural network with Keras
        if 'neural_network' in self.models:
        
            # input_shape expects shape of an instance (not including batch size)
            DNN = keras.models.Sequential()
            DNN.add(keras.layers.Flatten(input_shape=[X_Nsub_train.shape[1]]))
            DNN.add(keras.layers.Dense(300,activation='relu'))
            DNN.add(keras.layers.Dense(300,activation='relu'))
            DNN.add(keras.layers.Dense(100,activation='relu'))
            DNN.add(keras.layers.Dense(1,activation='sigmoid')) # softmax? # Last layer has to be 1 or 2 for binary classification?

            # Print DNN summary
            DNN.summary()
            
            # Compile DNN
            opt = keras.optimizers.Adam(learning_rate=0.001) # lr = 0.001 (cf 1810.05165)
            DNN.compile(loss="binary_crossentropy",          # Loss function - use categorical_crossentropy instead ?
                        optimizer=opt,                       # For Stochastic gradient descent use: "sgd"
                        metrics=["accuracy"])                # Measure accuracy during training

            # Train DNN - need validation set here (use test set for now)
            DNN.fit(X_Nsub_train,y_Nsub_train, epochs=39, validation_data=(X_Nsub_test,y_Nsub_test))
            
            # Get predictions for validation data set
            y_Nsub_test_preds_DNN = DNN.predict(X_Nsub_test).reshape(-1)
            
            # Get AUC
            Nsub_auc_DNN = sklearn.metrics.roc_auc_score(y_Nsub_test,y_Nsub_test_preds_DNN)
            print('Dense Neural Network: AUC = {} (validation set)'.format(Nsub_auc_DNN))
            
            # Get ROC curve results
            Nsub_fpr_DNN, Nsub_tpr_DNN, thresholds = sklearn.metrics.roc_curve(y_Nsub_test,y_Nsub_test_preds_DNN)
            
            # Plot ROC curve
            self.plot_roc_curve(Nsub_fpr_SGD,Nsub_tpr_SGD,Nsub_fpr_DNN, Nsub_tpr_DNN,"SGD_Nsub","DNN_Nsub")
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
        # Fit ML model -- 4. Deep Set/Particle Flow Networks
        if 'pfn' in self.models:

            # network architecture parameters
            Phi_sizes, F_sizes = (100, 100, 256), (100, 100, 100)
            
            # network training parameters
            num_epoch = 3
            batch_size = 500
            
            # Use PID information
            use_pids = True
            
            # convert labels to categorical
            Y_PFN = energyflow.utils.to_categorical(y, num_classes=2)
            
            # preprocess by centering jets and normalizing pts
            X_PFN = X
            for x_PFN in X_PFN:
                mask = x_PFN[:,0] > 0
                yphi_avg = np.average(x_PFN[mask,1:3], weights=x_PFN[mask,0], axis=0)
                x_PFN[mask,1:3] -= yphi_avg
                x_PFN[mask,0] /= x_PFN[:,0].sum()
            
            # handle particle id channel [?? ... remap_pids is not working]
            #if use_pids:
            #    energyflow.utils.remap_pids(X_PFN, pid_i=3)
            #else:
            X_PFN = X_PFN[:,:,:3]

            # Split data into train, val and test sets
            (X_PFN_train, X_PFN_val, X_PFN_test,Y_PFN_train, Y_PFN_val, Y_PFN_test) = energyflow.utils.data_split(X_PFN, Y_PFN,
                                                                                                 val=self.val, test=self.test)
            # build architecture
            pfn = energyflow.archs.PFN(input_dim=X_PFN.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes)

            # train model
            pfn.fit(X_PFN_train, Y_PFN_train,
                    epochs=num_epoch,
                    batch_size=batch_size,
                    validation_data=(X_PFN_val, Y_PFN_val),
                    verbose=1)
            
            # get predictions on test data
            preds_PFN = pfn.predict(X_PFN_test, batch_size=1000)

            # Get AUC and ROC curve + make plot
            auc_PFN = sklearn.metrics.roc_auc_score(Y_PFN_test[:,1], preds_PFN[:,1])
            print('Particle Flow Networks/Deep Sets: AUC = {} (test set)'.format(auc_PFN))
            
            fpr_PFN, tpr_PFN, threshs = sklearn.metrics.roc_curve(Y_PFN_test[:,1], preds_PFN[:,1])
            self.plot_roc_curve(Nsub_fpr_SGD,Nsub_tpr_SGD,fpr_PFN, tpr_PFN,"SGD_Nsub","PFN_woPID")
            
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            
            # Now we compare the PFN ROC curve to single observables

            # 1. Jet mass (Note: This takes in (pt,y,phi) and converts it to 4-vectors and computes jet mass)
            #             (Note: X_PFN_train is centered and normalized .. should be ok)
            masses = np.asarray([energyflow.ms_from_p4s(energyflow.p4s_from_ptyphims(x).sum(axis=0)) for x in X_PFN_train])
            mass_fpr, mass_tpr, threshs = sklearn.metrics.roc_curve(Y_PFN_train[:,1], -masses)
            
            # 2. Multiplicity (Is this a useful observable for pp vs AA?)
            mults = np.asarray([np.count_nonzero(x[:,0]) for x in X_PFN_train])
            mult_fpr, mult_tpr, threshs = sklearn.metrics.roc_curve(Y_PFN_train[:,1], -mults)
            
            # Make ROC curve plots
            self.plot_roc_curve(mass_fpr,mass_tpr,fpr_PFN, tpr_PFN,"Jet_mass","PFN_woPID")
            self.plot_roc_curve(mult_fpr,mult_tpr,fpr_PFN, tpr_PFN,"Multiplicity","PFN_woPID")
        
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
        # Do we need to train a DNN with 2 variables if we want to look at the discriminating power
        # of mass and multiplicity or just pass 2 features to the ROC curve?
        
    #--------------------------------------------------------------- 
    # Plot ROC curve                                                 
    #--------------------------------------------------------------- 
    def plot_roc_curve(self, fpr1=None, tpr1=None, fpr2=None, tpr2=None, label1=None, label2=None):
    
        plt.plot(fpr1, tpr1, "b:", label=label1)
        if label2:
            plt.plot(fpr2, tpr2, linewidth=2, label=label2)
        plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
        plt.axis([0, 1, 0, 1])                                   
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16) 
        plt.grid(True)    
        plt.legend(loc="lower right")        
        plt.tight_layout()
        
        outputfilename = 'ROC_{}'.format(label1)
        if label2:
            outputfilename += '_{}'.format(label2)
        plt.savefig(os.path.join(self.output_dir, '{}.pdf'.format(outputfilename)))
        plt.close()
            
##################################################################
if __name__ == '__main__':

    # Define arguments
    parser = argparse.ArgumentParser(description='Process qg')
    parser.add_argument('-c', '--configFile', action='store',
                        type=str, metavar='configFile',
                        default='../../../config/ml/qg.yaml',
                        help='Path of config file for analysis')
    parser.add_argument('-o', '--outputDir', action='store',
                        type=str, metavar='outputDir',
                        default='./TestOutput',
                        help='Output directory for output to be written to')

    # Parse the arguments
    args = parser.parse_args()

    print('Configuring...')
    print('configFile: \'{0}\''.format(args.configFile))
    print('ouputDir: \'{0}\"'.format(args.outputDir))

    # If invalid configFile is given, exit
    if not os.path.exists(args.configFile):
        print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
        sys.exit(0)

    analysis = AnalyzeQG(config_file=args.configFile, output_dir=args.outputDir)
    analysis.analyze_qg()