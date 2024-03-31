#!/usr/bin/env python3

"""
  Analysis class to read a ROOT TTree of MC track information
  and do jet-finding, and save response histograms.
  
  Author: James Mulligan (james.mulligan@berkeley.edu)
"""

from __future__ import print_function

# General
import os
import sys
import argparse

# Data analysis and plotting
import numpy as np
import ROOT
import yaml
import array
import math
# from array import *

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib
import fjtools
import ecorrel

# Analysis utilities
from pyjetty.alice_analysis.process.base import process_io
from pyjetty.alice_analysis.process.base import process_io_emb
from pyjetty.alice_analysis.process.base import jet_info
from pyjetty.alice_analysis.process.user.substructure import process_mc_base
from pyjetty.alice_analysis.process.base import thermal_generator
from pyjetty.mputils.csubtractor import CEventSubtractor

def linbins(xmin, xmax, nbins):
  lspace = np.linspace(xmin, xmax, nbins+1)
  arr = array.array('f', lspace)
  return arr

def logbins(xmin, xmax, nbins):
  lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
  arr = array.array('f', lspace)
  return arr


################################################################
class ProcessMC_JetTrk(process_mc_base.ProcessMCBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
  
    # Initialize base class
    super(ProcessMC_JetTrk, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)
    
    # find pt_hat for set of events in input_file, assumes all events in input_file are in the same pt_hat bin
    self.pt_hat_bin = int(input_file.split('/')[len(input_file.split('/'))-4]) # depends on exact format of input_file name
    with open("/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/data/LHC18b8/scaleFactors.yaml", 'r') as stream:
        pt_hat_yaml = yaml.safe_load(stream)
    self.pt_hat = pt_hat_yaml[self.pt_hat_bin]
    print("pt hat bin : " + str(self.pt_hat_bin))
    print("pt hat weight : " + str(self.pt_hat))

  #---------------------------------------------------------------
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_user_output_objects(self):

    # this is the only thing produced! contains all relevant det and truth-level info to fill all histograms

    # python array with the format (faster than np array!)
    # ['gen_R', 'gen_trk_pt', 'gen_jet_pt', 'obs_R', 'obs_trk_pt', 'obs_jet_pt', 'pt_hat', 'event_n']
    name = 'preprocessed_np_mc_jettrk'
    h = []
    setattr(self, name, h)
        
        
  #---------------------------------------------------------------
  # This function is called per jet subconfigration 
  # Fill matched jet histograms
  #---------------------------------------------------------------
  def fill_matched_jet_histograms(self, det_jet_matched, truth_jet_matched, det_pt_corrected):

    # assumes det and truth parts are matched beforehand:
    # matching particles are given matching user_index s
    # if some det or truth part does not have a match, it is given a unique index
    # also assumes all jet and particle level cuts have been applied already
    # TODO this is untested

    dummy_val = -9999

    name = 'preprocessed_np_mc_jettrk'

    c_truth = truth_jet_matched.constituents()
    c_det = det_jet_matched.constituents()

    ########################## TTree output generation #########################

    for t_part in c_truth:

      truth_R = self.calculate_distance(t_part, truth_jet_matched)

      match_found = False
      for d_part in c_det:

        det_R = self.calculate_distance(d_part, det_jet_matched)
        
        # if truth and det part are matched, and (if embedded event) det particle is not from the background
        # fill row of preprocessed with both info
        if t_part.user_index() == d_part.user_index() and not d_part.user_index() < 0:

          getattr(self, name).append([truth_R, t_part.perp(), truth_jet_matched.perp(), \
                                      det_R, d_part.perp(), det_pt_corrected, self.pt_hat, self.event_number])

          match_found = True 
          break

        # if no match is found for this truth particle, fill it in the table, but without any
        # corresponding det data
      if not match_found:
        getattr(self, name).append([truth_R, t_part.perp(), truth_jet_matched.perp(), dummy_val, dummy_val, dummy_val, self.pt_hat, self.event_number])


  def analyze_matched_pairs(self, fj_particles_det, fj_particles_truth):
    return

##################################################################
if __name__ == '__main__':
  # Define arguments
  parser = argparse.ArgumentParser(description='Process MC')
  parser.add_argument('-f', '--inputFile', action='store',
                      type=str, metavar='inputFile',
                      default='AnalysisResults.root',
                      help='Path of ROOT file containing TTrees')
  parser.add_argument('-c', '--configFile', action='store',
                      type=str, metavar='configFile',
                      default='config/analysis_config.yaml',
                      help="Path of config file for analysis")
  parser.add_argument('-o', '--outputDir', action='store',
                      type=str, metavar='outputDir',
                      default='./TestOutput',
                      help='Output directory for output to be written to')
  
  # Parse the arguments
  args = parser.parse_args()
  
  print('Configuring...')
  print('inputFile: \'{0}\''.format(args.inputFile))
  print('configFile: \'{0}\''.format(args.configFile))
  print('ouputDir: \'{0}\"'.format(args.outputDir))

  # If invalid inputFile is given, exit
  if not os.path.exists(args.inputFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.inputFile))
    sys.exit(0)
  
  # If invalid configFile is given, exit
  if not os.path.exists(args.configFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
    sys.exit(0)

  analysis = ProcessMC_JetTrk(input_file=args.inputFile, config_file=args.configFile, output_dir=args.outputDir)
  analysis.process_mc()
