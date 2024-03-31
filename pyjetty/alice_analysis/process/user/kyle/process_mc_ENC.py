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
class EEC_pair:
	def __init__(self, _index1, _index2, _weight, _r, _pt):
		self.index1 = _index1
		self.index2 = _index2
		self.weight = _weight
		self.r = _r
		self.pt = _pt

	def is_equal(self, pair2):
		return (self.index1 == pair2.index1 and self.index2 == pair2.index2) \
			or (self.index1 == pair2.index2 and self.index2 == pair2.index1)
	
	def __str__(self):
		return "EEC pair with (index1, index2, weight, RL, pt) = (" + \
			str(self.index1) + ", " + str(self.index2) + ", " + str(self.weight) + \
			", " + str(self.r) + ", " + str(self.pt) + ")"


################################################################
class ProcessMC_ENC(process_mc_base.ProcessMCBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
  
    # Initialize base class
    super(ProcessMC_ENC, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)
    
    # find pt_hat for set of events in input_file, assumes all events in input_file are in the same pt_hat bin
    self.pt_hat_bin = int(input_file.split('/')[len(input_file.split('/'))-4]) # depends on exact format of input_file name
    with open("/global/cfs/projectdirs/alice/alicepro/hiccup/rstorage/alice/data/LHC18b8/scaleFactors.yaml", 'r') as stream:
        pt_hat_yaml = yaml.safe_load(stream)
    self.pt_hat = pt_hat_yaml[self.pt_hat_bin]
    print("pt hat bin : " + str(self.pt_hat_bin))
    print("pt hat weight : " + str(self.pt_hat))

  #---------------------------------------------------------------
  # Calculate pair distance of two fastjet particles
  #---------------------------------------------------------------
  def calculate_distance(self, p0, p1):   
    dphiabs = math.fabs(p0.phi() - p1.phi())
    dphi = dphiabs

    if dphiabs > math.pi:
      dphi = 2*math.pi - dphiabs

    deta = p0.eta() - p1.eta()
    return math.sqrt(deta*deta + dphi*dphi)

  #---------------------------------------------------------------
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_user_output_objects(self):
    
    # python array with the format (faster than np array!)
    # ['gen_energy_weight', 'gen_R_L', 'gen_jet_pt', 'obs_energy_weight', 'obs_R_L', 'obs_jet_pt', 'pt_hat']
    name = 'preprocessed_np_mc_eec'
    h = []
    setattr(self, name, h)


  def analyze_matched_pairs(self, det_jets, truth_jets):
    # assumes det and truth parts are matched beforehand:
    # matching particles are given matching user_index s
    # if some det or truth part does not have a match, it is given a unique index
    # also assumes all jet and particle level cuts have been applied already
    # TODO this is untested

	  # truth level EEC pairs
    truth_pairs = []
    for jet in truth_jets:
      truth_pairs += self.get_EEC_pairs(jet, ipoint=2)

    # det level EEC pairs
    det_pairs = []
    for jet in det_jets:
      det_pairs += self.get_EEC_pairs(jet, ipoint=2)

    ########################## TTree output generation #########################
	  # composite of truth and smeared pairs, fill the TTree preprocessed
    dummyval = -9999

    # pair mathcing
    for t_pair in truth_pairs:

        gen_energy_weight = t_pair.weight
        gen_R_L = t_pair.r
        gen_jet_pt = t_pair.pt
        obs_thrown = 0

        match_found = False
        for d_pair in det_pairs:
          if d_pair.is_equal(t_pair):
            obs_energy_weight = d_pair.weight
            obs_R_L = d_pair.r
            obs_jet_pt = d_pair.pt
            match_found = True
            break
        if not match_found:
          obs_energy_weight = dummyval
          obs_R_L = dummyval
          obs_jet_pt = dummyval
          obs_thrown = 1
        
        # find TTree
        name = 'preprocessed_np_mc_eec'
        getattr(self, name).append([gen_energy_weight, gen_R_L, gen_jet_pt, obs_energy_weight, obs_R_L, obs_jet_pt, self.pt_hat, self.event_number])
        
    """
    line = ""
    for part in fj_particles_truth:
        line += str(part.user_index()) + " "
    print(line)
    
    line = ""
    for part in fj_particles_det:
        line += str(part.user_index()) + " "
    print(line)
    """


  def get_EEC_pairs(self, jet, ipoint=2):
    pairs = []

    jet_pt = jet.perp()

    #push constutents to a vector in python
    _v = fj.vectorPJ()
    _ = [_v.push_back(c) for c in jet.constituents()]

    # n-point correlator with all charged particles
    max_npoint = 2
    weight_power = 1
    dphi_cut = -9999
    deta_cut = -9999
    cb = ecorrel.CorrelatorBuilder(_v, jet_pt, max_npoint, weight_power, dphi_cut, deta_cut)

    EEC_cb = cb.correlator(ipoint)

    EEC_weights = EEC_cb.weights() # cb.correlator(npoint).weights() constains list of weights
    EEC_rs = EEC_cb.rs() # cb.correlator(npoint).rs() contains list of RL
    EEC_indicies1 = EEC_cb.indices1() # contains list of 1st track in the pair (index should be based on the indices in c_select)
    EEC_indicies2 = EEC_cb.indices2() # contains list of 2nd track in the pair

    for i in range(len(EEC_rs)):
      event_index1 = _v[EEC_indicies1[i]].user_index()
      event_index2 = _v[EEC_indicies2[i]].user_index()
      pairs.append(EEC_pair(event_index1, event_index2, EEC_weights[i], EEC_rs[i], jet_pt))

    return pairs

  def fill_matched_jet_histograms(self, det_jet_matched, truth_jet_matched):
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

  # perform analysis
  analysis = ProcessMC_ENC(input_file=args.inputFile, config_file=args.configFile, output_dir=args.outputDir)
  analysis.process_mc()
