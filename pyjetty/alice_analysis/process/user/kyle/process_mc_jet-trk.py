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

  #---------------------------------------------------------------
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_user_output_objects(self):

    # define binnings
    if self.is_pp:
      n_bins = [12, 15, 7] # WARNING RooUnfold seg faults if too many bins used
      binnings = [np.linspace(0,0.4,n_bins[0]+1), \
      np.array([0.15, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]).astype(float), \
      np.array([10, 20, 40, 60, 80, 100, 120, 140]).astype(float) ]
    else:
      n_bins = [12, 15, 8] # WARNING RooUnfold seg faults if too many bins used
      binnings = [np.linspace(0,0.4,n_bins[0]+1), \
      np.array([0.15, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]).astype(float), \
      np.array([20, 40, 60, 80, 100, 120, 140, 160, 200]).astype(float) ]

    # python array with the format (faster than np array!)
    # ['gen_R', 'gen_trk_pt', 'gen_jet_pt', 'obs_R', 'obs_trk_pt', 'obs_jet_pt', 'pt_hat', 'event_n']
    name = 'preprocessed_np_mc_jettrk'
    h = []
    setattr(self, name, h)

    # for purity correction
    name = 'reco'
    h = ROOT.TH3D("reco", "reco", n_bins[0], binnings[0], n_bins[1], binnings[1], n_bins[2], binnings[2])
    setattr(self, name, h)

    name = 'reco_unmatched'
    h = ROOT.TH3D("reco_unmatched", "reco_unmatched", n_bins[0], binnings[0], n_bins[1], binnings[1], n_bins[2], binnings[2])
    setattr(self, name, h)

    # jet axis before/after embedding study, function of delta R and det jet pT
    name = 'jet_axis_diff'
    N_delta_R_bins = 100
    delta_R_bins = np.linspace(0, 0.2, N_delta_R_bins+1)
    h = ROOT.TH2D("jet_axis_diff", "jet_axis_diff", N_delta_R_bins, delta_R_bins, n_bins[2], binnings[2])
    setattr(self, name, h)

  #---------------------------------------------------------------
  # Analyze jets of a given event.
  #---------------------------------------------------------------
  def analyze_jets(self, jets_det, jets_truth, jetR, rho = 0):
  
    jets_det_selected = fj.vectorPJ()
    jets_pt_selected = []
    for jet in jets_det:

      ############ APPLY LEADING PARTICLE and JET AREA CUTS ######
      # applied only to det/hybrid jets, not to truth jets

      if jet.is_pure_ghost(): continue

      leading_pt = np.max([c.perp() for c in jet.constituents()])

      # jet area and leading particle pt cut
      if jet.area() < 0.6*np.pi*jetR**2 or leading_pt < 5 or leading_pt > 100:
        continue

      jet_pt_corrected = jet.perp() - rho*jet.area()

      if jet_pt_corrected <= self.jetpt_min_det_subtracted:
        continue

      # print("{} -> {}".format(jet.perp(), jet_pt_corrected))

      jets_pt_selected.append(jet_pt_corrected)
      jets_det_selected.push_back(jet)

    if self.debug_level > 1:
      print('Number of det-level jets: {}'.format(len(jets_det_selected)))
      
    ############################## JET MATCHING ##############################
    # perform jet-matching, every det jet has a guaranteed truth jet match
    det_used = []
    for t_jet in jets_truth:
      candidates = []
      candidates_pt = []

      for i in range(jets_det_selected.size()):
        d_jet = jets_det_selected[i]

        if self.calculate_distance(t_jet, d_jet) < 0.2 and d_jet not in det_used:
          candidates.append(d_jet)
          candidates_pt.append(jets_pt_selected[i])

      # if match found
      if len(candidates) > 0:
        winner_arg = np.argmin(np.abs(np.array(candidates_pt) - t_jet.perp()))
        det_match = candidates[winner_arg]
        det_match_pt = candidates_pt[winner_arg]
        det_used.append(det_match)

        getattr(self, "preprocessed_np_mc_jetpt").append([t_jet.perp(), det_match_pt, self.pt_hat])
        # using the rho-subtracted jet pT for the matched reco jet here

        self.fill_matched_jet_histograms(det_match, t_jet, det_match_pt)

      # if match not found, DONT DO ANYTHING, we had a whole conversation about this...
        
        
  #---------------------------------------------------------------
  # This function is called per jet subconfigration 
  # Fill matched jet histograms
  #---------------------------------------------------------------
  def fill_matched_jet_histograms(self, det_jet_matched, truth_jet_matched, det_pt_corrected):

    # assumes det and truth parts are matched beforehand:
    # matching particles are given matching user_index s
    # if some det or truth part does not have a match, it is given a unique index
    # also assumes all jet and particle level cuts have been applied already

    name = 'preprocessed_np_mc_jettrk'

    c_truth = truth_jet_matched.constituents()
    c_det = det_jet_matched.constituents()

    ######### purity correction #########
    # calculate det EEC cross section irregardless if truth match exists

    for d_part in c_det:
      if d_part.perp() < 0.15 or d_part.user_index() < 0: continue # TODO assert not background here?

      det_R = self.calculate_distance(d_part, det_jet_matched)
      getattr(self, "reco_unmatched").Fill(det_R, d_part.perp(), det_pt_corrected, self.pt_hat)


    ######### jet axis before/after embedding check #########
    jetaxis_R = self.calculate_distance(det_jet_matched, truth_jet_matched)
    getattr(self, "jet_axis_diff").Fill(jetaxis_R, det_jet_matched.perp(), self.pt_hat)
    

    ########################## TTree output generation #########################
    # composite of truth and smeared pairs, fill the TTree preprocessed
    dummyval = -9999

    for t_part in c_truth:
      
      # cut on 0.15 track pT again so ghosts not filled 
      if t_part.perp() < 0.15: continue

      truth_R = self.calculate_distance(t_part, truth_jet_matched)

      match_found = False
      for d_part in c_det:

        # cut on 0.15 track pT again so ghosts not filled 
        if d_part.perp() < 0.15: continue

        det_R = self.calculate_distance(d_part, det_jet_matched)
        
        # if truth and det part are matched, and (if embedded event) det particle is not from the background
        # fill row of preprocessed with both info
        if t_part.user_index() == d_part.user_index() and not d_part.user_index() < 0:

          getattr(self, name).append([truth_R, t_part.perp(), truth_jet_matched.perp(), \
                                      det_R, d_part.perp(), det_pt_corrected, self.pt_hat, self.event_number])
          
          getattr(self, "reco").Fill(det_R, d_part.perp(), det_pt_corrected, self.pt_hat)

          # print("{}, {}, {} | {}, {}, {}".format(truth_R, t_part.perp(), truth_jet_matched.perp(), det_R, d_part.perp(), det_pt_corrected))
          #print("{} | {}".format(truth_R, det_R))

          match_found = True
          break

        # if no match is found for this truth particle, fill it in the table, but without any
        # corresponding det data
      if not match_found:
        getattr(self, name).append([truth_R, t_part.perp(), truth_jet_matched.perp(), dummyval, dummyval, dummyval, self.pt_hat, self.event_number])


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