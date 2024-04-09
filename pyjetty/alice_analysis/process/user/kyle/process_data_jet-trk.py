#!/usr/bin/env python3

"""
  Analysis class to read a ROOT TTree of track information
  and do jet-finding, and save basic histograms.
  
  Author: James Mulligan (james.mulligan@berkeley.edu)
"""

from __future__ import print_function

# General
import os
import sys
import argparse
import sys

# Data analysis and plotting
import ROOT
import yaml
import numpy as np
import array 
import math

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib
import ecorrel

# Base class
from pyjetty.alice_analysis.process.user.substructure import process_data_base

def linbins(xmin, xmax, nbins):
  lspace = np.linspace(xmin, xmax, nbins+1)
  arr = array.array('f', lspace)
  return arr

def logbins(xmin, xmax, nbins):
  lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
  arr = array.array('f', lspace)
  return arr

################################################################
class ProcessData_JetTrk(process_data_base.ProcessDataBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
  
    # print(sys.path)
    # print(sys.modules)

    # Initialize base class
    super(ProcessData_JetTrk, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)
    
    self.observable = self.observable_list[0]


  #---------------------------------------------------------------
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_user_output_objects(self):

    name = 'preprocessed_np_data_jettrk'
    h = []
    setattr(self, name, h)

    name = 'preprocessed_np_data_jettrk_bkgd'
    h = []
    setattr(self, name, h)

    for observable in self.observable_list:

      if observable == 'jet_pt_JetPt':
        name = 'h_{}'.format(observable)
        jetpt_bins = linbins(0,200,200)
        h = ROOT.TH1D(name, name, 200, jetpt_bins)
        h.GetXaxis().SetTitle('p_{T jet}')
        h.GetYaxis().SetTitle('Counts')
        setattr(self, name, h)

      if observable == "trk_pt_TrkPt":
        name = 'h_{}'.format(observable)
        trkpt_bins = linbins(0,20,200)
        h = ROOT.TH1D(name, name, 200, trkpt_bins)
        h.GetXaxis().SetTitle('p_{T,ch trk}')
        h.GetYaxis().SetTitle('Counts')
        setattr(self, name, h)

      if "_RL_TrkPt_JetPt" in observable:
        name = 'h_{}'.format(observable)
        RL_bins = linbins(0,0.4,50)
        trkpt_bins = linbins(0,20,200)
        jetpt_bins = linbins(0,200,200)
        h = ROOT.TH3D(name, name, 50, RL_bins, 200, trkpt_bins, 200, jetpt_bins)
        h.GetXaxis().SetTitle('#Delta R')
        h.GetYaxis().SetTitle('p_{T,ch trk}')
        h.GetZaxis().SetTitle('p_{T jet}')
        setattr(self, name, h)

      if "_RL_z_JetPt" in observable:
        name = 'h_{}'.format(observable)
        RL_bins = linbins(0,0.4,50)
        z_bins = logbins(1.e-5, 1., 200)
        jetpt_bins = linbins(0,200,200)
        h = ROOT.TH3D(name, name, 50, RL_bins, 200, z_bins, 200, jetpt_bins)
        h.GetXaxis().SetTitle('#Delta R')
        h.GetYaxis().SetTitle('z')
        h.GetZaxis().SetTitle('p_{T jet}')
        setattr(self, name, h)

      if observable == 'jet_pt_JetPt':
        name = 'h_perpcone_{}'.format(observable)
        jetpt_bins = linbins(0,200,200)
        h = ROOT.TH1D(name, name, 200, jetpt_bins)
        h.GetXaxis().SetTitle('p_{T jet}')
        h.GetYaxis().SetTitle('Counts')
        setattr(self, name, h)

      if observable == "trk_pt_TrkPt":
        name = 'h_perpcone_{}'.format(observable)
        trkpt_bins = linbins(0,20,200)
        h = ROOT.TH1D(name, name, 200, trkpt_bins)
        h.GetXaxis().SetTitle('p_{T,ch trk}')
        h.GetYaxis().SetTitle('Counts')
        setattr(self, name, h)

      if "_RL_TrkPt_JetPt" in observable:
        name = 'h_perpcone_{}'.format(observable)
        RL_bins = linbins(0,0.4,50)
        trkpt_bins = linbins(0,20,200)
        jetpt_bins = linbins(0,200,200)
        h = ROOT.TH3D(name, name, 50, RL_bins, 200, trkpt_bins, 200, jetpt_bins)
        h.GetXaxis().SetTitle('#Delta R')
        h.GetYaxis().SetTitle('p_{T,ch trk}')
        h.GetZaxis().SetTitle('p_{T jet}')
        setattr(self, name, h)

      if "_RL_z_JetPt" in observable:
        name = 'h_perpcone_{}'.format(observable)
        RL_bins = linbins(0,0.4,50)
        z_bins = logbins(1.e-5, 1., 200)
        jetpt_bins = linbins(0,200,200)
        h = ROOT.TH3D(name, name, 50, RL_bins, 200, z_bins, 200, jetpt_bins)
        h.GetXaxis().SetTitle('#Delta R')
        h.GetYaxis().SetTitle('z')
        h.GetZaxis().SetTitle('p_{T jet}')
        setattr(self, name, h)

    
  def fill_jet_histograms(self, jet, jet_pt_corrected, c_select=None, is_perpcone=False):
    # fills histgrams and thats it, make sure constituents are properlly selected beforehand
    # uses jet_pt_corrected instead if jet.perp() for all filled jet pt histogram information
    # jet only used for jet axis direction! All constituents are from c_select
    # if c_select not specified, use all constituents in jet

    if c_select is None: c_select = jet.constituents()

    for c in c_select:
      
      # cut on 0.15 track pT again so ghosts not filled 
      if c.perp() < 0.15: continue

      rl = self.calculate_distance(jet, c)

      if is_perpcone:
        name = 'preprocessed_np_data_jettrk_bkgd'
      else:
        name = 'preprocessed_np_data_jettrk'
      
      getattr(self, name).append([rl, c.perp(), jet_pt_corrected, self.event_number])

      for observable in self.observable_list:
        if 'jet-trk' in observable:

          if is_perpcone:
            name = 'h_perpcone_{}'.format(observable)
          else:
            name = 'h_perpcone_{}'.format(observable)
          h = getattr(self, name)

          # fill other hists
          if observable == "jet-trk_shape_RL_TrkPt_JetPt":
            h.Fill(rl, c.perp(), jet_pt_corrected)
          elif observable == "jet-trk_ptprofile_RL_TrkPt_JetPt":
            h.Fill(rl, c.perp(), jet_pt_corrected, c.perp())
          elif observable == "jet-trk_shape_RL_z_JetPt":
            h.Fill(rl, c.perp()/jet_pt_corrected, jet_pt_corrected)
          elif observable == "jet-trk_ptprofile_RL_z_JetPt":
            h.Fill(rl, c.perp()/jet_pt_corrected, jet_pt_corrected, c.perp())
    

##################################################################
if __name__ == '__main__':
  # Define arguments
  parser = argparse.ArgumentParser(description='Process data')
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
  print('----------------------------------------------------------------')
  
  # If invalid inputFile is given, exit
  if not os.path.exists(args.inputFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.inputFile))
    sys.exit(0)
  
  # If invalid configFile is given, exit
  if not os.path.exists(args.configFile):
    print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
    sys.exit(0)

  analysis = ProcessData_JetTrk(input_file=args.inputFile, config_file=args.configFile, output_dir=args.outputDir)
  analysis.process_data()
