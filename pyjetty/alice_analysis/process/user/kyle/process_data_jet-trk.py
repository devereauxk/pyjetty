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
        
    for jetR in self.jetR_list:
      for observable in self.observable_list:
        for trk_thrd in self.obs_settings[observable]:
        
          obs_label = self.utils.obs_label(trk_thrd, None) 

          if observable == 'jet_pt_JetPt':
            name = 'h_{}_R{}_{}'.format(observable, jetR, obs_label)
            jetpt_bins = linbins(0,200,200)
            h = ROOT.TH1D(name, name, 200, jetpt_bins)
            h.GetXaxis().SetTitle('p_{T jet}')
            h.GetYaxis().SetTitle('Counts')
            setattr(self, name, h)

          if observable == "trk_pt_TrkPt":
            name = 'h_{}_R{}_{}'.format(observable, jetR, obs_label)
            trkpt_bins = linbins(0,20,200)
            h = ROOT.TH1D(name, name, 200, trkpt_bins)
            h.GetXaxis().SetTitle('p_{T,ch trk}')
            h.GetYaxis().SetTitle('Counts')
            setattr(self, name, h)

          if "_RL_TrkPt_JetPt" in observable:
            name = 'h_{}_R{}_{}'.format(observable, jetR, obs_label)
            RL_bins = linbins(0,jetR,50)
            trkpt_bins = linbins(0,20,200)
            jetpt_bins = linbins(0,200,200)
            h = ROOT.TH3D(name, name, 50, RL_bins, 200, trkpt_bins, 200, jetpt_bins)
            h.GetXaxis().SetTitle('#Delta R')
            h.GetYaxis().SetTitle('p_{T,ch trk}')
            h.GetZaxis().SetTitle('p_{T jet}')
            setattr(self, name, h)

          if "_RL_z_JetPt" in observable:
            name = 'h_{}_R{}_{}'.format(observable, jetR, obs_label)
            RL_bins = linbins(0,jetR,50)
            z_bins = logbins(1.e-5, 1., 200)
            jetpt_bins = linbins(0,200,200)
            h = ROOT.TH3D(name, name, 50, RL_bins, 200, z_bins, 200, jetpt_bins)
            h.GetXaxis().SetTitle('#Delta R')
            h.GetYaxis().SetTitle('z')
            h.GetZaxis().SetTitle('p_{T jet}')
            setattr(self, name, h)
          
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

  def is_same_charge(self, corr_builder, ipoint, constituents, index):
    part1 = corr_builder.correlator(ipoint).indices1()[index]
    part2 = corr_builder.correlator(ipoint).indices2()[index]
    q1 = constituents[part1].python_info().charge
    q2 = constituents[part2].python_info().charge

    if q1*q2 > 0:
      return True
    else:
      return False
    
    
  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  #---------------------------------------------------------------
  def fill_jet_histograms(self, jet, jet_groomed_lund, jetR, obs_setting, grooming_setting,
                          obs_label, jet_pt_ungroomed, suffix):

    constituents = fj.sorted_by_pt(jet.constituents())
    c_select = fj.vectorPJ()
    trk_thrd = obs_setting

    for c in constituents:
      if c.pt() < trk_thrd:
        break
      c_select.append(c) # NB: use the break statement since constituents are already sorted

    hname = 'h_{}_R{}_{}'
    for observable in self.observable_list:
        
      if observable == 'jet_pt_JetPt':
        getattr(self, hname.format(observable, jetR, obs_label)).Fill(jet.perp())

      if observable == "trk_pt_TrkPt":
        for c in c_select:
            getattr(self, hname.format(observable, jetR, obs_label)).Fill(c.perp())
            
      if 'jet-trk' in observable:
        h = getattr(self, hname.format(observable, jetR, obs_label))
        
        for c in c_select:
            rl = jet.delta_R(c)

            if observable == "jet-trk_shape_RL_TrkPt_JetPt":
              h.Fill(rl, c.perp(), jet.perp())
            elif observable == "jet-trk_ptprofile_RL_TrkPt_JetPt":
              h.Fill(rl, c.perp(), jet.perp(), c.perp())
            elif observable == "jet-trk_shape_RL_z_JetPt":
              h.Fill(rl, c.perp()/jet.perp(), jet.perp())
            elif observable == "jet-trk_ptprofile_RL_z_JetPt":
              h.Fill(rl, c.perp()/jet.perp(), jet.perp(), c.perp())
          

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
