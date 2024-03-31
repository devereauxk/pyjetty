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

          if self.do_perpcone:

            if observable == 'jet_pt_JetPt':
              name = 'h_perpcone_{}_R{}_{}'.format(observable, jetR, obs_label)
              jetpt_bins = linbins(0,200,200)
              h = ROOT.TH1D(name, name, 200, jetpt_bins)
              h.GetXaxis().SetTitle('p_{T jet}')
              h.GetYaxis().SetTitle('Counts')
              setattr(self, name, h)

            if observable == "trk_pt_TrkPt":
              name = 'h_perpcone_{}_R{}_{}'.format(observable, jetR, obs_label)
              trkpt_bins = linbins(0,20,200)
              h = ROOT.TH1D(name, name, 200, trkpt_bins)
              h.GetXaxis().SetTitle('p_{T,ch trk}')
              h.GetYaxis().SetTitle('Counts')
              setattr(self, name, h)

            if "_RL_TrkPt_JetPt" in observable:
              name = 'h_perpcone_{}_R{}_{}'.format(observable, jetR, obs_label)
              RL_bins = linbins(0,jetR,50)
              trkpt_bins = linbins(0,20,200)
              jetpt_bins = linbins(0,200,200)
              h = ROOT.TH3D(name, name, 50, RL_bins, 200, trkpt_bins, 200, jetpt_bins)
              h.GetXaxis().SetTitle('#Delta R')
              h.GetYaxis().SetTitle('p_{T,ch trk}')
              h.GetZaxis().SetTitle('p_{T jet}')
              setattr(self, name, h)

            if "_RL_z_JetPt" in observable:
              name = 'h_perpcone_{}_R{}_{}'.format(observable, jetR, obs_label)
              RL_bins = linbins(0,jetR,50)
              z_bins = logbins(1.e-5, 1., 200)
              jetpt_bins = linbins(0,200,200)
              h = ROOT.TH3D(name, name, 50, RL_bins, 200, z_bins, 200, jetpt_bins)
              h.GetXaxis().SetTitle('#Delta R')
              h.GetYaxis().SetTitle('z')
              h.GetZaxis().SetTitle('p_{T jet}')
              setattr(self, name, h)
    
  def fill_jettrk_histograms(self, hname, c_select, jet, jet_pt_corrected, jetR):
    # fills histgrams and thats it, make sure constituents are properlly selected beforehand
    # uses jet_pt_corrected instead if jet.perp() for all filled jet pt histogram information
    
    for observable in self.observable_list:
      
      for c in c_select:
          rl = self.calculate_distance(jet, c)

          name = 'preprocessed_np_data_jettrk'
          getattr(self, name).append([rl, c.perp(), jet_pt_corrected, self.event_number])

          if 'jet-trk' in observable:
            h = getattr(self, hname.format(observable, jetR))

            # fill other hists
            if observable == "jet-trk_shape_RL_TrkPt_JetPt":
              h.Fill(rl, c.perp(), jet_pt_corrected)
            elif observable == "jet-trk_ptprofile_RL_TrkPt_JetPt":
              h.Fill(rl, c.perp(), jet_pt_corrected, c.perp())
            elif observable == "jet-trk_shape_RL_z_JetPt":
              h.Fill(rl, c.perp()/jet_pt_corrected, jet_pt_corrected)
            elif observable == "jet-trk_ptprofile_RL_z_JetPt":
              h.Fill(rl, c.perp()/jet_pt_corrected, jet_pt_corrected, c.perp())

  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  #---------------------------------------------------------------
  def fill_jet_histograms(self, jet, jetR, jet_pt_corrected, obs_setting, obs_label):

    hname = 'h_{}_R{}_' + str(obs_label)

    constituents = fj.sorted_by_pt(jet.constituents())
    c_select = fj.vectorPJ()
    trk_thrd = obs_setting

    for c in constituents:
      if c.pt() < trk_thrd:
        break
      c_select.append(c) # NB: use the break statement since constituents are already sorted

    for observable in self.observable_list:
        
      if observable == 'jet_pt_JetPt':
        getattr(self, hname.format(observable, jetR, obs_label)).Fill(jet_pt_corrected)
        # print("{} \t->\t {}".format(jet.perp(), jet_pt))

      if observable == "trk_pt_TrkPt":
        for c in c_select:
            getattr(self, hname.format(observable, jetR, obs_label)).Fill(c.perp())

    self.fill_jettrk_histograms(hname, c_select, jet, jet_pt_corrected, jetR)
          
  #---------------------------------------------------------------
  # This function is called twice for each jet subconfiguration
  # once for each of the two perp cones generated for a single sig cone
  #---------------------------------------------------------------
  def fill_perp_cone_histograms(self, cone_parts, jetR, jet, jet_pt_corrected, obs_setting, obs_label):
    # assumes every part in cone_parts is from background (in some cases sig jet is included)

    trk_thrd = obs_setting
    c_select_perp = fj.vectorPJ()

    cone_parts_sorted = fj.sorted_by_pt(cone_parts)
    for part in cone_parts_sorted:
      if part.pt() < trk_thrd:
        break
      c_select_perp.append(part)

    hname = 'h_perpcone_{}_R{}_' + str(obs_label)

    self.fill_jettrk_histograms(hname, c_select_perp, jet, jet_pt_corrected, jetR) # use signal jet pt here
    

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
