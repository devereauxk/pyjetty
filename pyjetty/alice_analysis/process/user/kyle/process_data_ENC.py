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
class ProcessData_ENC(process_data_base.ProcessDataBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
  
    # print(sys.path)
    # print(sys.modules)

    # Initialize base class
    super(ProcessData_ENC, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)
    
    self.observable = self.observable_list[0]


  #---------------------------------------------------------------
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_user_output_objects(self):
        
    # python array with the format (faster than np array!)
    # ['obs_energy_weight', 'obs_R_L', 'obs_jet_pt', 'event_n']
    if 'proprocessed' in self.observable_list:
      name = 'preprocessed_np_data'
      h = []
      setattr(self, name, h)

    for jetR in self.jetR_list:
      for observable in self.observable_list:
        for trk_thrd in self.obs_settings[observable]:
        
          obs_label = self.utils.obs_label(trk_thrd, None) 
          if self.is_pp:
              # Init ENC histograms
              if 'ENC' in observable:
                for ipoint in range(2, 3):
                    name = 'h_{}_JetPt_R{}_{}'.format(observable + str(ipoint), jetR, trk_thrd)
                    pt_bins = linbins(0,200,200)
                    RL_bins = logbins(1E-4,1,50)
                    h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                    h.GetXaxis().SetTitle('p_{T,ch jet}')
                    h.GetYaxis().SetTitle('R_{L}')
                    setattr(self, name, h)

                    name = 'h_{}Pt_JetPt_R{}_{}'.format(observable + str(ipoint), jetR, trk_thrd)
                    pt_bins = linbins(0,200,200)
                    ptRL_bins = logbins(1E-3,1E2,60)
                    h = ROOT.TH2D(name, name, 200, pt_bins, 60, ptRL_bins)
                    h.GetXaxis().SetTitle('p_{T,ch jet}')
                    h.GetYaxis().SetTitle('p_{T,ch jet}R_{L}') # NB: y axis scaled by jet pt (applied jet by jet)
                    setattr(self, name, h)

              if 'EEC_noweight' in observable:
                name = 'h_{}_JetPt_R{}_{}'.format(observable, jetR, obs_label)
                pt_bins = linbins(0,200,200)
                RL_bins = logbins(1E-4,1,50)
                h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                h.GetXaxis().SetTitle('p_{T,ch jet}')
                h.GetYaxis().SetTitle('R_{L}')
                setattr(self, name, h)

              if 'EEC_weight2' in observable: # NB: weight power = 2
                name = 'h_{}_JetPt_R{}_{}'.format(observable, jetR, obs_label)
                pt_bins = linbins(0,200,200)
                RL_bins = logbins(1E-4,1,50)
                h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                h.GetXaxis().SetTitle('p_{T,ch jet}')
                h.GetYaxis().SetTitle('R_{L}')
                setattr(self, name, h)

              if 'jet_pt' in observable:
                name = 'h_{}_JetPt_R{}_{}'.format(observable, jetR, obs_label)
                pt_bins = linbins(0,200,200)
                h = ROOT.TH1D(name, name, 200, pt_bins)
                h.GetXaxis().SetTitle('p_{T,ch jet}')
                h.GetYaxis().SetTitle('Counts')
                setattr(self, name, h)
            
              if observable == "trk_pt":
                name = 'h_{}_JetPt_R{}_{}'.format(observable, jetR, obs_label)
                pt_bins = linbins(0,100,200)
                h = ROOT.TH1D(name, name, 200, pt_bins)
                h.GetXaxis().SetTitle('p_{T,ch trk}')
                h.GetYaxis().SetTitle('Counts')
                setattr(self, name, h)
                
              if 'jet-trk_shape' in observable or 'jet-trk_ptprofile' in observable:
                name = 'h_{}_JetPt_R{}_{}'.format(observable, jetR, obs_label)
                RL_bins = linbins(0,jetR,50)
                pt_bins = linbins(0,100,200)
                z_bins = logbins(1.e-5, 1., 80)
                h = ROOT.TH3D(name, name, 50, RL_bins, 200, pt_bins, 80, z_bins)
                h.GetXaxis().SetTitle('#Delta R')
                h.GetYaxis().SetTitle('p_{T,ch trk}')
                h.GetZaxis().SetTitle('z')
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

    if self.ENC_pair_cut:
      dphi_cut = -9999 # means no dphi cut
      deta_cut = 0.008
    else:
      dphi_cut = -9999
      deta_cut = -9999

    hname = 'h_{}_JetPt_R{}_{}'
    new_corr = ecorrel.CorrelatorBuilder(c_select, jet.perp(), 2, 1, dphi_cut, deta_cut)
    for observable in self.observable_list:
      if 'ENC' in observable or 'EEC_noweight' in observable or 'EEC_weight2' in observable:
        for ipoint in range(2, 3):
          for index in range(new_corr.correlator(ipoint).rs().size()):

            # processing only like-sign pairs when self.ENC_pair_like is on
            if self.ENC_pair_like and (not self.is_same_charge(new_corr, ipoint, c_select, index)):
              continue

            # processing only unlike-sign pairs when self.ENC_pair_unlike is on
            if self.ENC_pair_unlike and self.is_same_charge(new_corr, ipoint, c_select, index):
              continue

            if 'ENC' in observable:
              getattr(self, hname.format(observable + str(ipoint), jetR, obs_label)).Fill(jet.perp(), new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index])
              getattr(self, hname.format(observable + str(ipoint) + 'Pt', jetR, obs_label)).Fill(jet.perp(), jet.perp()*new_corr.correlator(ipoint).rs()[index], new_corr.correlator(ipoint).weights()[index]) # NB: fill pt*RL

            if ipoint==2 and 'EEC_noweight' in observable:
              getattr(self, hname.format(observable, jetR, obs_label)).Fill(jet.perp(), new_corr.correlator(ipoint).rs()[index])

            if ipoint==2 and 'EEC_weight2' in observable:
              getattr(self, hname.format(observable, jetR, obs_label)).Fill(jet.perp(), new_corr.correlator(ipoint).rs()[index], pow(new_corr.correlator(ipoint).weights()[index],2))

      if 'jet_pt' in observable:
        getattr(self, hname.format(observable, jetR, obs_label)).Fill(jet.perp())

      if observable == "trk_pt":
        for c in c_select:
            getattr(self, hname.format(observable, jetR, obs_label)).Fill(c.perp())
            
      if 'jet-trk' in observable:
        h = getattr(self, hname.format(observable, jetR, obs_label))
        
        for c in c_select:
            rl = jet.delta_R(c)
            
            if 'shape' in observable:
                h.Fill(rl, c.perp(), c.perp()/jet.perp())
                
            elif 'ptprofile' in observable:
                h.Fill(rl, c.perp(), c.perp()/jet.perp(), c.perp())

      if 'preprocessed' in observable:
        self.fill_jet_tables(jet)

  def fill_jet_tables(self, jet, ipoint=2):
    constituents = fj.sorted_by_pt(jet.constituents())
    
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
    
    name = 'preprocessed_np_data'
    for i in range(len(EEC_rs)):
        new_row = [EEC_weights[i], EEC_rs[i], jet_pt, self.event_number]
        getattr(self, name).append(new_row)
          

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

  analysis = ProcessData_ENC(input_file=args.inputFile, config_file=args.configFile, output_dir=args.outputDir)
  analysis.process_data()
