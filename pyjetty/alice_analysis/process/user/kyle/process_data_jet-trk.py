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


  #---------------------------------------------------------------
  # Analyze jets of a given event.
  #---------------------------------------------------------------
  def analyze_jets(self, jets, jetR, fj_particles, rho = 0):

    for jet in jets:

      ############ APPLY LEADING PARTICLE and JET AREA CUTS ######
      # applied only to det/hybrid jets, not to truth jets

      if jet.is_pure_ghost(): continue

      leading_pt = np.max([c.perp() for c in jet.constituents()])

      # print("pt {} leading {} area {}".format(jet.perp(), leading_pt, jet.area()))

      # jet area and leading particle pt cut
      if jet.area() < 0.6*np.pi*jetR**2 or leading_pt < 5 or leading_pt > 100:
        continue

      jet_pt_corrected = jet.perp() - rho*jet.area()

      if jet_pt_corrected <= self.jet_pt_min:
        continue

      ################# JETS PASSED, FILL HISTS #################

      # for PbPb data
      if not self.is_pp:
        # handle jets that contain the embeded particle, fill hists, and skip analysis for this jet
        embeded_index = np.where(np.array([c.user_index() for c in jet.constituents()]) == -3)[0]
        if len(embeded_index) != 0:
          embeded_part = jet.constituents()[embeded_index[0]]
          delta_pt = jet_pt_corrected - embeded_part.perp() 
          getattr(self, "hDelta_pt").Fill(delta_pt)
          continue

        # fill background histograms
        self.analyze_perp_cones(jet, jet_pt_corrected, fj_particles)

      # print('jet that passes everything!')

      # fill signal histograms
      # must be placed after embedded 60 GeV particle check (PbPb only)!
      self.fill_jet_histograms(jet, jet_pt_corrected)

    
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


  #---------------------------------------------------------------
  # Analyze perpendicular cones to a given signal jet
  #---------------------------------------------------------------
  def analyze_perp_cones(self, jet, jets_pt_corrected, fj_particles):
    # do it for both + and - 90 deg bc wtf not, i guess the average will be a better estimate
      
    jetR_eff = np.sqrt(jet.area() / np.pi)

    perp_jet1 = fj.PseudoJet()
    perp_jet1.reset_PtYPhiM(jet.perp(), jet.rap(), jet.phi() + np.pi/2, jet.m())
    parts_in_perpcone1 = self.find_parts_around_jet(fj_particles, perp_jet1, jetR_eff) # jet R

    perp_jet2 = fj.PseudoJet()
    perp_jet2.reset_PtYPhiM(jet.perp(), jet.rap(), jet.phi() - np.pi/2, jet.m())
    parts_in_perpcone2 = self.find_parts_around_jet(fj_particles, perp_jet2, jetR_eff) # jet R

    self.fill_jet_histograms(perp_jet1, jets_pt_corrected, c_select=parts_in_perpcone1, is_perpcone=True)

    self.fill_jet_histograms(perp_jet2, jets_pt_corrected, c_select=parts_in_perpcone2, is_perpcone=True)


  def find_parts_around_jet(self, parts, jet, cone_R):
    # select particles around jet axis
    # perfect circle around jet axis
    cone_parts = fj.vectorPJ()
    for part in parts:
      if self.calculate_distance(jet, part):
        cone_parts.push_back(part)
    
    return cone_parts
    

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
