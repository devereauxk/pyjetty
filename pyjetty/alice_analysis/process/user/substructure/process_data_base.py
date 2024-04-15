#!/usr/bin/env python3

"""
Base class to read a ROOT TTree of track information
and do jet-finding, and save basic histograms.
  
To use this class, the following should be done:

  - Implement a user analysis class inheriting from this one, such as in user/james/process_data_XX.py
    You should implement the following functions:
      - initialize_user_output_objects()
      - fill_jet_histograms()
    
  - The histogram of the data should be named h_[obs]_JetPt_R[R]_[subobs]_[grooming setting]
    The grooming part is optional, and should be labeled e.g. zcut01_B0 — from CommonUtils::grooming_label({'sd':[zcut, beta]})
    For example: h_subjet_z_JetPt_R0.4_0.1
    For example: h_subjet_z_JetPt_R0.4_0.1_zcut01_B0

  - You also should modify observable-specific functions at the top of common_utils.py
  
Author: James Mulligan (james.mulligan@berkeley.edu)
"""

from __future__ import print_function

# General
import time

# Data analysis and plotting
import numpy as np
import ROOT
import yaml
import pandas as pd
import math

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib

# Analysis utilities
from pyjetty.alice_analysis.process.base import process_io
from pyjetty.alice_analysis.process.base import process_base
from pyjetty.mputils import CEventSubtractor

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)

################################################################
class ProcessDataBase(process_base.ProcessBase):

  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, **kwargs):
  
    # Initialize base class
    super(ProcessDataBase, self).__init__(input_file, config_file, output_dir, debug_level, **kwargs)

    # Initialize configuration
    self.initialize_config()
    
  #---------------------------------------------------------------
  # Initialize config file into class members
  #---------------------------------------------------------------
  def initialize_config(self):
    
    # Call base class initialization
    process_base.ProcessBase.initialize_config(self)
    
    # Read config file
    with open(self.config_file, 'r') as stream:
      config = yaml.safe_load(stream)

    if 'use_ev_id' in config:
      self.use_ev_id = config['use_ev_id']
    else:
      self.use_ev_id = True

    self.is_pp = config['is_pp']

    if 'do_perpcone' in config:
      self.do_perpcone = config['do_perpcone']
    else:
      self.do_perpcone = False

    if 'do_rho_subtraction' in config:
      self.do_rho_subtraction = config['do_rho_subtraction']
    else:
      self.do_rho_subtraction = False

    # Create dictionaries to store grooming settings and observable settings for each observable
    # Each dictionary entry stores a list of subconfiguration parameters
    #   The observable list stores the observable setting, e.g. subjetR
    #   The grooming list stores a list of grooming settings {'sd': [zcut, beta]} or {'dg': [a]}
    self.observable_list = config['process_observables']
    self.obs_settings = {}
    self.obs_grooming_settings = {}
    for observable in self.observable_list:
    
      obs_config_dict = config[observable]
      obs_config_list = [name for name in list(obs_config_dict.keys()) if 'config' in name ]
      
      obs_subconfig_list = [name for name in list(obs_config_dict.keys()) if 'config' in name ]
      self.obs_settings[observable] = self.utils.obs_settings(observable, obs_config_dict, obs_subconfig_list)
      self.obs_grooming_settings[observable] = self.utils.grooming_settings(obs_config_dict)
      
    # Construct set of unique grooming settings
    self.grooming_settings = []
    lists_grooming = [self.obs_grooming_settings[obs] for obs in self.observable_list]
    for observable in lists_grooming:
      for setting in observable:
        if setting not in self.grooming_settings and setting != None:
          self.grooming_settings.append(setting)

    # KD: assume one observable setting/label
    observable = self.observable_list[0]
    self.obs_setting = 0.15 # hard coded
    grooming_setting = self.obs_grooming_settings[observable][0]
    self.obs_label = self.utils.obs_label(self.obs_setting, grooming_setting)
          
  #---------------------------------------------------------------
  # Main processing function
  #---------------------------------------------------------------
  def process_data(self):
    
    self.start_time = time.time()

    # Use IO helper class to convert ROOT TTree into a SeriesGroupBy object of fastjet particles per event
    print('--- {} seconds ---'.format(time.time() - self.start_time))
    io = process_io.ProcessIO(input_file=self.input_file, track_tree_name='tree_Particle',
                              is_pp=self.is_pp, use_ev_id_ext=self.use_ev_id)
    self.df_fjparticles = io.load_data(m=self.m)
    self.nEvents = len(self.df_fjparticles.index)
    self.nTracks = len(io.track_df.index)
    print('--- {} seconds ---'.format(time.time() - self.start_time))
    
    # Initialize histograms
    self.initialize_output_objects()
    
    print(self)

    # Find jets and fill histograms
    print('Analyze events...')
    self.analyze_events()
    
    # Plot histograms
    print('Save histograms...')
    process_base.ProcessBase.save_output_objects(self)

    print('--- {} seconds ---'.format(time.time() - self.start_time))
  
  #---------------------------------------------------------------
  # Initialize histograms
  #---------------------------------------------------------------
  def initialize_output_objects(self):
  
    # Initialize user-specific histograms
    self.initialize_user_output_objects()
    
    # Initialize base histograms
    self.hNevents = ROOT.TH1F('hNevents', 'hNevents', 2, -0.5, 1.5)
    if self.event_number_max < self.nEvents:
      self.hNevents.Fill(1, self.event_number_max)
    else:
      self.hNevents.Fill(1, self.nEvents)
    
    self.hTrackEtaPhi = ROOT.TH2F('hTrackEtaPhi', 'hTrackEtaPhi', 200, -1., 1., 628, 0., 6.28)
    self.hTrackPt = ROOT.TH1F('hTrackPt', 'hTrackPt', 300, 0., 300.)
    
    if not self.is_pp:
      self.hRho = ROOT.TH1F('hRho', 'hRho', 100, 0., 300.)
      self.hSigma = ROOT.TH1F('hSigma', 'hSigma', 100, 0., 100.)
      self.hDelta_pt = ROOT.TH1F('hDelta_pt', 'hDelta_pt', 200, -60, 120)

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
  # Main function to loop through and analyze events
  #---------------------------------------------------------------
  def analyze_events(self):
    
    # Fill track histograms
    print('--- {} seconds ---'.format(time.time() - self.start_time))
    print('Fill track histograms')
    [[self.fillTrackHistograms(track) for track in fj_particles] for fj_particles in self.df_fjparticles]
    print('--- {} seconds ---'.format(time.time() - self.start_time))
    
    print('Find jets...')
    fj.ClusterSequence.print_banner()
    print()
    self.event_number = 0
  
    # Use list comprehension to do jet-finding and fill histograms
    _ = [self.analyze_event(fj_particles) for fj_particles in self.df_fjparticles]
    
    print('--- {} seconds ---'.format(time.time() - self.start_time))
    print('Save thn...')
    process_base.ProcessBase.save_thn_th3_objects(self)
  
  #---------------------------------------------------------------
  # Fill track histograms.
  #---------------------------------------------------------------
  def fillTrackHistograms(self, track):
    
    self.hTrackEtaPhi.Fill(track.eta(), track.phi())
    self.hTrackPt.Fill(track.pt())
  
  #---------------------------------------------------------------
  # Analyze jets of a given event.
  # fj_particles is the list of fastjet pseudojets for a single fixed event.
  #---------------------------------------------------------------
  def analyze_event(self, fj_particles):
    
    self.event_number += 1
    if self.event_number > self.event_number_max:
      return
    if self.debug_level > 1:
      print('-------------------------------------------------')
      print('event {}'.format(self.event_number))

    if self.event_number % 1000 == 0: print("analyzing event : " + str(self.event_number))
    
    # various checks
    if len(fj_particles) > 1:
      if np.abs(fj_particles[0].pt() - fj_particles[1].pt()) <  1e-10:
        print('WARNING: Duplicate particles may be present')
        print([p.user_index() for p in fj_particles])
        print([p.pt() for p in fj_particles])

    #handle case with no truth particles
    if type(fj_particles) is float:
        print("EVENT WITH NO PARTICLES!!!")
        return
    
    ##### PARTICLE PT CUT > 0.15
    fj_particles_pass = fj.vectorPJ()
    for part in fj_particles:
      if part.perp() > 0.15:
        fj_particles_pass.push_back(part)
    fj_particles = fj_particles_pass
  

    ############################# JET RECO ################################
    jetR = 0.4   

    # Set jet definition and a jet selector
    jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
    jet_selector = fj.SelectorPtMin(10.0) & fj.SelectorAbsRapMax(0.9 - 1.05*jetR)

    if self.debug_level > 2:
        print('jet definition is:', jet_def)
        print('jet selector is:', jet_selector,'\n')

    # Analyze
    if self.is_pp:

      cs = fj.ClusterSequenceArea(fj_particles, jet_def, fj.AreaDefinition(fj.active_area_explicit_ghosts))
      jets_selected = fj.sorted_by_pt(jet_selector(cs.inclusive_jets()))

      # KD: EEC preprocessed output
      self.analyze_pairs(jets_selected)
      return

      # KD: jet-trk preprocessed output
      # self.analyze_jets(jets_selected, jetR, fj_particles)

    else:
      
      # performs rho subtraction automatically

      # embeding random 60GeV particle with unique user_index=-3
      random_Y = np.random.uniform(-0.9, 0.9)
      random_phi = np.random.uniform(0, 2*np.pi)
      probe = fj.PseudoJet()
      probe.reset_PtYPhiM(60, random_Y, random_phi, 1)
      probe.set_user_index(-3)
      fj_particles.push_back(probe)

      # rho calculation
      bge = fj.GridMedianBackgroundEstimator(0.9, 0.4) # max eta, grid size
      bge.set_particles(fj_particles)
      rho = bge.rho()
      sigma = bge.sigma()

      # print(" RHO THIS EVENT : {}, +/- {}".format(rho, sigma)) 

      getattr(self, "hRho").Fill(rho)
      getattr(self, "hSigma").Fill(sigma)

      cs = fj.ClusterSequenceArea(fj_particles, jet_def, fj.AreaDefinition(fj.active_area_explicit_ghosts))
      jets_selected = fj.sorted_by_pt(jet_selector(cs.inclusive_jets()))

      self.analyze_jets(jets_selected, jetR, fj_particles, rho=rho)


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

      if jet_pt_corrected <= 10:
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


  #---------------------------------------------------------------
  # Analyze perpendicular cones to a given signal jet
  #---------------------------------------------------------------
  def analyze_perp_cones(self, jet, jets_pt_corrected, fj_particles):
      
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
    cone_parts = fj.vectorPJ()
    for part in parts:
      if self.calculate_distance(jet, part):
        cone_parts.push_back(part)
    
    return cone_parts

  #---------------------------------------------------------------
  # This function is called once
  # You must implement this
  #---------------------------------------------------------------
  def initialize_user_output_objects(self):
  
    raise NotImplementedError('You must implement initialize_user_output_objects()!')
