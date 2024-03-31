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
    
    if self.do_constituent_subtraction:
      self.is_pp = False
    else:
      self.is_pp = True

    if 'do_perpcone' in config:
      self.do_perpcone = config['do_perpcone']
    else:
      self.do_perpcone = False

    if 'do_rho_subtraction' in config:
      self.do_rho_subtraction = config['do_rho_subtraction']
    else:
      self.do_rho_subtraction = False

    if 'do_raw' in config:
      self.do_raw = config['do_raw']
    else:
      self.do_raw = False

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
    
    # Create constituent subtractor, if configured
    if not self.is_pp:
      self.constituent_subtractor = [CEventSubtractor(max_distance=R_max, alpha=self.alpha, max_eta=self.max_eta, bge_rho_grid_size=self.bge_rho_grid_size, max_pt_correct=self.max_pt_correct, ghost_area=self.ghost_area, distance_type=fjcontrib.ConstituentSubtractor.deltaR) for R_max in self.max_distance]
    
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

      cs = fj.ClusterSequence(fj_particles, jet_def, fj.AreaDefinition(fj.active_area_explicit_ghosts))
      jets_selected = fj.sorted_by_pt(jet_selector(cs.inclusive_jets()))

      # KD: EEC preprocessed output
      # self.analyze_matched_pairs(det_jets, truth_jets)

      # KD: jet-trk preprocessed output
      self.analyze_jets(jets, jetR)

    else:

      if not self.do_rho_subtraction:
        return

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

      cs = fj.ClusterSequenceArea(fj_particles, jet_def, fj.AreaDefinition(fj.active_area_explicit_ghosts)) # choice to use constituent subtraction here
      jets = fj.sorted_by_pt(cs.inclusive_jets())
      jets_selected = jet_selector(jets)

      for jet in jets_selected:

        leading_pt = np.max([c.perp() for c in jet.constituents()])

        # jet area and leading particle pt cut
        if jet.area() < 0.6*np.pi*jetR**2 or leading_pt < 5 or leading_pt > 100:
          continue

        jet_pt_corrected = jet.perp() - rho*jet.area()
        jetR_eff = np.sqrt(jet.area() / np.pi)

        if jet_pt_corrected <= 10:
          continue

        # handle jets that contain the embeded particle, fill hists, and skip analysis for this jet
        embeded_index = np.where(np.array([c.user_index() for c in jet.constituents()]) == -3)[0]
        if len(embeded_index) != 0:
          embeded_part = jet.constituents()[embeded_index[0]]
          delta_pt = jet_pt_corrected - embeded_part.perp()
          getattr(self, "hDelta_pt").Fill(delta_pt)
          continue

        # print("{} \t->\t {}, {}, R={}".format(jet.perp(), jet_pt_corrected, jet.area(), np.sqrt(jet.area() / np.pi)))

        # Call user function to fill histograms
        self.fill_jet_histograms(jet, jetR, jet_pt_corrected, self.obs_setting, self.obs_label)

        #perp_jet1 = fj.PseudoJet()
        #perp_jet1.reset_PtYPhiM(jet.perp(), jet.rap(), jet.phi(), jet.m())
        # make sure you use rapidity and NOT eta!
        #print("{}, {}, {}, {}, {}".format(jet.perp(), jet.eta(), jet.phi(), jet.m()))
        #print("{}, {}, {}, {}".format(perp_jet1.perp(), perp_jet1.eta(), perp_jet1.phi(), perp_jet1.m()))

        """
        regrouped_parts = self.find_parts_around_jet(fj_particles, jet, jetR_eff) # jet R
        hname = 'h_{}_R{}_' + str(self.obs_label)

        c_select = fj.vectorPJ()
        for c in regrouped_parts:
          if c.pt() < 0.15:
            break
          c_select.append(c) # NB: use the break statement since constituents are already sorted

        self.fill_jettrk_histograms(hname, c_select, jet, jet_pt_corrected, jetR)
        """

        perp_jet1 = fj.PseudoJet()
        perp_jet1.reset_PtYPhiM(jet.perp(), jet.rap(), jet.phi() + np.pi/2, jet.m())
        parts_in_perpcone1 = self.find_parts_around_jet(fj_particles, perp_jet1, jetR_eff) # jet R

        perp_jet2 = fj.PseudoJet()
        perp_jet2.reset_PtYPhiM(jet.perp(), jet.rap(), jet.phi() - np.pi/2, jet.m())
        parts_in_perpcone2 = self.find_parts_around_jet(fj_particles, perp_jet2, jetR_eff) # jet R

        self.fill_perp_cone_histograms(parts_in_perpcone1, jetR, perp_jet1, jet_pt_corrected, \
                                self.obs_setting, self.obs_label)
        
        self.fill_perp_cone_histograms(parts_in_perpcone2, jetR, perp_jet2, jet_pt_corrected, \
                                self.obs_setting, self.obs_label)

      # TODO do cs just for jet finiding and then recluster?
      # TODO idk what R_max is, used in cs?
        
      # TODO
      # from ALICE jet spectra in PbPb 5.02 ....
      # jet area cut, usually cut on area of jets that are A > 0.6 pi R*2
      # leading particle cut, only jets with leading pt >5 GeV, no jets with leading pt above 100 GeV
      # if weird, use the lower cuts that this paper uses
      # no c_factor
      # for jet eta cut, do like 5% of jet radius more so there is a little more padding
      # embed 60 GeV part somewhere in acceptance area - rho * area - part_pt

      # implement the imbeded 60 GeV part to test its delta pT with the rho subtraction

      # plot jet area hist


  def find_parts_around_jet(self, parts, jet, cone_R):
    # select particles around jet axis
    cone_parts = fj.vectorPJ()
    for part in parts:
      if self.calculate_distance(jet, part):
        cone_parts.push_back(part)
    
    return cone_parts

  def analyze_perp_cone(self, parts, jet, jetR, jet_pt_corrected):
    # analyze cones perpendicular to jet in the azimuthal plane
    # assumes inputted jets pass acceptance, pass reselection

    # TODO
    perpcone_R_list = [jetR] # FIX ME: not sure it's better to just use jet R or calculate from area, For now just use jet R so it's more consistent with the jet cone method
    # if self.do_rho_subtraction and rho_bge > 0:
    #   perpcone_R_list = [math.sqrt(jet.area()/np.pi)] # NB: jet area is available only when rho subtraction flag is on

    # purposfully use UNCORRECTED jet pt here since should have exact same jet def as original, just rotated
    perp_jet1 = fj.PseudoJet()
    perp_jet1.reset_PtYPhiM(jet.perp(), jet.rap(), jet.phi(), jet.m())
    perp_jet2 = fj.PseudoJet()
    perp_jet2.reset_PtYPhiM(jet.perp(), jet.rap(), jet.phi() - np.pi/2, jet.m())

    for perpcone_R in perpcone_R_list:

      # for each +/- 90 deg phi perpcone, find the parts from the event in that cone
      parts_in_perpcone1 = self.find_parts_around_jet(parts, perp_jet1, perpcone_R)
      
      parts_in_perpcone2 = self.find_parts_around_jet(parts, perp_jet2, perpcone_R)

      self.fill_perp_cone_histograms(parts_in_perpcone1, perpcone_R, perp_jet1, jet_pt_corrected, \
                                  self.obs_setting, self.obs_label)

      #self.fill_perp_cone_histograms(parts_in_perpcone2, perpcone_R, perp_jet2, jet_pt_corrected, \
      #                            self.obs_setting, self.obs_label)

  #---------------------------------------------------------------
  # This function is called once
  # You must implement this
  #---------------------------------------------------------------
  def initialize_user_output_objects(self):
  
    raise NotImplementedError('You must implement initialize_user_output_objects()!')

  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  # You must implement this
  #---------------------------------------------------------------
  def fill_jet_histograms(self, jet, jetR, jet_pt_corrected, obs_setting, obs_label):
  
    raise NotImplementedError('You must implement fill_jet_histograms()!')

  #---------------------------------------------------------------
  # This function is called once for each jet subconfiguration
  # You must implement this
  #---------------------------------------------------------------
  def fill_perp_cone_histograms(self, cone_parts, cone_R, jet, jet_pt_corrected, \
                                obs_setting, obs_label):
  
    raise NotImplementedError('You must implement fill_perp_cone_histograms()!')