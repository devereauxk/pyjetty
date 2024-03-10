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
      self.c_factor = ROOT.TH1F('c_factor', 'c_factor', 100, 0., 1.)

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
    
    if len(fj_particles) > 1:
      if np.abs(fj_particles[0].pt() - fj_particles[1].pt()) <  1e-10:
        print('WARNING: Duplicate particles may be present')
        print([p.user_index() for p in fj_particles])
        print([p.pt() for p in fj_particles])
  
    # Perform constituent subtraction for each R_max (do this once, for all jetR)
    if not self.is_pp:
      fj_particles_subtracted = [self.constituent_subtractor[i].process_event(fj_particles) for i, R_max in enumerate(self.max_distance)]
    
      #print("len fj parts: {}, len subed: {}".format(len(fj_particles), len(fj_particles_subtracted[0])))

    # Loop through jetR, and process event for each R
    for jetR in self.jetR_list:
    
      # Keep track of whether to fill R-independent histograms
      self.fill_R_indep_hists = (jetR == self.jetR_list[0])

      # Set jet definition and a jet selector
      jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
      jet_selector = fj.SelectorPtMin(10.0) & fj.SelectorAbsRapMax(0.9 - jetR)
      if self.debug_level > 2:
        print('jet definition is:', jet_def)
        print('jet selector is:', jet_selector,'\n')
        
      # Analyze
      if self.is_pp:
      
        # Do jet finding
        cs = fj.ClusterSequence(fj_particles, jet_def)
        jets = fj.sorted_by_pt(cs.inclusive_jets())
        jets_selected = jet_selector(jets)
      
        self.analyze_jets(jets_selected, jetR)
        
      else:

        if not self.do_rho_subtraction:
          return
        
        # jet finding using KT, just for rho calculation
        jet_def_rho = fj.JetDefinition(fj.kt_algorithm, jetR)
        jet_selector_rho = fj.SelectorPtMin(10.0) & fj.SelectorAbsRapMax(0.9 - jetR)

        cs_rho = fj.ClusterSequenceArea(fj_particles, jet_def_rho, fj.AreaDefinition(fj.active_area_explicit_ghosts))
        jets_rho = fj.sorted_by_pt(cs_rho.inclusive_jets())
        jets_selected_rho = jet_selector_rho(jets_rho)

        bge = fj.GridMedianBackgroundEstimator(0.9, 0.4) # max eta, grid size
        bge.set_particles(fj_particles)
        rho = bge.rho()
        sigma = bge.sigma()

        # print(" RHO THIS EVENT : {}, +/- {}".format(rho, sigma)) 

        # calculate occupancy factor: summed jet areas with no ghosts / total detector area
        c_num = 0
        for jet in jets_selected_rho:
          if not jet.is_pure_ghost():
            c_num += jet.area()

        A_acc = 0.9 * 2 * np.pi * 2
        c_factor = c_num / A_acc
        # print("c factor = {} / {} = {} ".format(c_num, A_acc, c_facter))

        getattr(self, 'hRho').Fill(rho)
        getattr(self, "hSigma").Fill(sigma)
        getattr(self, "c_factor").Fill(c_factor)

        pts = []
        areas = []
        corrected_pts = []
        # corrected_pts should have tje same order as jets_selected_rho, don't reorder either array
        for jet in jets_selected_rho:
          # (jet pT, jet area, jet pT sub)

          if jet.is_pure_ghost():
            continue

          pts.append(jet.perp())
          areas.append(jet.area())

          # KD: perform rho subtraction, stores subtracted jet_pt as jet_pt_corrected
          # jet fj object still has original pT if jet.perp() called
          if rho > 0:
            jet_pt_corrected = jet.perp() - rho*jet.area()*c_factor

          corrected_pts.append(jet_pt_corrected)

        pts = np.array(pts)
        areas = np.array(areas)
        pts_areas = pts / areas

        rho_by_hand = np.median(pts_areas)
        # print("rho calculated by hand : {}".format(rho_by_hand))


        # jet finding using ANITKT, for main jet finding and analysis
        jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
        jet_selector = fj.SelectorPtMin(10.0) & fj.SelectorAbsRapMax(0.9 - jetR)

        cs = fj.ClusterSequence(fj_particles, jet_def) # choice to use constituent subtraction here
        jets = fj.sorted_by_pt(cs.inclusive_jets())
        jets_selected = jet_selector(jets)

        for jet in jets_selected:
          # find corresponding kt jet, to find the corresponding corrected jet pt from the rho subtraction
          delta_Rs = [jet.delta_R(candidate) for candidate in jets_selected_rho]
          closest_index = np.argmin(delta_Rs)
          #print("{}, accepted? {}".format(delta_Rs[closest_index], delta_Rs[closest_index] < 0.6))
          if delta_Rs[closest_index] > 0.1: #TODO 0.6 is what i found for mc processing?
            continue

          jet_pt_corrected = corrected_pts[closest_index]

          if jet_pt_corrected <= 10:
            continue

          #print("{} \t->\t {}".format(jet.perp(), jet_pt_corrected))

          # Call user function to fill histograms
          self.fill_jet_histograms(jet, jetR, jet_pt_corrected, self.obs_setting, self.obs_label)
          
          self.analyze_perp_cone(fj_particles, jet, jetR, jet_pt_corrected)

        # TODO do cs just for jet finiding and then recluster?
        # TODO idk what R_max is, used in cs?

        """

        if self.do_rho_subtraction:
          cs_unsub = fj.ClusterSequenceArea(fj_particles, jet_def, fj.AreaDefinition(fj.active_area_explicit_ghosts))
          jets_unsub = fj.sorted_by_pt(cs_unsub.inclusive_jets())
          jets_selected_unsub = jet_selector(jets_unsub)

          self.analyze_jets(jets_selected_unsub, jetR, rho_bge = rho) # changing rho to zero here changes trk pt spectrum drastically
          if self.do_perpcone:
            self.analyze_perp_cones(fj_particles, jets_selected_unsub, jetR, rho_bge = rho)

        elif self.do_raw:
          cs = fj.ClusterSequence(fj_particles, jet_def)
          jets = fj.sorted_by_pt(cs.inclusive_jets())
          jets_selected = jet_selector(jets)
        
          self.analyze_jets(jets_selected, jetR)
          if self.do_perpcone:
            self.analyze_perp_cones(fj_particles, jets_selected, jetR)

        else:
          # Do jet finding (re-do each time, to make sure matching info gets reset)
          cs = fj.ClusterSequence(fj_particles_subtracted[i], jet_def) # TODO : not sure whether to enable the area or not
          jets = fj.sorted_by_pt(cs.inclusive_jets())
          jets_selected = jet_selector(jets)

          self.analyze_jets(jets_selected, jetR)
          if self.do_perpcone:
            self.analyze_perp_cones(fj_particles, jets_selected, jetR)

        """

  def find_parts_around_jet(self, parts, jet, cone_R):
    # select particles around jet axis
    cone_parts = fj.vectorPJ()
    for part in parts:
      if jet.delta_R(part) <= cone_R:
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
    perp_jet1.reset_PtYPhiM(jet.perp(), jet.eta(), jet.phi(), jet.m())
    perp_jet2 = fj.PseudoJet()
    perp_jet2.reset_PtYPhiM(jet.perp(), jet.eta(), jet.phi() - np.pi/2, jet.m())

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