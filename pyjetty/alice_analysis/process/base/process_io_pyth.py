#!/usr/bin/env python3

"""
  Analysis IO class for jet analysis with PYTHIA track dataframes at parton/hadron level.
  Each instance of the class handles the IO of a *single* track tree.
  
  Authors: James Mulligan
           Mateusz Ploskon
           Ezra Lesser
"""

from __future__ import print_function

import os   # for creating file on output
import sys

# Data analysis and plotting
import uproot
import pandas
import numpy as np

# Fastjet via python (from external library fjpydev)
import fastjet as fj
import fjext

# Base class
from pyjetty.alice_analysis.process.base import common_base

################################################################
class ProcessIO(common_base.CommonBase):
  
  #---------------------------------------------------------------
  # Constructor
  #---------------------------------------------------------------
  def __init__(self, input_file_MPIon='', input_file_MPIoff='', tree_name='t', betas=[], **kwargs):
    super(ProcessIO, self).__init__(**kwargs)
    self.input_file_MPIon = input_file_MPIon
    self.input_file_MPIoff = input_file_MPIoff
    self.tree_name = tree_name
    self.reset_dataframes()

    self.MPIoff_columns = ['p_pt', 'p_eta', 'p_phi']
    self.MPIon_columns = ['ch_pt', 'ch_eta', 'ch_phi']
    for beta in betas:
      self.MPIoff_columns.append('l_p_%s' % str(beta).replace('.',''))
      self.MPIon_columns.append('l_ch_%s' % str(beta).replace('.',''))
  
  #---------------------------------------------------------------
  # Clear dataframes
  #---------------------------------------------------------------
  def reset_dataframes(self):
    self.MPIon_df = None
    self.MPIoff_df = None
  
  #---------------------------------------------------------------
  # Convert ROOT TTree to SeriesGroupBy object of fastjet particles per event.
  # Optionally, remove a certain random fraction of tracks
  #---------------------------------------------------------------
  def load_data(self):

    self.reset_dataframes()

    print('Convert ROOT trees to pandas dataframes...')
    print('    tree_name = {}'.format(self.tree_name))

    self.ang_jets_df = self.load_dataframe()

    return self.ang_jets_df
  
  #---------------------------------------------------------------
  # Convert ROOT TTree to pandas dataframe
  # Return merged track+event dataframe from a given input file
  # Returned dataframe has one row per jet constituent:
  #     run_number, ev_id, ParticlePt, ParticleEta, ParticlePhi
  #---------------------------------------------------------------
  def load_dataframe(self):

    # Load MPI off tree into dataframe
    MPIoff_tree = uproot.open(self.input_file_MPIoff)[self.tree_name]
    if not MPIoff_tree:
      sys.exit('Tree {} not found in file {}'.format(self.tree_name, self.input_file_MPIoff))
    self.MPIoff_df = MPIoff_tree.pandas.df(self.MPIoff_columns)

    # Load MPI on tree into dataframe
    MPIon_tree = uproot.open(self.input_file_MPIon)[self.tree_name]
    if not MPIon_tree:
      sys.exit('Tree {} not found in file {}'.format(self.tree_name, self.input_file_MPIon))
    self.MPIon_df = MPIon_tree.pandas.df(self.MPIon_columns)

    # Merge event info into track tree
    self.track_df = self.MPIoff_df.join(self.MPIon_df)
    return self.track_df
