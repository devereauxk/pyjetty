#!/usr/bin/env python

from __future__ import print_function

import fastjet as fj
import fjcontrib
import fjext

import ROOT

import tqdm
import yaml
import copy
import argparse
import os
import array
import numpy as np

from pyjetty.mputils import *

from heppy.pythiautils import configuration as pyconf
import pythia8
import pythiafjext
import pythiaext
import ecorrel

from pyjetty.alice_analysis.process.base import process_base

# Prevent ROOT from stealing focus when plotting
ROOT.gROOT.SetBatch(True)
# Automatically set Sumw2 when creating new histograms
ROOT.TH1.SetDefaultSumw2()
ROOT.TH2.SetDefaultSumw2()

def linbins(xmin, xmax, nbins):
  lspace = np.linspace(xmin, xmax, nbins+1)
  arr = array.array('f', lspace)
  return arr

def logbins(xmin, xmax, nbins):
  lspace = np.logspace(np.log10(xmin), np.log10(xmax), nbins+1)
  arr = array.array('f', lspace)
  return arr

################################################################
class PythiaGenENC(process_base.ProcessBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, input_file='', config_file='', output_dir='', debug_level=0, args=None, **kwargs):

        super(PythiaGenENC, self).__init__(
            input_file, config_file, output_dir, debug_level, **kwargs)

        # Call base class initialization
        process_base.ProcessBase.initialize_config(self)

        # Read config file
        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.jet_levels = config["jet_levels"] # levels = ["p", "h", "ch"]

        self.jetR_list = config["jetR"] 

        self.nev = args.nev

        # particle level - ALICE tracking restriction
        self.max_eta_hadron = 0.9

        if 'beyond_jetR' in config:
            self.beyond_jetR = config['beyond_jetR']
        else:
            self.beyond_jetR = False

        self.ref_jet_level = "ch"
        self.ref_jetR = 0.4 # hard coded for now 
        self.part_levels = config["part_levels"] 

        # ENC settings
        self.dphi_cut = -9999
        self.deta_cut = -9999
        self.npoint = 2
        self.npower = 1

        if 'do_matching' in config:
            self.do_matching = config['do_matching']
        else:
            self.do_matching = False

        self.jet_matching_distance = config["jet_matching_distance"] 

        if 'do_tagging' in config:
            self.do_tagging = config['do_tagging']
        else:
            self.do_tagging = False

    #---------------------------------------------------------------
    # Main processing function
    #---------------------------------------------------------------
    def pythia_parton_hadron(self, args):
 
        # Create ROOT TTree file for storing raw PYTHIA particle information
        outf_path = os.path.join(self.output_dir, args.tree_output_fname)
        outf = ROOT.TFile(outf_path, 'recreate')
        outf.cd()

        mycfg = []
        mycfg.append("HadronLevel:all=off")
        pythia = pyconf.create_and_init_pythia_from_args(args, mycfg)

        # Initialize response histograms
        self.initialize_hist()

        # print the banner first
        fj.ClusterSequence.print_banner()
        print()

        self.init_jet_tools()
        self.calculate_events(pythia)
        pythia.stat()
        print()
        
        self.scale_print_final_info(pythia)

        outf.Write()
        outf.Close()

        self.save_output_objects()

    #---------------------------------------------------------------
    # Initialize histograms
    #---------------------------------------------------------------
    def initialize_hist(self):

        self.hNevents = ROOT.TH1I("hNevents", 'Number accepted events (unscaled)', 2, -0.5, 1.5)

        for jetR in self.jetR_list:

            # Store a list of all the histograms just so that we can rescale them later
            hist_list_name = "hist_list_R%s" % str(jetR).replace('.', '')
            setattr(self, hist_list_name, [])

            R_label = str(jetR).replace('.', '') + 'Scaled'

            for jet_level in self.jet_levels:
                # ENC histograms (jet level == part level)
                for ipoint in range(2, self.npoint+1):
                    name = 'h_ENC{}_JetPt_{}_R{}_trk00'.format(str(ipoint), jet_level, R_label)
                    print('Initialize histogram',name)
                    pt_bins = linbins(0,200,200)
                    RL_bins = logbins(1E-4,1,50)
                    h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                    h.GetXaxis().SetTitle('pT (jet)')
                    h.GetYaxis().SetTitle('R_{L}')
                    setattr(self, name, h)
                    getattr(self, hist_list_name).append(h)

                    name = 'h_ENC{}_JetPt_{}_R{}_trk10'.format(str(ipoint), jet_level, R_label)
                    print('Initialize histogram',name)
                    pt_bins = linbins(0,200,200)
                    RL_bins = logbins(1E-4,1,50)
                    h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                    h.GetXaxis().SetTitle('pT (jet)')
                    h.GetYaxis().SetTitle('R_{L}')
                    setattr(self, name, h)
                    getattr(self, hist_list_name).append(h)

                    # only save charge separation for pT>1GeV for now
                    if jet_level == "ch":
                        name = 'h_ENC{}_JetPt_{}_R{}_unlike_trk10'.format(str(ipoint), jet_level, R_label)
                        print('Initialize histogram',name)
                        pt_bins = linbins(0,200,200)
                        RL_bins = logbins(1E-4,1,50)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                        h.GetXaxis().SetTitle('pT (jet)')
                        h.GetYaxis().SetTitle('R_{L}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        name = 'h_ENC{}_JetPt_{}_R{}_like_trk10'.format(str(ipoint), jet_level, R_label)
                        print('Initialize histogram',name)
                        pt_bins = linbins(0,200,200)
                        RL_bins = logbins(1E-4,1,50)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                        h.GetXaxis().SetTitle('pT (jet)')
                        h.GetYaxis().SetTitle('R_{L}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                # Jet pt vs N constituents
                name = 'h_Nconst_JetPt_{}_R{}_trk00'.format(jet_level, R_label)
                print('Initialize histogram',name)
                pt_bins = linbins(0,200,200)
                Nconst_bins = linbins(0,50,50)
                h = ROOT.TH2D(name, name, 200, pt_bins, 50, Nconst_bins)
                h.GetXaxis().SetTitle('pT (jet)')
                h.GetYaxis().SetTitle('N_{const}')
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)

                name = 'h_Nconst_JetPt_{}_R{}_trk10'.format(jet_level, R_label)
                print('Initialize histogram',name)
                pt_bins = linbins(0,200,200)
                Nconst_bins = linbins(0,50,50)
                h = ROOT.TH2D(name, name, 200, pt_bins, 50, Nconst_bins)
                h.GetXaxis().SetTitle('pT (jet)')
                h.GetYaxis().SetTitle('N_{const}')
                setattr(self, name, h)
                getattr(self, hist_list_name).append(h)

                # NB: Only do the cone check for one reference radius and charged jets for now
                if self.beyond_jetR and (jetR == self.ref_jetR) and (jet_level == self.ref_jet_level):
                    for part_level in self.part_levels:
                        for ipoint in range(2, self.npoint+1):
                            name = 'h_ENC{}_cone_max_JetPt_{}_R{}_{}_trk00'.format(str(ipoint), jet_level, R_label, part_level)
                            print('Initialize histogram',name)
                            pt_bins = linbins(0,200,200)
                            RL_bins = logbins(1E-4,1,50)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('pT (jet)')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                            name = 'h_ENC{}_cone_max_JetPt_{}_R{}_{}_trk10'.format(str(ipoint), jet_level, R_label, part_level)
                            print('Initialize histogram',name)
                            pt_bins = linbins(0,200,200)
                            RL_bins = logbins(1E-4,1,50)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('pT (jet)')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                            name = 'h_ENC{}_cone_jetR_JetPt_{}_R{}_{}_trk00'.format(str(ipoint), jet_level, R_label, part_level)
                            print('Initialize histogram',name)
                            pt_bins = linbins(0,200,200)
                            RL_bins = logbins(1E-4,1,50)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('pT (jet)')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                            name = 'h_ENC{}_cone_jetR_JetPt_{}_R{}_{}_trk10'.format(str(ipoint), jet_level, R_label, part_level)
                            print('Initialize histogram',name)
                            pt_bins = linbins(0,200,200)
                            RL_bins = logbins(1E-4,1,50)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('pT (jet)')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

            if self.do_matching and (jetR == self.ref_jetR):
                for jet_level in ['p', 'h', 'ch']:
                    tag_levels = ['']
                    if self.do_tagging:
                        tag_levels = ['-1', '1', '2', '3', '4', '5', '6', '21']
                    for tag_level in tag_levels:
                        for ipoint in range(2, self.npoint+1):
                            name = 'h_matched_ENC{}_JetPt_{}{}_R{}_trk00'.format(str(ipoint), jet_level, tag_level, R_label)
                            print('Initialize histogram',name)
                            pt_bins = linbins(0,200,200)
                            RL_bins = logbins(1E-4,1,50)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('pT (jet)')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                            name = 'h_matched_ENC{}_JetPt_{}{}_R{}_trk10'.format(str(ipoint), jet_level, tag_level, R_label)
                            print('Initialize histogram',name)
                            pt_bins = linbins(0,200,200)
                            RL_bins = logbins(1E-4,1,50)
                            h = ROOT.TH2D(name, name, 200, pt_bins, 50, RL_bins)
                            h.GetXaxis().SetTitle('pT (jet)')
                            h.GetYaxis().SetTitle('R_{L}')
                            setattr(self, name, h)
                            getattr(self, hist_list_name).append(h)

                        # Jet pt vs N constituents
                        name = 'h_matched_Nconst_JetPt_{}{}_R{}_trk00'.format(jet_level, tag_level, R_label)
                        print('Initialize histogram',name)
                        pt_bins = linbins(0,200,200)
                        Nconst_bins = linbins(0,50,50)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, Nconst_bins)
                        h.GetXaxis().SetTitle('pT (jet)')
                        h.GetYaxis().SetTitle('N_{const}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

                        name = 'h_matched_Nconst_JetPt_{}{}_R{}_trk10'.format(jet_level, tag_level, R_label)
                        print('Initialize histogram',name)
                        pt_bins = linbins(0,200,200)
                        Nconst_bins = linbins(0,50,50)
                        h = ROOT.TH2D(name, name, 200, pt_bins, 50, Nconst_bins)
                        h.GetXaxis().SetTitle('pT (jet)')
                        h.GetYaxis().SetTitle('N_{const}')
                        setattr(self, name, h)
                        getattr(self, hist_list_name).append(h)

    #---------------------------------------------------------------
    # Initiate jet defs, selectors, and sd (if required)
    #---------------------------------------------------------------
    def init_jet_tools(self):
        
        for jetR in self.jetR_list:
            jetR_str = str(jetR).replace('.', '')      
            
            # set up our jet definition and a jet selector
            jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
            setattr(self, "jet_def_R%s" % jetR_str, jet_def)
            print(jet_def)

        # pwarning('max eta for particles after hadronization set to', self.max_eta_hadron)
        track_selector_ch = fj.SelectorPtMin(0.15)
        setattr(self, "track_selector_ch", track_selector_ch)

        pfc_selector1 = fj.SelectorPtMin(1.)
        setattr(self, "pfc_def_10", pfc_selector1)

        for jetR in self.jetR_list:
            jetR_str = str(jetR).replace('.', '')
            
            jet_selector = fj.SelectorPtMin(5) & fj.SelectorAbsEtaMax(self.max_eta_hadron - jetR) # FIX ME: use 5 or lower? use it on all ch, h, p jets?
            setattr(self, "jet_selector_R%s" % jetR_str, jet_selector)

    #---------------------------------------------------------------
    # Calculate events and pass information on to jet finding
    #---------------------------------------------------------------
    def calculate_events(self, pythia):
        
        iev = 0  # Event loop count

        while iev < self.nev:
            if iev % 100 == 0:
                print('ievt',iev)

            if not pythia.next():
                continue

            self.event = pythia.event

            leading_parton1 = fj.PseudoJet(pythia.event[5].px(),pythia.event[5].py(),pythia.event[5].pz(),pythia.event[5].e())
            leading_parton2 = fj.PseudoJet(pythia.event[6].px(),pythia.event[6].py(),pythia.event[6].pz(),pythia.event[6].e())

            # save absolute value of pdg id into user index
            leading_parton1.set_user_index(abs(pythia.event[5].id()))
            leading_parton2.set_user_index(abs(pythia.event[6].id()))
            
            self.parton_parents = [leading_parton1, leading_parton2]

            self.parts_pythia_p = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal], 0, True) # final stable partons

            hstatus = pythia.forceHadronLevel()
            if not hstatus:
                continue

            # full particle level
            self.parts_pythia_h = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal], 0, True)

            # charged particle level
            self.parts_pythia_ch = pythiafjext.vectorize_select(pythia, [pythiafjext.kFinal, pythiafjext.kCharged], 0, True)

            # Some "accepted" events don't survive hadronization step -- keep track here
            self.hNevents.Fill(0)

            self.find_jets_fill_trees()

            iev += 1

    def find_parts_around_jets(self, jet, parts_pythia, cone_R):
        # select particles around jet axis
        parts = fj.vectorPJ()
        for part in parts_pythia:
            if jet.delta_R(part) <= cone_R:
                parts.push_back(part)
        
        return parts

    #---------------------------------------------------------------
    # Form EEC using a cone around certain type of jets
    #---------------------------------------------------------------
    def fill_beyond_jet_histograms(self, jet_level, part_level, jet, jetR, R_label):
        # fill EEC histograms for cone around jet axis

        # Get the particles at certain level
        if part_level == "p":
            parts_pythia  = self.parts_pythia_p
        if part_level == "h":
            parts_pythia  = self.parts_pythia_h
        if part_level == "ch":
            parts_pythia  = self.parts_pythia_ch

        pfc_selector1 = getattr(self, "pfc_def_10")

        # select beyond constituents
        _p_select_cone_max = self.find_parts_around_jets(jet, parts_pythia, 1.0) # select within dR < 1
        _p_select_cone_jetR = self.find_parts_around_jets(jet, _p_select_cone_max, jetR) # select within previously selected parts

        _p_select0_cone_max = fj.vectorPJ()
        _ = [_p_select0_cone_max.push_back(p) for p in _p_select_cone_max]

        _p_select1_cone_max = fj.vectorPJ()
        _ = [_p_select1_cone_max.push_back(p) for p in pfc_selector1(_p_select_cone_max)]

        _p_select0_cone_jetR = fj.vectorPJ()
        _ = [_p_select0_cone_jetR.push_back(p) for p in _p_select_cone_jetR]

        _p_select1_cone_jetR = fj.vectorPJ()
        _ = [_p_select1_cone_jetR.push_back(p) for p in pfc_selector1(_p_select_cone_jetR)]

        cb0_cone_max = ecorrel.CorrelatorBuilder(_p_select0_cone_max, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)
        cb1_cone_max = ecorrel.CorrelatorBuilder(_p_select1_cone_max, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)

        cb0_cone_jetR = ecorrel.CorrelatorBuilder(_p_select0_cone_jetR, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)
        cb1_cone_jetR = ecorrel.CorrelatorBuilder(_p_select1_cone_jetR, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)

        for ipoint in range(2, self.npoint+1):
            for index in range(cb0_cone_max.correlator(ipoint).rs().size()):
                    getattr(self, 'h_ENC{}_cone_max_JetPt_{}_R{}_{}_trk00'.format(str(ipoint), jet_level, R_label, part_level)).Fill(jet.perp(), cb0_cone_max.correlator(ipoint).rs()[index], cb0_cone_max.correlator(ipoint).weights()[index])
            for index in range(cb1_cone_max.correlator(ipoint).rs().size()):
                    getattr(self, 'h_ENC{}_cone_max_JetPt_{}_R{}_{}_trk10'.format(str(ipoint), jet_level, R_label, part_level)).Fill(jet.perp(), cb1_cone_max.correlator(ipoint).rs()[index], cb1_cone_max.correlator(ipoint).weights()[index])
            for index in range(cb0_cone_jetR.correlator(ipoint).rs().size()):
                    getattr(self, 'h_ENC{}_cone_jetR_JetPt_{}_R{}_{}_trk00'.format(str(ipoint), jet_level, R_label, part_level)).Fill(jet.perp(), cb0_cone_jetR.correlator(ipoint).rs()[index], cb0_cone_jetR.correlator(ipoint).weights()[index])
            for index in range(cb1_cone_jetR.correlator(ipoint).rs().size()):
                    getattr(self, 'h_ENC{}_cone_jetR_JetPt_{}_R{}_{}_trk10'.format(str(ipoint), jet_level, R_label, part_level)).Fill(jet.perp(), cb1_cone_jetR.correlator(ipoint).rs()[index], cb1_cone_jetR.correlator(ipoint).weights()[index])

    #---------------------------------------------------------------
    # Form EEC using jet constituents
    #---------------------------------------------------------------
    def fill_jet_histograms(self, level, jet, R_label):
        # fill EEC histograms for jet constituents
        pfc_selector1 = getattr(self, "pfc_def_10")

        # select all constituents with no cut
        _c_select0 = fj.vectorPJ()
        _ = [_c_select0.push_back(c) for c in jet.constituents()]
        cb0 = ecorrel.CorrelatorBuilder(_c_select0, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)

        # select constituents with 1 GeV cut
        _c_select1 = fj.vectorPJ()
        _ = [_c_select1.push_back(c) for c in pfc_selector1(jet.constituents())]
        cb1 = ecorrel.CorrelatorBuilder(_c_select1, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)

        for ipoint in range(2, self.npoint+1):
            for index in range(cb0.correlator(ipoint).rs().size()):
                    getattr(self, 'h_ENC{}_JetPt_{}_R{}_trk00'.format(str(ipoint), level, R_label)).Fill(jet.perp(), cb0.correlator(ipoint).rs()[index], cb0.correlator(ipoint).weights()[index])
            for index in range(cb1.correlator(ipoint).rs().size()):
                    getattr(self, 'h_ENC{}_JetPt_{}_R{}_trk10'.format(str(ipoint), level, R_label)).Fill(jet.perp(), cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])
            
        if level == "ch":
            for ipoint in range(2, self.npoint+1):
                # only fill trk pt > 1 GeV here for now
                for index in range(cb1.correlator(ipoint).rs().size()):
                    part1 = cb1.correlator(ipoint).indices1()[index]
                    part2 = cb1.correlator(ipoint).indices2()[index]
                    c1 = _c_select1[part1]
                    c2 = _c_select1[part2]
                    if pythiafjext.getPythia8Particle(c1).charge()*pythiafjext.getPythia8Particle(c2).charge() < 0:
                        # print("unlike-sign pair ",pythiafjext.getPythia8Particle(c1).id(),pythiafjext.getPythia8Particle(c2).id())
                        getattr(self, 'h_ENC{}_JetPt_{}_R{}_unlike_trk10'.format(str(ipoint), level, R_label)).Fill(jet.perp(), cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])
                    else:
                        # print("likesign pair ",pythiafjext.getPythia8Particle(c1).id(),pythiafjext.getPythia8Particle(c2).id())
                        getattr(self, 'h_ENC{}_JetPt_{}_R{}_like_trk10'.format(str(ipoint), level, R_label)).Fill(jet.perp(), cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])

        getattr(self, 'h_Nconst_JetPt_{}_R{}_trk00'.format(level, R_label)).Fill(jet.perp(), len(_c_select0))
        getattr(self, 'h_Nconst_JetPt_{}_R{}_trk10'.format(level, R_label)).Fill(jet.perp(), len(_c_select1))

    #---------------------------------------------------------------
    # Form EEC using jet constituents for matched jets
    #---------------------------------------------------------------
    def fill_matched_jet_histograms(self, level, jet, ref_jet, R_label):
        # use the jet pt for energy weight but use the ref_jet pt when fill jet samples into jet pt bins
        pfc_selector1 = getattr(self, "pfc_def_10")
        # print(level,'with number of constituents',len(jet.constituents()),'(',len(pfc_selector1(jet.constituents())),')')
        # print('jet pt',jet.perp(),'ref jet pt',ref_jet.perp())

        # select all constituents with no cut
        _c_select0 = fj.vectorPJ()
        _ = [_c_select0.push_back(c) for c in jet.constituents()]
        cb0 = ecorrel.CorrelatorBuilder(_c_select0, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)

        # select constituents with 1 GeV cut
        _c_select1 = fj.vectorPJ()
        _ = [_c_select1.push_back(c) for c in pfc_selector1(jet.constituents())]
        cb1 = ecorrel.CorrelatorBuilder(_c_select1, jet.perp(), self.npoint, self.npower, self.dphi_cut, self.deta_cut)

        if self.do_tagging:
            if (jet.user_index()>0 and jet.user_index()<7): # quarks (1-6)
                level=level+str(jet.user_index())
            elif jet.user_index()==9 or jet.user_index()==21: # gluons
                level=level+'21'
            else:
                level=level+'-1'

        for ipoint in range(2, self.npoint+1):
            for index in range(cb0.correlator(ipoint).rs().size()):
                    getattr(self, 'h_matched_ENC{}_JetPt_{}_R{}_trk00'.format(str(ipoint), level, R_label)).Fill(ref_jet.perp(), cb0.correlator(ipoint).rs()[index], cb0.correlator(ipoint).weights()[index])
            for index in range(cb1.correlator(ipoint).rs().size()):
                    getattr(self, 'h_matched_ENC{}_JetPt_{}_R{}_trk10'.format(str(ipoint), level, R_label)).Fill(ref_jet.perp(), cb1.correlator(ipoint).rs()[index], cb1.correlator(ipoint).weights()[index])

        getattr(self, 'h_matched_Nconst_JetPt_{}_R{}_trk00'.format(level, R_label)).Fill(ref_jet.perp(), len(_c_select0))
        getattr(self, 'h_matched_Nconst_JetPt_{}_R{}_trk10'.format(level, R_label)).Fill(ref_jet.perp(), len(_c_select1))

    #---------------------------------------------------------------
    # Find jets, do matching between levels, and fill histograms & trees
    #---------------------------------------------------------------
    def find_jets_fill_trees(self):
        # Loop over jet radii
        for jetR in self.jetR_list:

            jetR_str = str(jetR).replace('.', '')
            jet_selector = getattr(self, "jet_selector_R%s" % jetR_str)
            jet_def = getattr(self, "jet_def_R%s" % jetR_str)
            track_selector_ch = getattr(self, "track_selector_ch")

            jets_p = fj.sorted_by_pt(jet_selector(jet_def(self.parts_pythia_p)))
            jets_h = fj.sorted_by_pt(jet_selector(jet_def(self.parts_pythia_h)))
            jets_ch = fj.sorted_by_pt(jet_selector(jet_def(track_selector_ch(self.parts_pythia_ch))))

            #-------------------------------------------------------------
            # match parton jets to the leading parton pdg id
            for jet_p in jets_p:
                matched_parton_parents = []
                for parton_parent in self.parton_parents:
                    if parton_parent.perp()/jet_p.perp() < 0.1:
                        break
                    if parton_parent.perp()/jet_p.perp() > 10:
                        continue
                    if self.is_geo_matched(jet_p, parton_parent, jetR):
                        matched_parton_parents.append(parton_parent)
                    
                if len(matched_parton_parents)==1: # accept if there is one match only (NB: but mayb be used multiple times)
                    jet_p.set_user_index(matched_parton_parents[0].user_index()) # save pdg id to user index (NB: absolute value)
                    # print('parton jet R',jetR,'pt',jet_p.perp(),'phi',jet_p.phi(),'eta',jet_p.eta())
                    # print('matched leading parton',matched_parton_parents[0].user_index(),'pt',matched_parton_parents[0].perp(),'phi',matched_parton_parents[0].phi(),'eta',matched_parton_parents[0].eta())
                else:
                    jet_p.set_user_index(-1) # set user index to -1 fr no match case

            R_label = str(jetR).replace('.', '') + 'Scaled'

            for jet_level in self.jet_levels:
                # Get the jets at different levels
                if jet_level == "p":
                    jets = jets_p
                if jet_level == "h":
                    jets = jets_h
                if jet_level == "ch":
                    jets = jets_ch

                #-------------------------------------------------------------
                # loop over jets and fill EEC histograms with jet constituents
                for j in jets:
                    self.fill_jet_histograms(jet_level, j, R_label)

                #-------------------------------------------------------------
                # loop over jets and fill EEC histograms inside a cone around jets
                if self.beyond_jetR and (jetR == self.ref_jetR) and (jet_level == self.ref_jet_level):
                    for j in jets:
                        for part_level in self.part_levels:
                            self.fill_beyond_jet_histograms(jet_level, part_level, j, jetR, R_label)
            
            if self.do_matching and (jetR == self.ref_jetR):
                # Loop through jets and find all h jets that can be matched to ch
                jets_h_matched_to_ch = []
                for jet_ch in jets_ch:
                    matched_jets_h = []
                    for index_jet_h, jet_h in enumerate(jets_h):
                        if jet_h.perp()/jet_ch.perp() < 0.1:
                            break
                        if jet_h.perp()/jet_ch.perp() > 10:
                            continue
                        if self.is_geo_matched(jet_ch, jet_h, jetR):
                            matched_jets_h.append(index_jet_h)
                    
                    if len(matched_jets_h)==1: # accept if there is one match only (NB: but mayb be used multiple times)
                        jets_h_matched_to_ch.append(matched_jets_h[0])
                    else:
                        jets_h_matched_to_ch.append(-1)

                # Loop through jets and find all p jets that can be matched to ch
                jets_p_matched_to_ch = []
                for jet_ch in jets_ch:
                    matched_jets_p = []
                    for index_jet_p, jet_p in enumerate(jets_p):
                        if jet_p.perp()/jet_ch.perp() < 0.1:
                            break
                        if jet_p.perp()/jet_ch.perp() > 10:
                            continue
                        if self.is_geo_matched(jet_ch, jet_p, jetR):
                            matched_jets_p.append(index_jet_p)
                    
                    if len(matched_jets_p)==1: # accept if there is one match only (NB: but mayb be used multiple times)
                        jets_p_matched_to_ch.append(matched_jets_p[0])
                    else:
                        jets_p_matched_to_ch.append(-1)

                #-------------------------------------------------------------
                # loop over matched jets and fill EEC histograms with jet constituents
                nmatched_ch = 0
                for index_j_ch, j_ch in enumerate(jets_ch):
                    imatched_p = jets_p_matched_to_ch[index_j_ch]
                    imatched_h = jets_h_matched_to_ch[index_j_ch]
                    if imatched_p > -1 and imatched_h > -1:
                        j_p = jets_p[imatched_p]
                        j_h = jets_h[imatched_h]
                        # print('matched ch',j_ch.perp(),'phi',j_ch.phi(),'eta',j_ch.eta())
                        # print('matched h',j_h.perp(),'phi',j_h.phi(),'eta',j_h.eta(),'dR',j_ch.delta_R(j_h))
                        # print('matched p',j_p.perp(),'phi',j_p.phi(),'eta',j_p.eta(),'dR',j_ch.delta_R(j_p))
                        nmatched_ch += 1

                        # used matched parton jet to tag the ch and h jet (qurak or gluon jet)
                        j_ch.set_user_index(j_p.user_index())
                        j_h.set_user_index(j_p.user_index())

                        # fill histograms
                        self.fill_matched_jet_histograms('ch', j_ch, j_ch, R_label)
                        self.fill_matched_jet_histograms('p', j_p, j_ch, R_label)
                        self.fill_matched_jet_histograms('h', j_h, j_ch, R_label)

                # print('matching efficiency:',nmatched_ch,'/',len(jets_ch))
                    
    #---------------------------------------------------------------
    # Compare two jets and store matching candidates in user_info
    #---------------------------------------------------------------
    def is_geo_matched(self, jet1, jet2, jetR):
        deltaR = jet1.delta_R(jet2)
      
        # Add a matching candidate to the list if it is within the geometrical cut
        if deltaR < self.jet_matching_distance * jetR:
            return True
        else:
            return False

    #---------------------------------------------------------------
    # Initiate scaling of all histograms and print final simulation info
    #---------------------------------------------------------------
    def scale_print_final_info(self, pythia):
        # Scale all jet histograms by the appropriate factor from generated cross section and the number of accepted events
        scale_f = pythia.info.sigmaGen() / self.hNevents.GetBinContent(1)
        print("scaling factor is",scale_f)

        for jetR in self.jetR_list:
            hist_list_name = "hist_list_R%s" % str(jetR).replace('.', '') 
            for h in getattr(self, hist_list_name):
                h.Scale(scale_f)

        print("N total final events:", int(self.hNevents.GetBinContent(1)), "with",
              int(pythia.info.nAccepted() - self.hNevents.GetBinContent(1)),
              "events rejected at hadronization step")
        self.hNevents.SetBinError(1, 0)

################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pythia8 fastjet on the fly',
                                     prog=os.path.basename(__file__))
    pyconf.add_standard_pythia_args(parser)
    # Could use --py-seed
    parser.add_argument('-o', '--output-dir', action='store', type=str, default='./', 
                        help='Output directory for generated ROOT file(s)')
    parser.add_argument('--tree-output-fname', default="AnalysisResults.root", type=str,
                        help="Filename for the (unscaled) generated particle ROOT TTree")
    parser.add_argument('-c', '--config_file', action='store', type=str, default='config/analysis_config.yaml',
                        help="Path of config file for observable configurations")

    args = parser.parse_args()

    # If invalid configFile is given, exit
    if not os.path.exists(args.config_file):
        print('File \"{0}\" does not exist! Exiting!'.format(args.configFile))
        sys.exit(0)

    # Have at least 1 event
    if args.nev < 1:
        args.nev = 1

    process = PythiaGenENC(config_file=args.config_file, output_dir=args.output_dir, args=args)
    process.pythia_parton_hadron(args)