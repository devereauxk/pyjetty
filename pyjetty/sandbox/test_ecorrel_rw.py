#!/usr/bin/env python3

from pyjetty.mputils.mputils import logbins
import operator as op
import itertools as it
import sys
import os
import argparse
from tqdm import tqdm
from heppy.pythiautils import configuration as pyconf
import pythiaext
import pythiafjext
import pythia8
import fjtools
import ecorrel
import fjcontrib
import fjext
import fastjet as fj
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
# ROOT.gSystem.AddDynamicPath('$HEPPY_DIR/external/roounfold/roounfold-current/lib')
# ROOT.gSystem.Load('libRooUnfold.dylib')
# ROOT.gSystem.AddDynamicPath('$PYJETTY_DIR/cpptools/lib')
# ROOT.gSystem.Load('libpyjetty_rutilext')
# _test = ROOT.RUtilExt.Test()
from tqdm import tqdm
import argparse
import os
import array


def get_args_from_settings(ssettings):
	sys.argv = sys.argv + ssettings.split()
	parser = argparse.ArgumentParser(description='pythia8 fastjet on the fly')
	pyconf.add_standard_pythia_args(parser)
	parser.add_argument('--output', default="test_ecorrel_rw.root", type=str)
	parser.add_argument('--user-seed', help='pythia seed',
											default=1111, type=int)
	args = parser.parse_args()
	return args


def main():
	mycfg = []
	ssettings = "--py-ecm 7000 --py-pthatmin 500"
	args = get_args_from_settings(ssettings)
	pythia_hard = pyconf.create_and_init_pythia_from_args(args, mycfg)
	jet_R0 = 0.5
	max_eta_jet = 1.9 # open CMS data
	max_eta_hadron = max_eta_jet + jet_R0 * 2.
	parts_selector_h = fj.SelectorAbsEtaMax(max_eta_hadron)
	# jet_selector = fj.SelectorPtMin(500.0) & fj.SelectorPtMax(550.0) & fj.SelectorAbsEtaMax(max_eta_hadron - 1.05 * jet_R0)
	jet_selector = fj.SelectorPtMin(500.0) & fj.SelectorPtMax(550.0) & fj.SelectorAbsEtaMax(max_eta_jet) 
	pfc_selector0 = fj.SelectorPtMin(0.)
	pfc_selector1 = fj.SelectorPtMin(1.)

	# print the banner first
	fj.ClusterSequence.print_banner()
	print()
	# set up our jet definition and a jet selector
	jet_def = fj.JetDefinition(fj.antikt_algorithm, jet_R0)
	print(jet_def)

	nbins = int(9.)
	lbins = logbins(1.e-3, 1., nbins)
	fout = ROOT.TFile(args.output, 'recreate')
	fout.cd()

	hec0 = []
	hec1 = []
	for i in range(3):
		h = ROOT.TH1F('hec0_{}'.format(i+2), 'hec0_{}'.format(i+2), nbins, lbins)
		hec0.append(h)
		h = ROOT.TH1F('hec1_{}'.format(i+2), 'hec1_{}'.format(i+2), nbins, lbins)
		hec1.append(h)

	hjpt = ROOT.TH1F('hjpt', 'hjpt', 10, 500, 550)
 
	for n in tqdm(range(args.nev)):
		if not pythia_hard.next():
				continue
		# parts_pythia_h = pythiafjext.vectorize_select(pythia_hard, [pythiafjext.kFinal, pythiafjext.kCharged], 0, True)
		parts_pythia_h = pythiafjext.vectorize_select(pythia_hard, [pythiafjext.kFinal], 0, True)
		parts_pythia_h_selected = parts_selector_h(parts_pythia_h)

		jets_h = fj.sorted_by_pt(jet_selector(jet_def(parts_pythia_h_selected))) 
		if len(jets_h) < 1:
				continue

		# print('njets:', hjpt.Integral())
		
		for j in jets_h:
			hjpt.Fill(j.perp())
			# note: the EEC,... takes vector<PseudoJet> while PseudoJet::constituents() returns a tuple in python
			# so we use a helper function (in SWIG only basic types typles handled easily...)
			# vconstits = ecorrel.constituents_as_vector(j)
			# eecs_alt = ecorrel.EEC(vconstits, scale=j.perp())   
			# e3cs_alt = ecorrel.E3C(vconstits, scale=j.perp())
			# e4cs_alt = ecorrel.E4C(vconstits, scale=j.perp())

   			# alternative: push constutents to a vector in python
			_v = fj.vectorPJ()
			_ = [_v.push_back(c) for c in j.constituents()]

			# select only charged constituents
			_vc = fj.vectorPJ()
			_ = [_vc.push_back(c) for c in j.constituents()
                            if pythiafjext.getPythia8Particle(c).isCharged()]
			# n-point correlator with all charged particles
			cb = ecorrel.CorrelatorBuilder(_v, j.perp(), 4)
   
   			# select only charged constituents with 1 GeV cut
			_vc1 = fj.vectorPJ()
			_ = [_vc1.push_back(c) for c in pfc_selector1(j.constituents())
                            if pythiafjext.getPythia8Particle(c).isCharged()]
			# n-point correlator with charged particles pt > 1
			cb1 = ecorrel.CorrelatorBuilder(_vc1, j.perp(), 4)
   
			for i in range(3):
				if cb.correlator(i+2).rs().size() > 0:
					hec0[i].FillN(	cb.correlator(i+2).rs().size(), 
                   					array.array('d', cb.correlator(i+2).rs()), 
                     				array.array('d', cb.correlator(i+2).weights()))
				if cb1.correlator(i+2).rs().size() > 0:
					hec1[i].FillN(	cb1.correlator(i+2).rs().size(),
                   					array.array('d', cb1.correlator(i+2).rs()), 
                     				array.array('d', cb1.correlator(i+2).weights()))

	njets = hjpt.Integral()
	if njets == 0:
		njets = 1.

	fout.cd()

	for hg in [hec0, hec1]:
		for i in range(3):
			hg[i].Sumw2()
			intg = hg[i].Integral()
			if intg > 0:
				hg[i].Scale(1./intg)
			if i > 0:
				fout.cd()
				hc = hg[i].Clone(hg[i].GetName() + '_ratio_to_EEC')
				hc.Sumw2()
				hc.Divide(hg[0])  
				hc.Write()

	pythia_hard.stat()

	fout.Write()
	fout.Close()
	print('[i] written ', fout.GetName())


if __name__ == '__main__':
	main()
