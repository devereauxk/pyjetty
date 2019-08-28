#!/usr/bin/env python

import os
import argparse
import uproot
import ROOT as r
import pandas as pd
import joblib
import time

import fastjet as fj
import fjcontrib
import fjext

#----------------------------------------------------------------------
# this is from examples/python/02-area.py
def print_jets(jets):
	print("{0:>5s} {1:>10s} {2:>10s} {3:>10s} {4:>10s}".format(
		"jet #", "pt", "rap", "phi", "area"))
	for ijet in range(len(jets)):
		jet = jets[ijet]
		print("{0:5d} {1:10.3f} {2:10.4f} {3:10.4f} {3:10.4f}".format(
			ijet, jet.pt(), jet.rap(), jet.phi(), jet.area))


def fj_example_02_area(event):
	# cluster the event
	jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.4)
	area_def = fj.AreaDefinition(fj.active_area, fj.GhostedAreaSpec(5.0))
	cs = fj.ClusterSequenceArea(event, jet_def, area_def)
	jets = fj.SelectorPtMin(5.0)(fj.sorted_by_pt(cs.inclusive_jets()))
	print("jet def:", jet_def)
	print("area def:", area_def)
	print("#-------------------- initial jets --------------------")
	print_jets(jets)
	#----------------------------------------------------------------------
	# estimate the background
	maxrap       = 4.0
	grid_spacing = 0.55
	gmbge = fj.GridMedianBackgroundEstimator(maxrap, grid_spacing)
	gmbge.set_particles(event)
	print("#-------------------- background properties --------------------")
	print("rho   = ", gmbge.rho())
	print("sigma = ", gmbge.sigma())
	print()	
	#----------------------------------------------------------------------
	# subtract the jets
	subtractor = fj.Subtractor(gmbge)
	subtracted_jets = subtractor(jets)
	print("#-------------------- subtracted jets --------------------")
	print_jets(subtracted_jets)
#----------------------------------------------------------------------


# MP note: read more about uproot unpacking at https://github.com/scikit-hep/uproot#filling-pandas-dataframes
# MP note: there may be more efficient way to do this...

def main(args):
	fname = args.fname
	file = uproot.open(fname)
	all_ttrees = dict(file.allitems(filterclass=lambda cls: issubclass(cls, uproot.tree.TTreeMethods)))
	tracks = all_ttrees[b'PWGHF_TreeCreator/tree_Particle;1']
	pds_trks = tracks.pandas.df() # entrystop=10)
	events = all_ttrees[b'PWGHF_TreeCreator/tree_event_char;1']
	pds_evs = events.pandas.df()

	# print the banner first
	fj.ClusterSequence.print_banner()

	# signal jet definition
	maxrap = 0.9
	jet_R0 = args.jetR
	jet_def = fj.JetDefinition(fj.antikt_algorithm, jet_R0)
	jet_selector = fj.SelectorPtMin(0.0) & fj.SelectorAbsEtaMax(1)
	jet_area_def = fj.AreaDefinition(fj.active_area, fj.GhostedAreaSpec(maxrap))
	print(jet_def)

	# background estimation
	grid_spacing = maxrap/10.
	gmbge = fj.GridMedianBackgroundEstimator(maxrap, grid_spacing)

	print()

	output_columns = ['evid', 'pt', 'eta', 'phi', 'area', 'ptsub']
	e_jets = pd.DataFrame(columns=output_columns)

	for i, e in pds_evs.iterrows():
		iev_id = int(e['ev_id'])
		run_number = int(e['run_number'])
		_ts = pds_trks.loc[pds_trks['ev_id'] == iev_id].loc[pds_trks['run_number'] == run_number]
		_tpsj = fjext.vectorize_pt_eta_phi(_ts['ParticlePt'].values, _ts['ParticleEta'].values, _ts['ParticlePhi'].values)
		# print('maximum particle rapidity:', max([psj.rap() for psj in _tpsj]))
		_cs = fj.ClusterSequenceArea(_tpsj, jet_def, jet_area_def)
		_jets = jet_selector(fj.sorted_by_pt(_cs.inclusive_jets()))
		gmbge.set_particles(_tpsj)
		# print("rho   = ", gmbge.rho(), "sigma = ", gmbge.sigma())
		_jets_df = pd.DataFrame([[iev_id, j.perp(), j.eta(), j.phi(), j.area(), j.perp() - gmbge.rho() * j.area()] for j in _jets], 
								columns=output_columns)
		e_jets = e_jets.append(_jets_df, ignore_index=True)
		# print('event', i, 'number of parts', len(_tpsj), 'number of jets', len(_jets))
		# print(_jets_df.describe())
		if args.fjsubtract:
			fj_example_02_area(_tpsj)

	# print(e_jets.describe())
	joblib.dump(e_jets, args.output)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='jet reco on alice data', prog=os.path.basename(__file__))
	parser.add_argument('-n', '--nevents', help='number of events', default=1000, type=int)
	parser.add_argument('-f', '--fname', help='input file name', type=str, default=None, required=True)
	parser.add_argument('-R', '--jetR', help='jet radius', default=0.4, type=float)
	parser.add_argument('-o', '--output', help='output file name', default='{}_output.joblib'.format(os.path.basename(__file__)), type=str)
	parser.add_argument('--fjsubtract', help='do and show fj subtraction', action='store_true', default=False)
	args = parser.parse_args()	
	main(args)
	#fname = '/Users/ploskon/data/HFtree_trains/13-06-2019/488_20190613-0256/unmerged/child_1/0001/AnalysisResults.root'
