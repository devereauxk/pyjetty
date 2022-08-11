#ifndef __PYJETTY_ECORREL_HH
#define __PYJETTY_ECORREL_HH

#include <vector>
#include <fastjet/PseudoJet.hh>
namespace EnergyCorrelators
{
	class CorrelatorsContainer
	{
		public:
			CorrelatorsContainer();
			virtual ~CorrelatorsContainer();
			void addwr(const double &w, const double &r);
			void addwr(const double &w, const double &r, const double &w1, const double &w2);
			void clear();
			std::vector<double> *weights();
			std::vector<double> *rs();
			std::vector<double> *rxw();
			std::vector<double> *weights1(); // weight of object 1 in the pair
			std::vector<double> *weights2(); // weight of object 2 in the pair

			const double *wa();
			const double *ra();

		private:
			std::vector<double> fr;
			std::vector<double> fw;
			std::vector<double> frxw;
			std::vector<double> fw1;
			std::vector<double> fw2;
	};

	class CorrelatorBuilder
	{
		public:
			CorrelatorBuilder();
			// note by default we use energy correlators - one could use different weighting... future: pass as a param
			CorrelatorBuilder(const std::vector<fastjet::PseudoJet> &parts, const double &scale, const int &nmax);
			CorrelatorsContainer *correlator(int n);
			virtual ~CorrelatorBuilder();

		private:
			int fncmax;
			std::vector<CorrelatorsContainer*> fec;
	};

	std::vector<fastjet::PseudoJet> constituents_as_vector(const fastjet::PseudoJet &jet);

	std::vector<fastjet::PseudoJet> merge_signal_background_pjvectors(const std::vector<fastjet::PseudoJet> &signal, 
																	  const std::vector<fastjet::PseudoJet> &background,
																	  const double pTcut,
																	  const int bg_index_start);
};

#endif
