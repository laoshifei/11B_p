#ifndef H11B162BoseModel_H
#define H11B162BoseModel_H 1

#include <array>
#include <complex>
#include <functional>
#include <vector>

// A controlled 162-keV, unpolarized J=L=j=l'=2 amplitude model, in MeV.
// Uses the existing BW magnitude and BW phase, without R-matrix shift,
// external Coulomb/hard-sphere phases, or the three-body Coulomb correction.
// This is NOT a reproduction of the complete Laursen M4 model.
namespace H11B162Bose {
using Vector = std::array<double, 3>;
struct FourVector { Vector p; double e; };
using Event = std::array<FourVector, 3>;
enum class Mode { SinglePath = 1, Incoherent = 2, Coherent = 3 };

struct Parameters {
  double pole = 3.12184; // above two-alpha threshold, not excitation above 8Be(gs)
  double width = 1.513;
  bool full_pair_penetrability = true;
  bool full_primary_penetrability = true;
};

// The sampler and amplitude use the SAME piecewise-linear radial density.
// This makes the mixture proposal and its analytic rejection bound exact
// for the specified tabulated model. Grid convergence is tested separately.
class RadialTable {
public:
  RadialTable(double available_energy, double alpha_mass,
              Parameters parameters = {}, int bins = 1024);
  double Density(double pair_energy) const;
  double SampleEnergy(double uniform) const;
  double Width(double pair_energy) const;
  double RawDensity(double pair_energy) const;
  double available_energy;
  double alpha_mass;
  Parameters parameters;
  int bins;
private:
  double step;
  std::vector<double> density, cdf;
};

struct Weights {
  double single = 0.0;
  double incoherent = 0.0;
  double coherent = 0.0;
  double acceptance = 0.0; // coherent / (3 * incoherent), bounded by Cauchy
};
struct Sample {
  Event event;
  Weights weights;
  int attempts = 0;
};

Vector BoostMomentum(const FourVector& v, const Vector& beta);
FourVector Boost(const FourVector& v, const Vector& beta);
std::array<std::complex<double>, 5> AngularAmplitude(const Vector& primary,
                                                   const Vector& secondary);
Weights Evaluate(const Event& event, const RadialTable& radial);
Sample Generate(const RadialTable& radial, Mode mode,
                const std::function<double()>& uniform, int max_attempts = 100000);
} // namespace H11B162Bose
#endif
