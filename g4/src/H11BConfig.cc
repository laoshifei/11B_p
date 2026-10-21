#include "H11BConfig.hh"

#include "G4Exception.hh"
#include "G4SystemOfUnits.hh"
#include "G4ios.hh"

#include <algorithm>
#include <cmath>
#include <utility>

namespace {
H11B162AlphaDecayModel g_162_alpha_decay_model = H11BDefault162AlphaDecayModel;
G4int g_162_bose_radial_bins = H11B162BoseDefaultRadialBins;
G4int g_162_bose_max_attempts = H11B162BoseDefaultMaxAttempts;
G4double g_162_alpha0_branching_fraction = H11BDefault162Alpha0BranchingFraction;
G4bool g_enable_162_primary_angular_distribution = H11BDefaultEnable162PrimaryAngularDistribution;
G4double g_162_primary_a1 = H11BDefault162PrimaryAngularA1;
G4double g_162_primary_a2 = H11BDefault162PrimaryAngularA2;
G4double g_162_primary_a0_a1 = H11BDefault162PrimaryAlpha0AngularA1;
G4double g_162_primary_a0_a2 = H11BDefault162PrimaryAlpha0AngularA2;

G4bool g_enable_675_primary_angular_distribution = H11BDefaultEnable675PrimaryAngularDistribution;
G4double g_675_primary_a1 = H11BDefault675PrimaryAngularA1;
G4double g_675_primary_a2 = H11BDefault675PrimaryAngularA2;

G4bool g_enable_162_alpha1_secondary_angular_correlation = H11BDefaultEnable162Alpha1SecondaryAngularCorrelation;
G4double g_162_alpha1_secondary_a2 = H11BDefault162Alpha1SecondaryA2;
G4double g_162_alpha1_secondary_a4 = H11BDefault162Alpha1SecondaryA4;
G4bool g_enable_675_alpha1_secondary_angular_correlation = H11BDefaultEnable675Alpha1SecondaryAngularCorrelation;
G4double g_675_alpha1_secondary_a2 = H11BDefault675Alpha1SecondaryA2;
G4double g_675_alpha1_secondary_a4 = H11BDefault675Alpha1SecondaryA4;

H11B675AlphaDecayModel g_675_alpha_decay_model = H11BDefault675AlphaDecayModel;
G4double g_675_strict_l1_fraction = H11B675StrictDefaultL1Fraction;
G4double g_675_strict_l13_phase = H11B675StrictDefaultL13Phase;
G4bool g_675_strict_coherent_l13 = H11B675StrictDefaultCoherentL13;
G4bool g_675_strict_permutation_symmetrized = H11B675StrictDefaultPermutationSymmetrized;
G4double g_675_strict_8be_lambda_energy = H11B675StrictDefault8BeLambdaEnergy;
G4double g_675_strict_8be_reduced_width_squared = H11B675StrictDefault8BeReducedWidthSquared;
G4double g_675_strict_weight_max_safety_factor = H11B675StrictDefaultWeightMaxSafetyFactor;
G4int g_675_strict_weight_max_scan_candidates = H11B675StrictDefaultWeightMaxScanCandidates;
G4int g_675_strict_max_sampling_attempts = H11B675StrictDefaultMaxSamplingAttempts;

G4bool g_enable_162_gamma0_angular_distribution = H11BDefaultEnable162Gamma0AngularDistribution;
G4double g_162_gamma0_a1 = H11B162Gamma0AngularA1Default;
G4double g_162_gamma0_a2 = H11B162Gamma0AngularA2Default;
G4bool g_enable_675_gamma_angular_distribution = H11BDefaultEnable675GammaAngularDistribution;
G4double g_675_gamma_a1 = H11B675GammaAngularA1Default;
G4double g_675_gamma_a2 = H11B675GammaAngularA2Default;
G4bool g_enable_162_gamma_capture = H11BDefault162GammaCaptureEnabled;
G4bool g_enable_675_gamma_capture = H11BDefault675GammaCaptureEnabled;
G4double g_gamma_bias_factor = H11BDefaultGammaBiasFactor;

G4bool IsValidFactor(G4double factor)
{
  return std::isfinite(factor) && factor >= 1.0;
}

G4bool IsFiniteNonNegative(G4double value)
{
  return std::isfinite(value) && value >= 0.0;
}

G4bool IsFinitePositive(G4double value)
{
  return std::isfinite(value) && value > 0.0;
}

G4double Clamp01(G4double value)
{
  return std::clamp(value, 0.0, 1.0);
}

G4double WeightPrimaryA1A2(G4double x, G4double a1, G4double a2)
{
  const G4double p1 = x;
  const G4double p2 = 0.5 * (3.0 * x * x - 1.0);
  return 1.0 + a1 * p1 + a2 * p2;
}

void ValidatePrimaryAngularDistribution(const char* where, const char* code, const char* message,
                                        G4bool enabled, G4double a1, G4double a2)
{
  if (!enabled) {
    return;
  }

  constexpr G4int n_grid = 2000;
  for (G4int i = 0; i <= n_grid; ++i) {
    const G4double x = -1.0 + 2.0 * static_cast<G4double>(i) / n_grid;
    if (WeightPrimaryA1A2(x, a1, a2) < 0.0) {
      G4Exception(where, code, FatalException, message);
    }
  }
}

void Validate162PrimaryAngularDistribution()
{
  ValidatePrimaryAngularDistribution("H11BConfig::Validate162PrimaryAngularDistribution",
                                     "H11B162PrimaryAngular001",
                                     "162-keV primary alpha1 angular distribution has negative weight.",
                                     g_enable_162_primary_angular_distribution,
                                     g_162_primary_a1,
                                     g_162_primary_a2);
  ValidatePrimaryAngularDistribution("H11BConfig::Validate162PrimaryAngularDistribution",
                                     "H11B162PrimaryAngular002",
                                     "162-keV primary alpha0 angular distribution has negative weight.",
                                     g_enable_162_primary_angular_distribution,
                                     g_162_primary_a0_a1,
                                     g_162_primary_a0_a2);
}

void Validate675PrimaryAngularDistribution()
{
  ValidatePrimaryAngularDistribution("H11BConfig::Validate675PrimaryAngularDistribution",
                                     "H11B675PrimaryAngular001",
                                     "675-keV primary alpha angular distribution has negative weight.",
                                     g_enable_675_primary_angular_distribution,
                                     g_675_primary_a1,
                                     g_675_primary_a2);
}

void Validate162Gamma0AngularDistribution()
{
  ValidatePrimaryAngularDistribution("H11BConfig::Validate162Gamma0AngularDistribution",
                                     "H11B162Gamma0Angular001",
                                     "162-keV gamma0 angular distribution has negative weight.",
                                     g_enable_162_gamma0_angular_distribution,
                                     g_162_gamma0_a1,
                                     g_162_gamma0_a2);
}

void Validate675GammaAngularDistribution()
{
  ValidatePrimaryAngularDistribution("H11BConfig::Validate675GammaAngularDistribution",
                                     "H11B675GammaAngular001",
                                     "675-keV gamma angular distribution has negative weight.",
                                     g_enable_675_gamma_angular_distribution,
                                     g_675_gamma_a1,
                                     g_675_gamma_a2);
}

G4double WeightLegendreA2A4(G4double x, G4double a2, G4double a4)
{
  const G4double x2 = x * x;
  const G4double x4 = x2 * x2;
  const G4double p2 = 0.5 * (3.0 * x2 - 1.0);
  const G4double p4 = (35.0 * x4 - 30.0 * x2 + 3.0) / 8.0;
  return 1.0 + a2 * p2 + a4 * p4;
}

void ValidateLegendreA2A4(const char* where, const char* code, const char* message, G4double a2, G4double a4)
{
  constexpr G4int n_grid = 2000;
  for (G4int i = 0; i <= n_grid; ++i) {
    const G4double x = -1.0 + 2.0 * static_cast<G4double>(i) / n_grid;
    if (WeightLegendreA2A4(x, a2, a4) < 0.0) {
      G4Exception(where, code, FatalException, message);
    }
  }
}

void Validate162Alpha1SecondaryAngularDistribution()
{
  if (!g_enable_162_alpha1_secondary_angular_correlation) {
    return;
  }

  ValidateLegendreA2A4("H11BConfig::Validate162Alpha1SecondaryAngularDistribution",
                       "H11B162Alpha1Secondary001",
                       "162-keV alpha1 secondary angular correlation has negative weight.",
                       g_162_alpha1_secondary_a2,
                       g_162_alpha1_secondary_a4);
}

void Validate675Alpha1SecondaryAngularDistribution()
{
  if (!g_enable_675_alpha1_secondary_angular_correlation) {
    return;
  }

  ValidateLegendreA2A4("H11BConfig::Validate675Alpha1SecondaryAngularDistribution",
                       "H11B675Alpha1Secondary001",
                       "675-keV alpha1 secondary angular correlation has negative weight.",
                       g_675_alpha1_secondary_a2,
                       g_675_alpha1_secondary_a4);
}
} // namespace

namespace H11BConfig {
H11B162AlphaDecayModel Get162AlphaDecayModel() { return g_162_alpha_decay_model; }
const char* Get162AlphaDecayModelName()
{
  switch(g_162_alpha_decay_model) {
  case H11B162AlphaDecayModel::LegacyLegendreA2A4: return "legacyLegendreA2A4";
  case H11B162AlphaDecayModel::SinglePathL2BW: return "singlePathL2BW";
  case H11B162AlphaDecayModel::IncoherentL2BW: return "incoherentL2BW";
  case H11B162AlphaDecayModel::BoseSymmetricL2BW: return "boseSymmetricL2BW";
  }
  return "invalid";
}
void Set162AlphaDecayModel(const G4String& model)
{
  if(model=="legacyLegendreA2A4") g_162_alpha_decay_model=H11B162AlphaDecayModel::LegacyLegendreA2A4;
  else if(model=="singlePathL2BW") g_162_alpha_decay_model=H11B162AlphaDecayModel::SinglePathL2BW;
  else if(model=="incoherentL2BW") g_162_alpha_decay_model=H11B162AlphaDecayModel::IncoherentL2BW;
  else if(model=="boseSymmetricL2BW") g_162_alpha_decay_model=H11B162AlphaDecayModel::BoseSymmetricL2BW;
  else G4Exception("H11BConfig::Set162AlphaDecayModel", "H11B162BoseConfig", FatalException,
                   "Unknown 162 alpha1 model; no silent fallback.");
}
G4int Get162BoseRadialBins() { return g_162_bose_radial_bins; }
void Set162BoseRadialBins(G4int bins)
{
  if(bins<64 || bins>65536) {
    G4Exception("H11BConfig::Set162BoseRadialBins", "H11B162BoseGrid", FatalException,
                "Radial bins must be in [64, 65536].");
    return;
  }
  g_162_bose_radial_bins=bins;
}
G4int Get162BoseMaxAttempts() { return g_162_bose_max_attempts; }
void Set162BoseMaxAttempts(G4int attempts)
{
  if(attempts<1) {
    G4Exception("H11BConfig::Set162BoseMaxAttempts", "H11B162BoseAttempts", FatalException,
                "Max attempts must be positive.");
    return;
  }
  g_162_bose_max_attempts=attempts;
}

void Set162Alpha0BranchingFraction(G4double value)
{
  if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
    G4cerr << "Invalid /h11b/162Alpha0BranchingFraction " << value
           << ". Use a finite value in [0, 1]." << G4endl;
    return;
  }

  g_162_alpha0_branching_fraction = value;
}

G4double Get162Alpha0BranchingFraction()
{
  return g_162_alpha0_branching_fraction;
}

G4bool GetEnable162PrimaryAngularDistribution()
{
  return g_enable_162_primary_angular_distribution;
}

void SetEnable162PrimaryAngularDistribution(G4bool enabled)
{
  g_enable_162_primary_angular_distribution = enabled;
  Validate162PrimaryAngularDistribution();
}

void Set162PrimaryAngularA1(G4double value)
{
  g_162_primary_a1 = value;
  Validate162PrimaryAngularDistribution();
}

void Set162PrimaryAngularA2(G4double value)
{
  g_162_primary_a2 = value;
  Validate162PrimaryAngularDistribution();
}

G4double Get162PrimaryAngularA1()
{
  return g_162_primary_a1;
}

G4double Get162PrimaryAngularA2()
{
  return g_162_primary_a2;
}

void Set162PrimaryAlpha0AngularA1(G4double value)
{
  g_162_primary_a0_a1 = value;
  Validate162PrimaryAngularDistribution();
}

void Set162PrimaryAlpha0AngularA2(G4double value)
{
  g_162_primary_a0_a2 = value;
  Validate162PrimaryAngularDistribution();
}

G4double Get162PrimaryAlpha0AngularA1()
{
  return g_162_primary_a0_a1;
}

G4double Get162PrimaryAlpha0AngularA2()
{
  return g_162_primary_a0_a2;
}

std::pair<G4double, G4double> Get162PrimaryA1A2(G4bool alpha1_branch)
{
  // alpha1 -> 8Be(2+): isotropic coefficients (Spraker 2012).
  // alpha0 -> 8Be(g.s.): Becker 1987 Fig.10 coefficients.
  if (alpha1_branch) {
    return {g_162_primary_a1, g_162_primary_a2};
  }
  return {g_162_primary_a0_a1, g_162_primary_a0_a2};
}

G4bool GetEnable675PrimaryAngularDistribution()
{
  return g_enable_675_primary_angular_distribution;
}

void SetEnable675PrimaryAngularDistribution(G4bool enabled)
{
  g_enable_675_primary_angular_distribution = enabled;
  Validate675PrimaryAngularDistribution();
}

void Set675PrimaryAngularA1(G4double value)
{
  g_675_primary_a1 = value;
  Validate675PrimaryAngularDistribution();
}

void Set675PrimaryAngularA2(G4double value)
{
  g_675_primary_a2 = value;
  Validate675PrimaryAngularDistribution();
}

G4double Get675PrimaryAngularA1()
{
  return g_675_primary_a1;
}

G4double Get675PrimaryAngularA2()
{
  return g_675_primary_a2;
}

G4bool GetEnable162Alpha1SecondaryAngularCorrelation()
{
  return g_enable_162_alpha1_secondary_angular_correlation;
}

void SetEnable162Alpha1SecondaryAngularCorrelation(G4bool enabled)
{
  g_enable_162_alpha1_secondary_angular_correlation = enabled;
  Validate162Alpha1SecondaryAngularDistribution();
}

const char* Get162Alpha1SecondaryAngularCorrelationName()
{
  return g_enable_162_alpha1_secondary_angular_correlation ? "legendreA2A4" : "isotropic";
}

void Set162Alpha1SecondaryA2(G4double value)
{
  g_162_alpha1_secondary_a2 = value;
  Validate162Alpha1SecondaryAngularDistribution();
}

void Set162Alpha1SecondaryA4(G4double value)
{
  g_162_alpha1_secondary_a4 = value;
  Validate162Alpha1SecondaryAngularDistribution();
}

G4double Get162Alpha1SecondaryA2()
{
  return g_162_alpha1_secondary_a2;
}

G4double Get162Alpha1SecondaryA4()
{
  return g_162_alpha1_secondary_a4;
}

G4bool GetEnable675Alpha1SecondaryAngularCorrelation()
{
  return g_enable_675_alpha1_secondary_angular_correlation;
}

void SetEnable675Alpha1SecondaryAngularCorrelation(G4bool enabled)
{
  g_enable_675_alpha1_secondary_angular_correlation = enabled;
  Validate675Alpha1SecondaryAngularDistribution();
}

const char* Get675Alpha1SecondaryAngularCorrelationName()
{
  return g_enable_675_alpha1_secondary_angular_correlation ? "legendreA2A4" : "isotropic";
}

void Set675Alpha1SecondaryA2(G4double value)
{
  g_675_alpha1_secondary_a2 = value;
  Validate675Alpha1SecondaryAngularDistribution();
}

void Set675Alpha1SecondaryA4(G4double value)
{
  g_675_alpha1_secondary_a4 = value;
  Validate675Alpha1SecondaryAngularDistribution();
}

G4double Get675Alpha1SecondaryA2()
{
  return g_675_alpha1_secondary_a2;
}

G4double Get675Alpha1SecondaryA4()
{
  return g_675_alpha1_secondary_a4;
}

H11B675AlphaDecayModel Get675AlphaDecayModel()
{
  return g_675_alpha_decay_model;
}

void Set675AlphaDecayModel(H11B675AlphaDecayModel model)
{
  g_675_alpha_decay_model = model;
}

void Set675AlphaDecayModel(const G4String& model)
{
  if (model == "legacy" || model == "legacyLegendreA2A4" || model == "sequentialLegendreA2A4") {
    Set675AlphaDecayModel(H11B675AlphaDecayModel::LegacyLegendreA2A4);
    return;
  }

  if (model == "strict" || model == "symmetrizedCoherentL1L3" || model == "coherentL1L3") {
    Set675AlphaDecayModel(H11B675AlphaDecayModel::SymmetrizedCoherentL1L3);
    return;
  }

  G4cerr << "Unknown /h11b/675AlphaDecayModel '" << model
         << "'. Use legacyLegendreA2A4 or symmetrizedCoherentL1L3." << G4endl;
}

const char* Get675AlphaDecayModelName()
{
  switch (g_675_alpha_decay_model) {
  case H11B675AlphaDecayModel::LegacyLegendreA2A4:
    return "legacyLegendreA2A4";
  case H11B675AlphaDecayModel::SymmetrizedCoherentL1L3:
    return "symmetrizedCoherentL1L3";
  }

  return "unknown";
}

G4double Get675StrictL1Fraction()
{
  return g_675_strict_l1_fraction;
}

void Set675StrictL1Fraction(G4double value)
{
  if (!std::isfinite(value)) {
    G4cerr << "Invalid /h11b/675StrictL1Fraction " << value << ". Use a finite value in [0,1]." << G4endl;
    return;
  }
  g_675_strict_l1_fraction = Clamp01(value);
}

G4double Get675StrictL13Phase()
{
  return g_675_strict_l13_phase;
}

void Set675StrictL13Phase(G4double value)
{
  if (!std::isfinite(value)) {
    G4cerr << "Invalid /h11b/675StrictL13Phase " << value << ". Use a finite value in radians." << G4endl;
    return;
  }
  g_675_strict_l13_phase = value;
}

G4bool Get675StrictCoherentL13()
{
  return g_675_strict_coherent_l13;
}

void Set675StrictCoherentL13(G4bool enabled)
{
  g_675_strict_coherent_l13 = enabled;
}

G4bool Get675StrictPermutationSymmetrized()
{
  return g_675_strict_permutation_symmetrized;
}

void Set675StrictPermutationSymmetrized(G4bool enabled)
{
  g_675_strict_permutation_symmetrized = enabled;
}

G4double Get675Strict8BeLambdaEnergy()
{
  return g_675_strict_8be_lambda_energy;
}

void Set675Strict8BeLambdaEnergy(G4double value)
{
  if (!IsFinitePositive(value)) {
    G4cerr << "Invalid /h11b/675Strict8BeLambdaEnergy " << value << ". Use a finite positive energy." << G4endl;
    return;
  }
  g_675_strict_8be_lambda_energy = value;
}

G4double Get675Strict8BeReducedWidthSquared()
{
  return g_675_strict_8be_reduced_width_squared;
}

void Set675Strict8BeReducedWidthSquared(G4double value)
{
  if (!IsFinitePositive(value)) {
    G4cerr << "Invalid /h11b/675Strict8BeReducedWidthSquared " << value << ". Use a finite positive energy." << G4endl;
    return;
  }
  g_675_strict_8be_reduced_width_squared = value;
}

G4double Get675StrictWeightMaxSafetyFactor()
{
  return g_675_strict_weight_max_safety_factor;
}

void Set675StrictWeightMaxSafetyFactor(G4double value)
{
  if (!IsFinitePositive(value)) {
    G4cerr << "Invalid /h11b/675StrictWeightMaxSafetyFactor " << value << ". Use a finite positive value." << G4endl;
    return;
  }
  g_675_strict_weight_max_safety_factor = value;
}

G4int Get675StrictWeightMaxScanCandidates()
{
  return g_675_strict_weight_max_scan_candidates;
}

void Set675StrictWeightMaxScanCandidates(G4int value)
{
  if (value < 1) {
    G4cerr << "Invalid /h11b/675StrictWeightMaxScanCandidates " << value << ". Use an integer >= 1." << G4endl;
    return;
  }
  g_675_strict_weight_max_scan_candidates = value;
}

G4int Get675StrictMaxSamplingAttempts()
{
  return g_675_strict_max_sampling_attempts;
}

void Set675StrictMaxSamplingAttempts(G4int value)
{
  if (value < 1) {
    G4cerr << "Invalid /h11b/675StrictMaxSamplingAttempts " << value << ". Use an integer >= 1." << G4endl;
    return;
  }
  g_675_strict_max_sampling_attempts = value;
}

G4bool GetEnable162Gamma0AngularDistribution()
{
  return g_enable_162_gamma0_angular_distribution;
}

void SetEnable162Gamma0AngularDistribution(G4bool enabled)
{
  g_enable_162_gamma0_angular_distribution = enabled;
  Validate162Gamma0AngularDistribution();
}

const char* Get162Gamma0AngularDistributionName()
{
  return g_enable_162_gamma0_angular_distribution ? "fixedA1A2" : "isotropic";
}

H11BGammaAngularMode Get162Gamma0AngularModeForOutput()
{
  return g_enable_162_gamma0_angular_distribution ? H11BGammaAngularMode::FixedA1A2 : H11BGammaAngularMode::Isotropic;
}

void Set162Gamma0AngularA1(G4double value)
{
  g_162_gamma0_a1 = value;
  Validate162Gamma0AngularDistribution();
}

void Set162Gamma0AngularA2(G4double value)
{
  g_162_gamma0_a2 = value;
  Validate162Gamma0AngularDistribution();
}

G4double Get162Gamma0AngularA1()
{
  return g_162_gamma0_a1;
}

G4double Get162Gamma0AngularA2()
{
  return g_162_gamma0_a2;
}

G4bool GetEnable675GammaAngularDistribution()
{
  return g_enable_675_gamma_angular_distribution;
}

void SetEnable675GammaAngularDistribution(G4bool enabled)
{
  g_enable_675_gamma_angular_distribution = enabled;
  Validate675GammaAngularDistribution();
}

const char* Get675GammaAngularDistributionName()
{
  return g_enable_675_gamma_angular_distribution ? "fixedA1A2" : "isotropic";
}

H11BGammaAngularMode Get675GammaAngularModeForOutput()
{
  return g_enable_675_gamma_angular_distribution ? H11BGammaAngularMode::FixedA1A2 : H11BGammaAngularMode::Isotropic;
}

void Set675GammaAngularA1(G4double value)
{
  g_675_gamma_a1 = value;
  Validate675GammaAngularDistribution();
}

void Set675GammaAngularA2(G4double value)
{
  g_675_gamma_a2 = value;
  Validate675GammaAngularDistribution();
}

G4double Get675GammaAngularA1()
{
  return g_675_gamma_a1;
}

G4double Get675GammaAngularA2()
{
  return g_675_gamma_a2;
}

G4bool Get162GammaCaptureEnabled()
{
  return g_enable_162_gamma_capture;
}

void Set162GammaCaptureEnabled(G4bool enabled)
{
  g_enable_162_gamma_capture = enabled;
}

G4bool Get675GammaCaptureEnabled()
{
  return g_enable_675_gamma_capture;
}

void Set675GammaCaptureEnabled(G4bool enabled)
{
  g_enable_675_gamma_capture = enabled;
}

G4double GetGammaBiasFactor()
{
  return g_gamma_bias_factor;
}

void SetGammaBiasFactor(G4double factor)
{
  if (!IsValidFactor(factor)) {
    G4cerr << "Invalid /h11b/gammaBiasFactor " << factor << ". Use a finite value >= 1." << G4endl;
    return;
  }

  g_gamma_bias_factor = factor;
}


} // namespace H11BConfig
