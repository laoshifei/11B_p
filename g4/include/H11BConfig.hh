#ifndef H11BConfig_H
#define H11BConfig_H 1

#include "Constants.hh"
#include "G4String.hh"
#include "globals.hh"

#include <utility>

namespace H11BConfig {
H11B162AlphaDecayModel Get162AlphaDecayModel();
void Set162AlphaDecayModel(const G4String& model);
const char* Get162AlphaDecayModelName();
G4int Get162BoseRadialBins();
void Set162BoseRadialBins(G4int bins);
G4int Get162BoseMaxAttempts();
void Set162BoseMaxAttempts(G4int attempts);
void Set162Alpha0BranchingFraction(G4double value);
G4double Get162Alpha0BranchingFraction();

G4bool GetEnable162PrimaryAngularDistribution();
void SetEnable162PrimaryAngularDistribution(G4bool enabled);
void Set162PrimaryAngularA1(G4double value);
void Set162PrimaryAngularA2(G4double value);
G4double Get162PrimaryAngularA1();
G4double Get162PrimaryAngularA2();
// alpha0-channel primary coefficients (distinct from the alpha1 values above)
void Set162PrimaryAlpha0AngularA1(G4double value);
void Set162PrimaryAlpha0AngularA2(G4double value);
G4double Get162PrimaryAlpha0AngularA1();
G4double Get162PrimaryAlpha0AngularA2();
std::pair<G4double, G4double> Get162PrimaryA1A2(G4bool alpha1_branch);

G4bool GetEnable675PrimaryAngularDistribution();
void SetEnable675PrimaryAngularDistribution(G4bool enabled);
void Set675PrimaryAngularA1(G4double value);
void Set675PrimaryAngularA2(G4double value);
G4double Get675PrimaryAngularA1();
G4double Get675PrimaryAngularA2();

G4bool GetEnable162Alpha1SecondaryAngularCorrelation();
void SetEnable162Alpha1SecondaryAngularCorrelation(G4bool enabled);
const char* Get162Alpha1SecondaryAngularCorrelationName();
void Set162Alpha1SecondaryA2(G4double value);
void Set162Alpha1SecondaryA4(G4double value);
G4double Get162Alpha1SecondaryA2();
G4double Get162Alpha1SecondaryA4();

G4bool GetEnable675Alpha1SecondaryAngularCorrelation();
void SetEnable675Alpha1SecondaryAngularCorrelation(G4bool enabled);
const char* Get675Alpha1SecondaryAngularCorrelationName();
void Set675Alpha1SecondaryA2(G4double value);
void Set675Alpha1SecondaryA4(G4double value);
G4double Get675Alpha1SecondaryA2();
G4double Get675Alpha1SecondaryA4();

H11B675AlphaDecayModel Get675AlphaDecayModel();
void Set675AlphaDecayModel(H11B675AlphaDecayModel model);
void Set675AlphaDecayModel(const G4String& model);
const char* Get675AlphaDecayModelName();
G4double Get675StrictL1Fraction();
void Set675StrictL1Fraction(G4double value);
G4double Get675StrictL13Phase();
void Set675StrictL13Phase(G4double value);
G4bool Get675StrictCoherentL13();
void Set675StrictCoherentL13(G4bool enabled);
G4bool Get675StrictPermutationSymmetrized();
void Set675StrictPermutationSymmetrized(G4bool enabled);
G4double Get675Strict8BeLambdaEnergy();
void Set675Strict8BeLambdaEnergy(G4double value);
G4double Get675Strict8BeReducedWidthSquared();
void Set675Strict8BeReducedWidthSquared(G4double value);
G4double Get675StrictWeightMaxSafetyFactor();
void Set675StrictWeightMaxSafetyFactor(G4double value);
G4int Get675StrictWeightMaxScanCandidates();
void Set675StrictWeightMaxScanCandidates(G4int value);
G4int Get675StrictMaxSamplingAttempts();
void Set675StrictMaxSamplingAttempts(G4int value);

G4bool GetEnable162Gamma0AngularDistribution();
void SetEnable162Gamma0AngularDistribution(G4bool enabled);
const char* Get162Gamma0AngularDistributionName();
H11BGammaAngularMode Get162Gamma0AngularModeForOutput();
void Set162Gamma0AngularA1(G4double value);
void Set162Gamma0AngularA2(G4double value);
G4double Get162Gamma0AngularA1();
G4double Get162Gamma0AngularA2();

G4bool GetEnable675GammaAngularDistribution();
void SetEnable675GammaAngularDistribution(G4bool enabled);
const char* Get675GammaAngularDistributionName();
H11BGammaAngularMode Get675GammaAngularModeForOutput();
void Set675GammaAngularA1(G4double value);
void Set675GammaAngularA2(G4double value);
G4double Get675GammaAngularA1();
G4double Get675GammaAngularA2();

G4bool Get162GammaCaptureEnabled();
void Set162GammaCaptureEnabled(G4bool enabled);

G4bool Get675GammaCaptureEnabled();
void Set675GammaCaptureEnabled(G4bool enabled);

G4double GetGammaBiasFactor();
void SetGammaBiasFactor(G4double factor);

} // namespace H11BConfig

#endif
