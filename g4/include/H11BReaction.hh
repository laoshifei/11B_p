#ifndef H11BReaction_H
#define H11BReaction_H 1

#include "RunAction.hh"
#include "G4String.hh"
#include "G4HadronicInteraction.hh"
#include "globals.hh"

#include "DataStructure.hh"
#include "H11BCrossSection.hh"
#include "H11BParticleLabel.hh"
#include "RootIO.hh"
#include "Constants.hh"

#include <memory>

class G4GenericMessenger;

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
class H11BReaction : public G4HadronicInteraction
{
public:
  enum class ReactionChannel
  {
    Resonance162 = 0,
    Resonance675 = 1,
    DirectDecay3Alpha = 2,
    GammaCapture12C = 3
  };

  enum ResonanceType
  {
    Resonance162,
    Resonance675
  };

public:
  H11BReaction();
  ~H11BReaction() override;

public:
  G4HadFinalState* ApplyYourself(const G4HadProjectile& projectile, G4Nucleus& target) override;
  void ReactionKinematic(const G4HadProjectile& projectile, G4ParticleDefinition* target, G4ParticleDefinition* product1, G4ParticleDefinition* product2, G4ParticleDefinition* product3);
  void SetCrossSectionBiasFactorCommand(G4double factor);
  void Set162SequentialDecayFractionCommand(G4double fraction);
  void Set162Alpha0BranchingFractionCommand(G4double fraction);
  void Set675SequentialDecayFractionCommand(G4double fraction);
  void SetDirectDecayEnabledCommand(G4bool enabled);
  void Set162BWScaleFactorCommand(G4double factor);
  void Set675ScaleFactorCommand(G4double factor);
  void Set675AlphaDecayModelCommand(const G4String& model);
  void Set162AlphaDecayModelCommand(const G4String& model);
  void Set162BoseRadialBinsCommand(G4int bins);
  void Set162BoseMaxAttemptsCommand(G4int attempts);
  void Set675StrictL1FractionCommand(G4double value);
  void Set675StrictL13PhaseCommand(G4double value);
  void Set675StrictCoherentL13Command(G4bool enabled);
  void Set675StrictPermutationSymmetrizedCommand(G4bool enabled);
  void Set675Strict8BeLambdaEnergyCommand(G4double value);
  void Set675Strict8BeReducedWidthSquaredCommand(G4double value);
  void Set675StrictWeightMaxSafetyFactorCommand(G4double value);
  void Set675StrictWeightMaxScanCandidatesCommand(G4int value);
  void Set675StrictMaxSamplingAttemptsCommand(G4int value);
  void SetEnable675PrimaryAngularDistributionCommand(G4bool enabled);
  void Set675PrimaryAngularA1Command(G4double value);
  void Set675PrimaryAngularA2Command(G4double value);
  void SetEnable675Alpha1SecondaryAngularCorrelationCommand(G4bool enabled);
  void Set675Alpha1SecondaryA2Command(G4double value);
  void Set675Alpha1SecondaryA4Command(G4double value);
  void SetEnable162PrimaryAngularDistributionCommand(G4bool enabled);
  void Set162PrimaryAngularA1Command(G4double value);
  void Set162PrimaryAngularA2Command(G4double value);
  void Set162PrimaryAlpha0AngularA1Command(G4double value);
  void Set162PrimaryAlpha0AngularA2Command(G4double value);
  void SetEnable162Alpha1SecondaryAngularCorrelationCommand(G4bool enabled);
  void Set162Alpha1SecondaryA2Command(G4double value);
  void Set162Alpha1SecondaryA4Command(G4double value);
  void SetEnable162Gamma0AngularDistributionCommand(G4bool enabled);
  void Set162Gamma0AngularA1Command(G4double value);
  void Set162Gamma0AngularA2Command(G4double value);
  void SetEnable675GammaAngularDistributionCommand(G4bool enabled);
  void Set675GammaAngularA1Command(G4double value);
  void Set675GammaAngularA2Command(G4double value);
  void Set162GammaCaptureEnabledCommand(G4bool enabled);
  void Set675GammaCaptureEnabledCommand(G4bool enabled);
  void SetGammaBiasFactorCommand(G4double factor);
  void SetForcedDetailedChannelCommand(const G4String& channel);
  void PrintConfigCommand();

private:
  H11BReactionData reaction_data;
  ResonanceType resonance_type = Resonance162;
  ReactionChannel reaction_channel = ReactionChannel::Resonance162;
  H11BGammaResonance gamma_resonance = H11BGammaResonance::None;
  H11BGammaBranch gamma_branch = H11BGammaBranch::None;
  H11BReactionChannel forced_detailed_channel = H11BReactionChannel::Unknown;
  H11BCrossSectionComponents selected_components;
  std::unique_ptr<G4GenericMessenger> messenger;

private:
  G4bool SelectReactionChannel(G4double kinetic_energy_lab);
  void DefineCommands();
  void GenerateThreeBodyPhaseSpace(const G4HadProjectile& projectile, G4ParticleDefinition* target, G4ParticleDefinition* product1, G4ParticleDefinition* product2, G4ParticleDefinition* product3);
  void Generate675SymmetrizedCoherent3Alpha(const G4HadProjectile& projectile, G4ParticleDefinition* target, G4ParticleDefinition* product1, G4ParticleDefinition* product2, G4ParticleDefinition* product3);
  void Generate162L2Bose3Alpha(const G4HadProjectile& projectile, G4ParticleDefinition* target, G4ParticleDefinition* product1, G4ParticleDefinition* product2, G4ParticleDefinition* product3);
  void GenerateGammaCapture(const G4HadProjectile& projectile, G4ParticleDefinition* target);
  void Generate162Gamma0(const G4LorentzVector& initial_cm, const G4LorentzVector& initial_lab, G4ParticleDefinition* c12_ground, G4ParticleDefinition* gamma_particle, const G4ThreeVector& beam_axis_cm);
  void Generate162Gamma1Cascade(const G4LorentzVector& initial_cm, const G4LorentzVector& initial_lab, G4ParticleDefinition* c12_ground, G4ParticleDefinition* gamma_particle);
  void Generate675GammaLine(const G4LorentzVector& initial_cm, const G4LorentzVector& initial_lab, G4ParticleDefinition* c12_ground, G4ParticleDefinition* gamma_particle, const G4ThreeVector& beam_axis_cm);
  G4double SamplePhaseSpaceM23(G4double sqrt_s, G4double m_alpha) const;
  G4double Sample8Be2PlusExcitationEnergy(G4double ex_max);
  G4double Weight8Be2Plus(G4double ex_8Be, G4double ex_max) const;
  G4double WeightAlpha8BeFirstBreakup(G4double e_alpha8Be, G4double e_alpha8Be_ref) const;
  G4bool ChooseAlpha0Channel() const;
  G4double GetAlpha0BranchingRatio() const;
  const char* GetResonanceName() const;
};

#endif
