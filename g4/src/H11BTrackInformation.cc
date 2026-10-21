#include "H11BTrackInformation.hh"

#include "G4ios.hh"

namespace {
// G4HadFinalState has no user-information slot, but it preserves a distinct
// creator-model integer for every secondary. Values above the Geant4 physics
// model catalog range are used transiently as H11B tags and decoded before the
// track's first step. No analysis output depends on these internal values.
constexpr G4int kH11BCreatorTagBase = 50000;
constexpr G4int kH11BCreatorTagLimit = 50800;
} // namespace

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
H11BTrackInformation::H11BTrackInformation(H11BReactionChannel channel,
                                           H11BParticleRole role,
                                           H11BParticleSource source,
                                           G4int generator_index)
    : reaction_channel(channel), particle_role(role), particle_source(source),
      generator_particle_index(generator_index)
{
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
void H11BTrackInformation::Print() const
{
  G4cout << " H11B channel=" << static_cast<G4int>(reaction_channel)
         << " role=" << static_cast<G4int>(particle_role)
         << " source=" << static_cast<G4int>(particle_source)
         << " generator_index=" << generator_particle_index << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
G4int H11BTrackInformation::CreatorModelTag(H11BReactionChannel channel,
                                            H11BParticleRole role,
                                            G4int generator_index)
{
  return kH11BCreatorTagBase
      + 100 * static_cast<G4int>(channel)
      + 10 * static_cast<G4int>(role)
      + generator_index + 1;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
G4bool H11BTrackInformation::DecodeCreatorModelTag(
    G4int creator_model_id,
    H11BReactionChannel& channel,
    H11BParticleRole& role,
    G4int& generator_index)
{
  if (creator_model_id < kH11BCreatorTagBase
      || creator_model_id >= kH11BCreatorTagLimit)
    return false;

  const G4int encoded = creator_model_id - kH11BCreatorTagBase;
  const G4int channel_value = encoded / 100;
  const G4int role_value = (encoded % 100) / 10;
  const G4int index_value = encoded % 10 - 1;

  if (channel_value < static_cast<G4int>(H11BReactionChannel::Seq162BeGround)
      || channel_value > static_cast<G4int>(H11BReactionChannel::Gamma675)
      || role_value < static_cast<G4int>(H11BParticleRole::Unknown)
      || role_value > static_cast<G4int>(H11BParticleRole::SymmetricSequentialAlpha)
      || index_value < -1 || index_value > 2)
    return false;

  channel = static_cast<H11BReactionChannel>(channel_value);
  role = static_cast<H11BParticleRole>(role_value);
  generator_index = index_value;
  return true;
}
