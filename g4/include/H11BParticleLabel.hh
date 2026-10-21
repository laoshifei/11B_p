#ifndef H11BParticleLabel_H
#define H11BParticleLabel_H 1

#include "globals.hh"

// Canonical generator-truth classification stored as integer ROOT branches.
// These values are analysis schema and must remain stable.
enum class H11BReactionChannel : G4int
{
  Unknown = 0,
  Seq162BeGround = 1,
  Seq162Be2Plus = 2,
  Direct162 = 3,
  Seq675Be2Plus = 4,
  Direct675 = 5,
  Gamma162 = 6,
  Gamma675 = 7
};

enum class H11BParticleRole : G4int
{
  Unknown = 0,
  FirstStepAlpha = 1,
  BeDecayAlpha = 2,
  DirectDecayAlpha = 3,
  PrimaryGamma = 4,
  CascadeGamma = 5,
  // Symmetric sequential sample: no physically unique first-emitted alpha.
  SymmetricSequentialAlpha = 6
};

enum class H11BParticleSource : G4int
{
  Unknown = 0,
  H11BReactionProduct = 1,
  TransportSecondary = 2,
  BeamParticle = 3
};

inline const char* H11BReactionChannelLabel(H11BReactionChannel channel)
{
  switch (channel) {
  case H11BReactionChannel::Seq162BeGround:
    return "162seq_Be_gs";
  case H11BReactionChannel::Seq162Be2Plus:
    return "162seq_Be2plus";
  case H11BReactionChannel::Direct162:
    return "162direct";
  case H11BReactionChannel::Seq675Be2Plus:
    return "675seq_Be2plus";
  case H11BReactionChannel::Direct675:
    return "675direct";
  case H11BReactionChannel::Gamma162:
    return "162gamma";
  case H11BReactionChannel::Gamma675:
    return "675gamma";
  case H11BReactionChannel::Unknown:
    return "unknown";
  }

  return "unknown";
}

#endif
