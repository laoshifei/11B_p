#ifndef RootIO_H
#define RootIO_H 1

#include "Constants.hh"
#include "DataStructure.hh"
#include <globals.hh>

#include "Rtypes.h"
#include "TFile.h"
#include "TTree.h"

class RootIO
{
public:
  RootIO();
  ~RootIO();

  void SetRandomSeed(ULong64_t seed);

  // Unified physics/detector output file:
  //   si_pixel_map -> reaction -> virtual_sphere -> event
  void OpenDataFile();
  void CloseDataFile();
  void FillReactionTree(H11BReactionData& data);
  void FillVirtualSphereTree(VirtualSphereData& data);
  void FillEventTree(EventData& data);


private:
  void CreateSiPixelMapTree();
  void CreateReactionTree();
  void CreateVirtualSphereTree();
  void CreateEventTree();

private:
  ULong64_t random_seed = 0;
  G4int output_schema_version = 9;
  G4int configured_162_alpha_decay_model = 0;
  G4int configured_162_bose_radial_bins = 0;
  G4int configured_162_bose_max_attempts = 0;
  G4double configured_162_alpha0_branching_fraction = H11BDefault162Alpha0BranchingFraction;
  G4double configured_162_alpha1_branching_fraction = 1.0 - H11BDefault162Alpha0BranchingFraction;
  G4int virtual_sphere_enabled = 0;
  G4double virtual_sphere_radius_mm = 0.;
  G4double virtual_sphere_thickness_um = 0.;
  G4int virtual_sphere_save_electrons = 0;
  G4int virtual_sphere_save_optical_photons = 0;
  G4double virtual_sphere_min_kinetic_energy_keV = 0.;

  TFile* data_file = nullptr;
  TTree* si_pixel_map_tree = nullptr;
  TTree* reaction_tree = nullptr;
  TTree* virtual_sphere_tree = nullptr;
  TTree* event_tree = nullptr;
  TTree* run_info_tree = nullptr;

  SiPixelMapData si_pixel_map_data;
  H11BReactionData reaction_data;
  VirtualSphereData virtual_sphere_data;
  EventData event_data;


  char file_name[1024];
};

#endif
