#include "RootIO.hh"

#include "H11BConfig.hh"
#include "OutputConfig.hh"
#include "OutputPath.hh"
#include "SiArray.hh"
#include "VirtualSphereConfig.hh"

#include "G4Threading.hh"
#include "G4SystemOfUnits.hh"
#include "G4ios.hh"

#include <cstddef>
#include <cstdio>
#include <ctime>

namespace {
void FillThreadedFileTag(char* file_name, std::size_t file_name_size)
{
  time_t t = time(nullptr);
  struct tm tt;
  localtime_r(&t, &tt);

  // Worker IDs are non-negative.  In sequential mode Geant4 returns a
  // negative sentinel, which is mapped to t0 because no master output file is
  // created.  The resulting name is always <timestamp>_t<thread>.root.
  const G4int geant4_thread_id = G4Threading::G4GetThreadId();
  const G4int output_thread_id = geant4_thread_id >= 0 ? geant4_thread_id : 0;
  snprintf(file_name, file_name_size, "%d%02d%02d_%02dh%02dm%02ds_t%d", tt.tm_year + 1900,
           tt.tm_mon + 1, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec, output_thread_id);
}
} // namespace

RootIO::RootIO()
{
  reaction_data.Clear();
  virtual_sphere_data.Clear();
  event_data.Clear();
}

RootIO::~RootIO()
{
  if (data_file) {
    delete data_file;
    data_file = nullptr;
  }
}

void RootIO::SetRandomSeed(ULong64_t seed)
{
  random_seed = seed;
}

void RootIO::OpenDataFile()
{
  if (data_file) {
    delete data_file;
    data_file = nullptr;
  }
  si_pixel_map_tree = nullptr;
  reaction_tree = nullptr;
  virtual_sphere_tree = nullptr;
  event_tree = nullptr;
  run_info_tree = nullptr;

  FillThreadedFileTag(file_name, sizeof(file_name));
  const auto output_path = HBOutputPath::MakeOutputFilePath(file_name);

  data_file = new TFile(output_path.string().c_str(), "RECREATE");
  if (!data_file || data_file->IsZombie()) {
    G4cout << "RootIO: failed to create " << output_path.string() << G4endl;
    delete data_file;
    data_file = nullptr;
    return;
  }

  // Create trees in the requested conceptual order. The close routine writes
  // them in the same order: si_pixel_map, reaction, virtual_sphere, event.
  CreateSiPixelMapTree();
  if (OutputConfig::GetSaveReaction()) CreateReactionTree();
  if (OutputConfig::GetSaveVirtualSphere() && VirtualSphereConfig::GetEnabled()) CreateVirtualSphereTree();
  if (OutputConfig::GetSaveEvent()) CreateEventTree();

  virtual_sphere_enabled = VirtualSphereConfig::GetEnabled() ? 1 : 0;
  configured_162_alpha0_branching_fraction = H11BConfig::Get162Alpha0BranchingFraction();
  configured_162_alpha1_branching_fraction = 1.0 - configured_162_alpha0_branching_fraction;
  configured_162_alpha_decay_model = static_cast<G4int>(H11BConfig::Get162AlphaDecayModel());
  configured_162_bose_radial_bins = H11BConfig::Get162BoseRadialBins();
  configured_162_bose_max_attempts = H11BConfig::Get162BoseMaxAttempts();
  virtual_sphere_radius_mm = VirtualSphereConfig::GetRadius() / mm;
  virtual_sphere_thickness_um = VirtualSphereConfig::GetThickness() / um;
  virtual_sphere_save_electrons = VirtualSphereConfig::GetSaveElectrons() ? 1 : 0;
  virtual_sphere_save_optical_photons = VirtualSphereConfig::GetSaveOpticalPhotons() ? 1 : 0;
  virtual_sphere_min_kinetic_energy_keV = VirtualSphereConfig::GetMinKineticEnergy() / keV;

  run_info_tree = new TTree("RunInfo", "run information");
  run_info_tree->Branch("random_seed", &random_seed, "random_seed/l");
  run_info_tree->Branch("output_schema_version", &output_schema_version, "output_schema_version/I");
  run_info_tree->Branch("configured_162_alpha_decay_model", &configured_162_alpha_decay_model,
                       "configured_162_alpha_decay_model/I");
  run_info_tree->Branch("configured_162_bose_radial_bins", &configured_162_bose_radial_bins,
                       "configured_162_bose_radial_bins/I");
  run_info_tree->Branch("configured_162_bose_max_attempts", &configured_162_bose_max_attempts,
                       "configured_162_bose_max_attempts/I");
  run_info_tree->Branch("configured_162_alpha0_branching_fraction",
                        &configured_162_alpha0_branching_fraction,
                        "configured_162_alpha0_branching_fraction/D");
  run_info_tree->Branch("configured_162_alpha1_branching_fraction",
                        &configured_162_alpha1_branching_fraction,
                        "configured_162_alpha1_branching_fraction/D");
  run_info_tree->Branch("virtual_sphere_enabled", &virtual_sphere_enabled, "virtual_sphere_enabled/I");
  run_info_tree->Branch("virtual_sphere_radius_mm", &virtual_sphere_radius_mm, "virtual_sphere_radius_mm/D");
  run_info_tree->Branch("virtual_sphere_thickness_um", &virtual_sphere_thickness_um,
                        "virtual_sphere_thickness_um/D");
  run_info_tree->Branch("virtual_sphere_save_electrons", &virtual_sphere_save_electrons,
                        "virtual_sphere_save_electrons/I");
  run_info_tree->Branch("virtual_sphere_save_optical_photons", &virtual_sphere_save_optical_photons,
                        "virtual_sphere_save_optical_photons/I");
  run_info_tree->Branch("virtual_sphere_min_kinetic_energy_keV", &virtual_sphere_min_kinetic_energy_keV,
                        "virtual_sphere_min_kinetic_energy_keV/D");
  if (G4Threading::G4GetThreadId() <= 0) run_info_tree->Fill();

  G4cout << "RootIO: opened unified output " << output_path.string() << G4endl;
}

void RootIO::CreateSiPixelMapTree()
{
  if (!data_file) return;
  data_file->cd();

  si_pixel_map_tree = new TTree("si_pixel_map", "DSSD ideal-pixel centre geometry");
  si_pixel_map_tree->Branch("detector_model", &si_pixel_map_data.detector_model, "detector_model/I");
  si_pixel_map_tree->Branch("subarray_id", &si_pixel_map_data.subarray_id, "subarray_id/I");
  si_pixel_map_tree->Branch("module_id", &si_pixel_map_data.module_id, "module_id/I");
  si_pixel_map_tree->Branch("front_strip_id", &si_pixel_map_data.front_strip_id, "front_strip_id/I");
  si_pixel_map_tree->Branch("back_strip_id", &si_pixel_map_data.back_strip_id, "back_strip_id/I");
  si_pixel_map_tree->Branch("x_center_mm", &si_pixel_map_data.x_center_mm, "x_center_mm/D");
  si_pixel_map_tree->Branch("y_center_mm", &si_pixel_map_data.y_center_mm, "y_center_mm/D");
  si_pixel_map_tree->Branch("z_center_mm", &si_pixel_map_data.z_center_mm, "z_center_mm/D");
  si_pixel_map_tree->Branch("theta_lab_center_deg", &si_pixel_map_data.theta_lab_center_deg,
                            "theta_lab_center_deg/D");
  si_pixel_map_tree->Branch("phi_lab_center_deg", &si_pixel_map_data.phi_lab_center_deg,
                            "phi_lab_center_deg/D");

  // In a multi-thread run every worker has the same static map. Fill it only
  // in worker 0 (or the serial thread -1) so hadd does not duplicate entries.
  const G4int thread_id = G4Threading::G4GetThreadId();
  if (thread_id <= 0) {
    const auto pixel_map = SiArray::BuildPixelMap();
    for (const auto& entry : pixel_map) {
      si_pixel_map_data = entry;
      si_pixel_map_tree->Fill();
    }
  }
}

void RootIO::CreateReactionTree()
{
  if (!data_file) return;
  data_file->cd();
  reaction_tree = new TTree("reaction", "reaction simulation data");
  reaction_tree->Branch("event", &reaction_data.event, "event/L");
  reaction_tree->Branch("e_alpha1", &reaction_data.e_alpha1, "e_alpha1/D");
  reaction_tree->Branch("e_alpha2", &reaction_data.e_alpha2, "e_alpha2/D");
  reaction_tree->Branch("e_alpha3", &reaction_data.e_alpha3, "e_alpha3/D");
  reaction_tree->Branch("theta_lab_alpha1", &reaction_data.theta_lab_alpha1, "theta_lab_alpha1/D");
  reaction_tree->Branch("theta_lab_alpha2", &reaction_data.theta_lab_alpha2, "theta_lab_alpha2/D");
  reaction_tree->Branch("theta_lab_alpha3", &reaction_data.theta_lab_alpha3, "theta_lab_alpha3/D");
  reaction_tree->Branch("phi_lab_alpha1", &reaction_data.phi_lab_alpha1, "phi_lab_alpha1/D");
  reaction_tree->Branch("phi_lab_alpha2", &reaction_data.phi_lab_alpha2, "phi_lab_alpha2/D");
  reaction_tree->Branch("phi_lab_alpha3", &reaction_data.phi_lab_alpha3, "phi_lab_alpha3/D");
  reaction_tree->Branch("resonance_id", &reaction_data.resonance_id, "resonance_id/I");
  reaction_tree->Branch("branch_id", &reaction_data.branch_id, "branch_id/I");
  reaction_tree->Branch("reaction_channel", &reaction_data.reaction_channel, "reaction_channel/I");
  reaction_tree->Branch("h11b_reaction_channel", &reaction_data.h11b_reaction_channel,
                        "h11b_reaction_channel/I");
  reaction_tree->Branch("alpha1_role", &reaction_data.alpha1_role, "alpha1_role/I");
  reaction_tree->Branch("alpha2_role", &reaction_data.alpha2_role, "alpha2_role/I");
  reaction_tree->Branch("alpha3_role", &reaction_data.alpha3_role, "alpha3_role/I");
  reaction_tree->Branch("gamma1_role", &reaction_data.gamma1_role, "gamma1_role/I");
  reaction_tree->Branch("gamma2_role", &reaction_data.gamma2_role, "gamma2_role/I");
  reaction_tree->Branch("background_mode", &reaction_data.background_mode, "background_mode/I");
  reaction_tree->Branch("gamma_resonance", &reaction_data.gamma_resonance, "gamma_resonance/I");
  reaction_tree->Branch("gamma_branch", &reaction_data.gamma_branch, "gamma_branch/I");
  reaction_tree->Branch("gamma_angular_mode", &reaction_data.gamma_angular_mode, "gamma_angular_mode/I");
  reaction_tree->Branch("n_prompt_gammas", &reaction_data.n_prompt_gammas, "n_prompt_gammas/I");
  reaction_tree->Branch("gamma1_energy", &reaction_data.gamma1_energy, "gamma1_energy/D");
  reaction_tree->Branch("gamma2_energy", &reaction_data.gamma2_energy, "gamma2_energy/D");
  reaction_tree->Branch("gamma1_theta_lab", &reaction_data.gamma1_theta_lab, "gamma1_theta_lab/D");
  reaction_tree->Branch("gamma2_theta_lab", &reaction_data.gamma2_theta_lab, "gamma2_theta_lab/D");
  reaction_tree->Branch("gamma1_phi_lab", &reaction_data.gamma1_phi_lab, "gamma1_phi_lab/D");
  reaction_tree->Branch("gamma2_phi_lab", &reaction_data.gamma2_phi_lab, "gamma2_phi_lab/D");
  reaction_tree->Branch("gamma1_theta_cm", &reaction_data.gamma1_theta_cm, "gamma1_theta_cm/D");
  reaction_tree->Branch("gamma2_theta_cm", &reaction_data.gamma2_theta_cm, "gamma2_theta_cm/D");
  reaction_tree->Branch("cos_theta_gamma_cm", &reaction_data.cos_theta_gamma_cm, "cos_theta_gamma_cm/D");
  reaction_tree->Branch("gamma_event_weight", &reaction_data.gamma_event_weight, "gamma_event_weight/D");
  reaction_tree->Branch("gamma_bias_factor", &reaction_data.gamma_bias_factor, "gamma_bias_factor/D");
  reaction_tree->Branch("event_sampling_weight", &reaction_data.event_sampling_weight, "event_sampling_weight/D");
  reaction_tree->Branch("gamma_final_state_energy_MeV", &reaction_data.gamma_final_state_energy_MeV, "gamma_final_state_energy_MeV/D");
  reaction_tree->Branch("gamma_primary_energy_MeV", &reaction_data.gamma_primary_energy_MeV, "gamma_primary_energy_MeV/D");
  reaction_tree->Branch("gamma_relative_intensity_used", &reaction_data.gamma_relative_intensity_used, "gamma_relative_intensity_used/D");
  reaction_tree->Branch("gamma_branch_fraction_used", &reaction_data.gamma_branch_fraction_used, "gamma_branch_fraction_used/D");
  reaction_tree->Branch("gamma_branch_is_upper_limit", &reaction_data.gamma_branch_is_upper_limit, "gamma_branch_is_upper_limit/I");
  reaction_tree->Branch("gamma_branch_from_relative_table", &reaction_data.gamma_branch_from_relative_table, "gamma_branch_from_relative_table/I");
  reaction_tree->Branch("gamma_cascade_generated", &reaction_data.gamma_cascade_generated, "gamma_cascade_generated/I");
  reaction_tree->Branch("enable_675_gamma_angular_distribution", &reaction_data.enable_675_gamma_angular_distribution, "enable_675_gamma_angular_distribution/I");
  reaction_tree->Branch("a1_675_gamma", &reaction_data.a1_675_gamma, "a1_675_gamma/D");
  reaction_tree->Branch("a2_675_gamma", &reaction_data.a2_675_gamma, "a2_675_gamma/D");
  reaction_tree->Branch("e_cm_p11B", &reaction_data.e_cm_p11B, "e_cm_p11B/D");
  reaction_tree->Branch("projectile_kinetic_lab", &reaction_data.projectile_kinetic_lab, "projectile_kinetic_lab/D");
  reaction_tree->Branch("projectile_px_lab", &reaction_data.projectile_px_lab, "projectile_px_lab/D");
  reaction_tree->Branch("projectile_py_lab", &reaction_data.projectile_py_lab, "projectile_py_lab/D");
  reaction_tree->Branch("projectile_pz_lab", &reaction_data.projectile_pz_lab, "projectile_pz_lab/D");
  reaction_tree->Branch("projectile_p_lab", &reaction_data.projectile_p_lab, "projectile_p_lab/D");
  reaction_tree->Branch("projectile_theta_lab", &reaction_data.projectile_theta_lab, "projectile_theta_lab/D");
  reaction_tree->Branch("projectile_phi_lab", &reaction_data.projectile_phi_lab, "projectile_phi_lab/D");
  reaction_tree->Branch("ex_max_8Be", &reaction_data.ex_max_8Be, "ex_max_8Be/D");
  reaction_tree->Branch("eaa_8Be", &reaction_data.eaa_8Be, "eaa_8Be/D");
  reaction_tree->Branch("e_alpha8Be", &reaction_data.e_alpha8Be, "e_alpha8Be/D");
  reaction_tree->Branch("cos_theta_primary_cm", &reaction_data.cos_theta_primary_cm, "cos_theta_primary_cm/D");
  reaction_tree->Branch("cos_chi_exit", &reaction_data.cos_chi_exit, "cos_chi_exit/D");
  reaction_tree->Branch("phi_primary_cm", &reaction_data.phi_primary_cm, "phi_primary_cm/D");
  reaction_tree->Branch("cos_chi_secondary_8be", &reaction_data.cos_chi_secondary_8be, "cos_chi_secondary_8be/D");
  reaction_tree->Branch("cos_theta_secondary_correlation", &reaction_data.cos_theta_secondary_correlation, "cos_theta_secondary_correlation/D");
  reaction_tree->Branch("primary_angular_mode", &reaction_data.primary_angular_mode, "primary_angular_mode/I");
  reaction_tree->Branch("enable_162_primary_angular_distribution", &reaction_data.enable_162_primary_angular_distribution, "enable_162_primary_angular_distribution/I");
  reaction_tree->Branch("a1_162_primary", &reaction_data.a1_162_primary, "a1_162_primary/D");
  reaction_tree->Branch("a2_162_primary", &reaction_data.a2_162_primary, "a2_162_primary/D");
  reaction_tree->Branch("enable_675_primary_angular_distribution", &reaction_data.enable_675_primary_angular_distribution, "enable_675_primary_angular_distribution/I");
  reaction_tree->Branch("a1_675_primary", &reaction_data.a1_675_primary, "a1_675_primary/D");
  reaction_tree->Branch("a2_675_primary", &reaction_data.a2_675_primary, "a2_675_primary/D");
  reaction_tree->Branch("primary_a1_used", &reaction_data.primary_a1_used, "primary_a1_used/D");
  reaction_tree->Branch("primary_a2_used", &reaction_data.primary_a2_used, "primary_a2_used/D");
  reaction_tree->Branch("h11b162_alpha_decay_model", &reaction_data.h11b162_alpha_decay_model, "h11b162_alpha_decay_model/I");
  reaction_tree->Branch("h11b162_decay_model_used", &reaction_data.h11b162_decay_model_used, "h11b162_decay_model_used/I");
  reaction_tree->Branch("h11b162_labels_are_bookkeeping", &reaction_data.h11b162_labels_are_bookkeeping, "h11b162_labels_are_bookkeeping/I");
  reaction_tree->Branch("h11b162_bose_radial_bins", &reaction_data.h11b162_bose_radial_bins, "h11b162_bose_radial_bins/I");
  reaction_tree->Branch("h11b162_bose_sampling_attempts", &reaction_data.h11b162_bose_sampling_attempts, "h11b162_bose_sampling_attempts/I");
  reaction_tree->Branch("h11b162_bose_incoherent_weight", &reaction_data.h11b162_bose_incoherent_weight, "h11b162_bose_incoherent_weight/D");
  reaction_tree->Branch("h11b162_bose_coherent_weight", &reaction_data.h11b162_bose_coherent_weight, "h11b162_bose_coherent_weight/D");
  reaction_tree->Branch("h11b162_bose_acceptance", &reaction_data.h11b162_bose_acceptance, "h11b162_bose_acceptance/D");
  reaction_tree->Branch("h11b675_decay_model", &reaction_data.h11b675_decay_model, "h11b675_decay_model/I");
  reaction_tree->Branch("h11b675_decay_model_used", &reaction_data.h11b675_decay_model_used, "h11b675_decay_model_used/I");
  reaction_tree->Branch("enable_675_alpha1_secondary_angular_correlation", &reaction_data.enable_675_alpha1_secondary_angular_correlation, "enable_675_alpha1_secondary_angular_correlation/I");
  reaction_tree->Branch("h11b675_alpha_decay_model", &reaction_data.h11b675_alpha_decay_model, "h11b675_alpha_decay_model/I");
  reaction_tree->Branch("h11b675_strict_coherent_l13", &reaction_data.h11b675_strict_coherent_l13, "h11b675_strict_coherent_l13/I");
  reaction_tree->Branch("h11b675_strict_permutation_symmetrized", &reaction_data.h11b675_strict_permutation_symmetrized, "h11b675_strict_permutation_symmetrized/I");
  reaction_tree->Branch("h11b675_strict_l1_fraction", &reaction_data.h11b675_strict_l1_fraction, "h11b675_strict_l1_fraction/D");
  reaction_tree->Branch("h11b675_strict_l13_phase", &reaction_data.h11b675_strict_l13_phase, "h11b675_strict_l13_phase/D");
  reaction_tree->Branch("h11b675_strict_8be_lambda_energy_keV", &reaction_data.h11b675_strict_8be_lambda_energy_keV, "h11b675_strict_8be_lambda_energy_keV/D");
  reaction_tree->Branch("h11b675_strict_8be_reduced_width_squared_keV", &reaction_data.h11b675_strict_8be_reduced_width_squared_keV, "h11b675_strict_8be_reduced_width_squared_keV/D");
  reaction_tree->Branch("h11b675_strict_weight", &reaction_data.h11b675_strict_weight, "h11b675_strict_weight/D");
  reaction_tree->Branch("h11b675_strict_weight_max", &reaction_data.h11b675_strict_weight_max, "h11b675_strict_weight_max/D");
  reaction_tree->Branch("h11b675_strict_sampling_attempts", &reaction_data.h11b675_strict_sampling_attempts, "h11b675_strict_sampling_attempts/I");
  reaction_tree->Branch("secondary_angular_model", &reaction_data.secondary_angular_model, "secondary_angular_model/I");
  reaction_tree->Branch("secondary_a2_used", &reaction_data.secondary_a2_used, "secondary_a2_used/D");
  reaction_tree->Branch("secondary_a4_used", &reaction_data.secondary_a4_used, "secondary_a4_used/D");
  reaction_tree->Branch("background_sequential_model", &reaction_data.background_sequential_model, "background_sequential_model/I");
  reaction_tree->Branch("cos_chi_675_internal", &reaction_data.cos_chi_675_internal, "cos_chi_675_internal/D");
  reaction_tree->Branch("phi_chi_675_internal", &reaction_data.phi_chi_675_internal, "phi_chi_675_internal/D");
  reaction_tree->Branch("opening_angle_alpha12_cm", &reaction_data.opening_angle_alpha12_cm, "opening_angle_alpha12_cm/D");
  reaction_tree->Branch("opening_angle_alpha13_cm", &reaction_data.opening_angle_alpha13_cm, "opening_angle_alpha13_cm/D");
  reaction_tree->Branch("opening_angle_alpha23_cm", &reaction_data.opening_angle_alpha23_cm, "opening_angle_alpha23_cm/D");
  reaction_tree->Branch("e_alpha1_cm", &reaction_data.e_alpha1_cm, "e_alpha1_cm/D");
  reaction_tree->Branch("e_alpha2_cm", &reaction_data.e_alpha2_cm, "e_alpha2_cm/D");
  reaction_tree->Branch("e_alpha3_cm", &reaction_data.e_alpha3_cm, "e_alpha3_cm/D");
  reaction_tree->Branch("e_8be_excitation", &reaction_data.e_8be_excitation, "e_8be_excitation/D");
  reaction_tree->Branch("event_weight", &reaction_data.event_weight, "event_weight/D");
  reaction_tree->Branch("e_3alpha_cm_alpha1", &reaction_data.e_3alpha_cm_alpha1, "e_3alpha_cm_alpha1/D");
  reaction_tree->Branch("e_3alpha_cm_alpha2", &reaction_data.e_3alpha_cm_alpha2, "e_3alpha_cm_alpha2/D");
  reaction_tree->Branch("e_3alpha_cm_alpha3", &reaction_data.e_3alpha_cm_alpha3, "e_3alpha_cm_alpha3/D");
  reaction_tree->Branch("ex_8Be", &reaction_data.ex_8Be, "ex_8Be/D");
  reaction_tree->Branch("e_8Be", &reaction_data.e_8Be, "e_8Be/D");
  reaction_tree->Branch("theta_lab_8Be", &reaction_data.theta_lab_8Be, "theta_lab_8Be/D");
  reaction_tree->Branch("phi_lab_8Be", &reaction_data.phi_lab_8Be, "phi_lab_8Be/D");
  reaction_tree->Branch("x", &reaction_data.x, "x/D");
  reaction_tree->Branch("y", &reaction_data.y, "y/D");
  reaction_tree->Branch("z", &reaction_data.z, "z/D");
  reaction_tree->Branch("sigma_eval_barn", &reaction_data.sigma_eval_barn, "sigma_eval_barn/D");
  reaction_tree->Branch("sigma_162_model_barn", &reaction_data.sigma_162_model_barn, "sigma_162_model_barn/D");
  reaction_tree->Branch("sigma_675_model_barn", &reaction_data.sigma_675_model_barn, "sigma_675_model_barn/D");
  reaction_tree->Branch("sigma_162_total_barn", &reaction_data.sigma_162_total_barn, "sigma_162_total_barn/D");
  reaction_tree->Branch("sigma_675_total_barn", &reaction_data.sigma_675_total_barn, "sigma_675_total_barn/D");
  reaction_tree->Branch("sigma_162_used_barn", &reaction_data.sigma_162_used_barn, "sigma_162_used_barn/D");
  reaction_tree->Branch("sigma_675_used_barn", &reaction_data.sigma_675_used_barn, "sigma_675_used_barn/D");
  reaction_tree->Branch("sigma_total_used_barn", &reaction_data.sigma_total_used_barn, "sigma_total_used_barn/D");
  reaction_tree->Branch("sigma_162_sampling_barn", &reaction_data.sigma_162_sampling_barn, "sigma_162_sampling_barn/D");
  reaction_tree->Branch("sigma_675_sampling_barn", &reaction_data.sigma_675_sampling_barn, "sigma_675_sampling_barn/D");
  reaction_tree->Branch("sigma_162_directdecay_sampling_barn", &reaction_data.sigma_162_directdecay_sampling_barn, "sigma_162_directdecay_sampling_barn/D");
  reaction_tree->Branch("sigma_675_directdecay_sampling_barn", &reaction_data.sigma_675_directdecay_sampling_barn, "sigma_675_directdecay_sampling_barn/D");
  reaction_tree->Branch("sigma_background_sampling_barn", &reaction_data.sigma_background_sampling_barn, "sigma_background_sampling_barn/D");
  reaction_tree->Branch("sigma_directdecay_sampling_barn", &reaction_data.sigma_directdecay_sampling_barn, "sigma_directdecay_sampling_barn/D");
  reaction_tree->Branch("sigma_3alpha_sampling_total_barn", &reaction_data.sigma_3alpha_sampling_total_barn, "sigma_3alpha_sampling_total_barn/D");
  reaction_tree->Branch("sigma_model_sum_barn", &reaction_data.sigma_model_sum_barn, "sigma_model_sum_barn/D");
  reaction_tree->Branch("model_scale_factor", &reaction_data.model_scale_factor, "model_scale_factor/D");
  reaction_tree->Branch("cross_section_bias_factor", &reaction_data.cross_section_bias_factor, "cross_section_bias_factor/D");
  reaction_tree->Branch("background_bias_factor", &reaction_data.background_bias_factor, "background_bias_factor/D");
  reaction_tree->Branch("direct_decay_fraction", &reaction_data.direct_decay_fraction, "direct_decay_fraction/D");
  reaction_tree->Branch("sequential_decay_fraction_162", &reaction_data.sequential_decay_fraction_162, "sequential_decay_fraction_162/D");
  reaction_tree->Branch("sequential_decay_fraction_675", &reaction_data.sequential_decay_fraction_675, "sequential_decay_fraction_675/D");
  reaction_tree->Branch("configured_162_alpha0_branching_fraction",
                        &reaction_data.configured_162_alpha0_branching_fraction,
                        "configured_162_alpha0_branching_fraction/D");
  reaction_tree->Branch("configured_162_alpha1_branching_fraction",
                        &reaction_data.configured_162_alpha1_branching_fraction,
                        "configured_162_alpha1_branching_fraction/D");
  reaction_tree->Branch("direct_decay_fraction_162", &reaction_data.direct_decay_fraction_162, "direct_decay_fraction_162/D");
  reaction_tree->Branch("direct_decay_fraction_675", &reaction_data.direct_decay_fraction_675, "direct_decay_fraction_675/D");
  reaction_tree->Branch("enable_direct_decay", &reaction_data.enable_direct_decay, "enable_direct_decay/I");
  reaction_tree->Branch("scale_factor_162", &reaction_data.scale_factor_162, "scale_factor_162/D");
  reaction_tree->Branch("scale_factor_675", &reaction_data.scale_factor_675, "scale_factor_675/D");
  reaction_tree->Branch("sigma_background_barn", &reaction_data.sigma_background_barn, "sigma_background_barn/D");
  reaction_tree->Branch("sigma_162_directdecay_barn", &reaction_data.sigma_162_directdecay_barn, "sigma_162_directdecay_barn/D");
  reaction_tree->Branch("sigma_675_directdecay_barn", &reaction_data.sigma_675_directdecay_barn, "sigma_675_directdecay_barn/D");
  reaction_tree->Branch("sigma_directdecay_barn", &reaction_data.sigma_directdecay_barn, "sigma_directdecay_barn/D");
  reaction_tree->Branch("sigma_3alpha_eval_barn", &reaction_data.sigma_3alpha_eval_barn, "sigma_3alpha_eval_barn/D");
  reaction_tree->Branch("sigma_gamma_162_0_barn", &reaction_data.sigma_gamma_162_0_barn, "sigma_gamma_162_0_barn/D");
  reaction_tree->Branch("sigma_gamma_162_1_barn", &reaction_data.sigma_gamma_162_1_barn, "sigma_gamma_162_1_barn/D");
  reaction_tree->Branch("sigma_gamma_162_total_barn", &reaction_data.sigma_gamma_162_total_barn, "sigma_gamma_162_total_barn/D");
  reaction_tree->Branch("sigma_gamma_675_total_barn", &reaction_data.sigma_gamma_675_total_barn, "sigma_gamma_675_total_barn/D");
  reaction_tree->Branch("sigma_gamma_total_barn", &reaction_data.sigma_gamma_total_barn, "sigma_gamma_total_barn/D");
  reaction_tree->Branch("sigma_total_physical_all_barn", &reaction_data.sigma_total_physical_all_barn, "sigma_total_physical_all_barn/D");
  reaction_tree->Branch("sigma_total_sampling_all_barn", &reaction_data.sigma_total_sampling_all_barn, "sigma_total_sampling_all_barn/D");
  reaction_tree->Branch("sigma_gamma_162_0_physical_barn", &reaction_data.sigma_gamma_162_0_physical_barn, "sigma_gamma_162_0_physical_barn/D");
  reaction_tree->Branch("sigma_gamma_162_1_physical_barn", &reaction_data.sigma_gamma_162_1_physical_barn, "sigma_gamma_162_1_physical_barn/D");
  reaction_tree->Branch("sigma_gamma_675_physical_barn", &reaction_data.sigma_gamma_675_physical_barn, "sigma_gamma_675_physical_barn/D");
  reaction_tree->Branch("sigma_gamma_162_0_sampling_barn", &reaction_data.sigma_gamma_162_0_sampling_barn, "sigma_gamma_162_0_sampling_barn/D");
  reaction_tree->Branch("sigma_gamma_162_1_sampling_barn", &reaction_data.sigma_gamma_162_1_sampling_barn, "sigma_gamma_162_1_sampling_barn/D");
  reaction_tree->Branch("sigma_gamma_675_sampling_barn", &reaction_data.sigma_gamma_675_sampling_barn, "sigma_gamma_675_sampling_barn/D");
  reaction_tree->Branch("sigma_gamma_675_to_ground_physical_barn", &reaction_data.sigma_gamma_675_to_ground_physical_barn, "sigma_gamma_675_to_ground_physical_barn/D");
  reaction_tree->Branch("sigma_gamma_675_to_4439_physical_barn", &reaction_data.sigma_gamma_675_to_4439_physical_barn, "sigma_gamma_675_to_4439_physical_barn/D");
  reaction_tree->Branch("sigma_gamma_675_to_7654_physical_barn", &reaction_data.sigma_gamma_675_to_7654_physical_barn, "sigma_gamma_675_to_7654_physical_barn/D");
  reaction_tree->Branch("sigma_gamma_675_to_12710_physical_barn", &reaction_data.sigma_gamma_675_to_12710_physical_barn, "sigma_gamma_675_to_12710_physical_barn/D");
  reaction_tree->Branch("sigma_gamma_675_to_15110_physical_barn", &reaction_data.sigma_gamma_675_to_15110_physical_barn, "sigma_gamma_675_to_15110_physical_barn/D");
  reaction_tree->Branch("sigma_gamma_675_to_ground_sampling_barn", &reaction_data.sigma_gamma_675_to_ground_sampling_barn, "sigma_gamma_675_to_ground_sampling_barn/D");
  reaction_tree->Branch("sigma_gamma_675_to_4439_sampling_barn", &reaction_data.sigma_gamma_675_to_4439_sampling_barn, "sigma_gamma_675_to_4439_sampling_barn/D");
  reaction_tree->Branch("sigma_gamma_675_to_7654_sampling_barn", &reaction_data.sigma_gamma_675_to_7654_sampling_barn, "sigma_gamma_675_to_7654_sampling_barn/D");
  reaction_tree->Branch("sigma_gamma_675_to_12710_sampling_barn", &reaction_data.sigma_gamma_675_to_12710_sampling_barn, "sigma_gamma_675_to_12710_sampling_barn/D");
  reaction_tree->Branch("sigma_gamma_675_to_15110_sampling_barn", &reaction_data.sigma_gamma_675_to_15110_sampling_barn, "sigma_gamma_675_to_15110_sampling_barn/D");
  reaction_tree->Branch("sigma_gamma_sampling_total_barn", &reaction_data.sigma_gamma_sampling_total_barn, "sigma_gamma_sampling_total_barn/D");
  reaction_tree->Branch("sigma_total_all_barn", &reaction_data.sigma_total_all_barn, "sigma_total_all_barn/D");
  reaction_tree->Branch("channel_probability_162", &reaction_data.channel_probability_162, "channel_probability_162/D");
  reaction_tree->Branch("channel_probability_675", &reaction_data.channel_probability_675, "channel_probability_675/D");
  reaction_tree->Branch("channel_probability_background", &reaction_data.channel_probability_background, "channel_probability_background/D");
  reaction_tree->Branch("channel_probability_directdecay", &reaction_data.channel_probability_directdecay, "channel_probability_directdecay/D");
  reaction_tree->Branch("channel_probability_directdecay_162", &reaction_data.channel_probability_directdecay_162, "channel_probability_directdecay_162/D");
  reaction_tree->Branch("channel_probability_directdecay_675", &reaction_data.channel_probability_directdecay_675, "channel_probability_directdecay_675/D");
  reaction_tree->Branch("channel_probability_gamma", &reaction_data.channel_probability_gamma, "channel_probability_gamma/D");
  reaction_tree->Branch("probability_gamma_162_0", &reaction_data.probability_gamma_162_0, "probability_gamma_162_0/D");
  reaction_tree->Branch("probability_gamma_162_1", &reaction_data.probability_gamma_162_1, "probability_gamma_162_1/D");
  reaction_tree->Branch("probability_gamma_675_total", &reaction_data.probability_gamma_675_total, "probability_gamma_675_total/D");
  reaction_tree->Branch("probability_gamma_675_to_ground", &reaction_data.probability_gamma_675_to_ground, "probability_gamma_675_to_ground/D");
  reaction_tree->Branch("probability_gamma_675_to_4439", &reaction_data.probability_gamma_675_to_4439, "probability_gamma_675_to_4439/D");
  reaction_tree->Branch("probability_gamma_675_to_7654", &reaction_data.probability_gamma_675_to_7654, "probability_gamma_675_to_7654/D");
  reaction_tree->Branch("probability_gamma_675_to_12710", &reaction_data.probability_gamma_675_to_12710, "probability_gamma_675_to_12710/D");
  reaction_tree->Branch("probability_gamma_675_to_15110", &reaction_data.probability_gamma_675_to_15110, "probability_gamma_675_to_15110/D");
  reaction_tree->Branch("reaction", reaction_data.reaction, "reaction/C");
}

void RootIO::CreateVirtualSphereTree()
{
  if (!data_file) return;
  data_file->cd();

  virtual_sphere_tree = new TTree("virtual_sphere", "first outward crossing of the virtual sphere");
  virtual_sphere_tree->Branch("event_id", &virtual_sphere_data.event_id, "event_id/L");
  virtual_sphere_tree->Branch("track_id", &virtual_sphere_data.track_id, "track_id/I");
  virtual_sphere_tree->Branch("parent_id", &virtual_sphere_data.parent_id, "parent_id/I");
  virtual_sphere_tree->Branch("pdg", &virtual_sphere_data.pdg, "pdg/I");
  virtual_sphere_tree->Branch("h11b_reaction_channel", &virtual_sphere_data.h11b_reaction_channel,
                              "h11b_reaction_channel/I");
  virtual_sphere_tree->Branch("h11b_particle_role", &virtual_sphere_data.h11b_particle_role,
                              "h11b_particle_role/I");
  virtual_sphere_tree->Branch("h11b_particle_source", &virtual_sphere_data.h11b_particle_source,
                              "h11b_particle_source/I");
  virtual_sphere_tree->Branch("generator_particle_index", &virtual_sphere_data.generator_particle_index,
                              "generator_particle_index/I");
  virtual_sphere_tree->Branch("kinetic_energy_MeV", &virtual_sphere_data.kinetic_energy_MeV,
                              "kinetic_energy_MeV/F");
  virtual_sphere_tree->Branch("px_MeV_c", &virtual_sphere_data.px_MeV_c, "px_MeV_c/F");
  virtual_sphere_tree->Branch("py_MeV_c", &virtual_sphere_data.py_MeV_c, "py_MeV_c/F");
  virtual_sphere_tree->Branch("pz_MeV_c", &virtual_sphere_data.pz_MeV_c, "pz_MeV_c/F");
  virtual_sphere_tree->Branch("x_mm", &virtual_sphere_data.x_mm, "x_mm/F");
  virtual_sphere_tree->Branch("y_mm", &virtual_sphere_data.y_mm, "y_mm/F");
  virtual_sphere_tree->Branch("z_mm", &virtual_sphere_data.z_mm, "z_mm/F");
  virtual_sphere_tree->Branch("theta_lab_deg", &virtual_sphere_data.theta_lab_deg, "theta_lab_deg/F");
  virtual_sphere_tree->Branch("phi_lab_deg", &virtual_sphere_data.phi_lab_deg, "phi_lab_deg/F");
  virtual_sphere_tree->Branch("global_time_ns", &virtual_sphere_data.global_time_ns, "global_time_ns/F");
}

void RootIO::CreateEventTree()
{
  if (!data_file) return;
  data_file->cd();

  event_tree = new TTree("event", "one entry per Geant4 event");
  event_tree->Branch("event_id", &event_data.event_id, "event_id/L");

  event_tree->Branch("si_subarray_id", &event_data.si_subarray_id);
  event_tree->Branch("si_module_id", &event_data.si_module_id);
  event_tree->Branch("si_side", &event_data.si_side);
  event_tree->Branch("si_strip_id", &event_data.si_strip_id);
  event_tree->Branch("si_edep_MeV", &event_data.si_edep_MeV);
  event_tree->Branch("si_time_ns", &event_data.si_time_ns);

  event_tree->Branch("si_hit_detector_id", &event_data.si_hit_detector_id);
  event_tree->Branch("si_hit_subarray_id", &event_data.si_hit_subarray_id);
  event_tree->Branch("si_hit_module_id", &event_data.si_hit_module_id);
  event_tree->Branch("si_hit_track_id", &event_data.si_hit_track_id);
  event_tree->Branch("si_hit_parent_id", &event_data.si_hit_parent_id);
  event_tree->Branch("si_hit_pdg", &event_data.si_hit_pdg);
  event_tree->Branch("si_hit_edep_MeV", &event_data.si_hit_edep_MeV);
  event_tree->Branch("si_hit_time_ns", &event_data.si_hit_time_ns);

  event_tree->Branch("si_hit_x_entry_mm", &event_data.si_hit_x_entry_mm);
  event_tree->Branch("si_hit_y_entry_mm", &event_data.si_hit_y_entry_mm);
  event_tree->Branch("si_hit_z_entry_mm", &event_data.si_hit_z_entry_mm);

  event_tree->Branch("si_hit_x_edep_mm", &event_data.si_hit_x_edep_mm);
  event_tree->Branch("si_hit_y_edep_mm", &event_data.si_hit_y_edep_mm);
  event_tree->Branch("si_hit_z_edep_mm", &event_data.si_hit_z_edep_mm);

  event_tree->Branch("labr3_detector_id", &event_data.labr3_detector_id);
  event_tree->Branch("labr3_edep_MeV", &event_data.labr3_edep_MeV);
  event_tree->Branch("labr3_time_ns", &event_data.labr3_time_ns);

  event_tree->Branch("hpge_detector_id", &event_data.hpge_detector_id);
  event_tree->Branch("hpge_edep_MeV", &event_data.hpge_edep_MeV);
  event_tree->Branch("hpge_time_ns", &event_data.hpge_time_ns);
}

void RootIO::FillReactionTree(H11BReactionData& data)
{
  if (!reaction_tree) return;
  reaction_data = data;
  reaction_tree->Fill();
}

void RootIO::FillVirtualSphereTree(VirtualSphereData& data)
{
  if (!virtual_sphere_tree) return;
  virtual_sphere_data = data;
  virtual_sphere_tree->Fill();
}

void RootIO::FillEventTree(EventData& data)
{
  if (!event_tree) return;
  event_data = data;
  event_tree->Fill();
}

void RootIO::CloseDataFile()
{
  if (!data_file) return;

  data_file->cd();
  // Explicit write order requested by the project data model.
  if (si_pixel_map_tree) si_pixel_map_tree->Write();
  if (reaction_tree) reaction_tree->Write();
  if (virtual_sphere_tree) virtual_sphere_tree->Write();
  if (event_tree) event_tree->Write();
  if (run_info_tree) run_info_tree->Write();

  data_file->Close();
  delete data_file;
  data_file = nullptr;
  si_pixel_map_tree = nullptr;
  reaction_tree = nullptr;
  virtual_sphere_tree = nullptr;
  event_tree = nullptr;
  run_info_tree = nullptr;
  G4cout << "\n----> unified ROOT output saved.\n\n";
}
