// Export unrestricted electronic and nuclear alpha stopping powers for the
// literature-response sensitivity study. This is not the authors' SRIM table.
#include "G4Alpha.hh"
#include "G4Box.hh"
#include "G4EmCalculator.hh"
#include "G4EmParameters.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4Version.hh"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
const std::array<const char*, 3> materials = {"G4_B", "G4_C", "G4_Si"};

class Geometry final : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    auto* world = new G4LogicalVolume(new G4Box("world", 10*cm, 10*cm, 10*cm),
                                      nist->FindOrBuildMaterial("G4_Galactic"), "world");
    auto* placement = new G4PVPlacement(nullptr, {}, world, "world", nullptr, false, 0);
    for (std::size_t i = 0; i < materials.size(); ++i) {
      auto* volume = new G4LogicalVolume(new G4Box(materials[i], mm, mm, mm),
          nist->FindOrBuildMaterial(materials[i]), materials[i]);
      new G4PVPlacement(nullptr, {G4double(i)*3*mm, 0, 0}, volume,
                        materials[i], world, false, G4int(i));
    }
    return placement;
  }
};

class Physics final : public G4VModularPhysicsList {
 public:
  Physics() { SetVerboseLevel(0); RegisterPhysics(new G4EmStandardPhysics_option4(0)); }
  void SetCuts() override { SetCutsWithDefault(); }
};
}

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("Usage: export_162_stopping_power OUTPUT.json");
    if (std::filesystem::exists(argv[1])) throw std::runtime_error("Preserve existing output");
    auto manager = std::make_unique<G4RunManager>();
    manager->SetVerboseLevel(0);
    manager->SetUserInitialization(new Geometry);
    manager->SetUserInitialization(new Physics);
    G4EmParameters::Instance()->SetMinEnergy(100*eV);
    G4EmParameters::Instance()->SetMaxEnergy(1*GeV);
    G4EmParameters::Instance()->SetVerbose(0);
    manager->Initialize();
    // Physics tables are built at run initialization, before calculator access.
    // No primary or event loop is needed for this material-table export.
    manager->RunInitialization();
    G4EmCalculator calculator;
    std::ofstream output(argv[1]);
    if (!output) throw std::runtime_error("Cannot open output");
    output << std::setprecision(17)
           << "{\"geant4\":\"" << G4Version << "\",\"physics\":\"G4EmStandardPhysics_option4\","
           << "\"particle\":\"alpha\",\"energy_unit\":\"MeV\","
           << "\"stopping_unit\":\"MeV cm2/g\",\"materials\":{";
    for (std::size_t m = 0; m < materials.size(); ++m) {
      auto* material = G4NistManager::Instance()->FindOrBuildMaterial(materials[m]);
      if (m) output << ',';
      output << '"' << materials[m] << "\":{\"density_g_cm3\":"
             << material->GetDensity()/(g/cm3) << ",\"rows\":[";
      for (int i = 0; i <= 2400; ++i) {
        const double energy = .001*std::pow(10000., i/2400.);
        const double electronic = calculator.ComputeElectronicDEDX(energy*MeV, G4Alpha::Alpha(), material);
        const double nuclear = calculator.ComputeNuclearDEDX(energy*MeV, G4Alpha::Alpha(), material);
        const double total = calculator.ComputeTotalDEDX(energy*MeV, G4Alpha::Alpha(), material);
        if (!std::isfinite(total) || electronic <= 0 || nuclear < 0 ||
            std::abs(electronic+nuclear-total) > 1e-9*total)
          throw std::runtime_error("Invalid stopping power or decomposition");
        const double scale = 1/(material->GetDensity()*(MeV*cm2/g));
        if (i) output << ',';
        output << '[' << energy << ',' << electronic*scale << ',' << nuclear*scale << ']';
      }
      output << "]}";
    }
    output << "}}\n";
    output.close();
    if (!output) throw std::runtime_error("Output write failed");
    std::cout << "Stopping-power export complete: " << argv[1] << '\n';
    manager->RunTermination();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
