// Read-only numerical driver for the SAME amplitude/sampler used by Geant4.
// Output is streamed to Python/Jupyter; no files are written by this program.
#include "H11B162BoseModel.hh"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

int main(int argc,char** argv)
{
  using namespace H11B162Bose;
  try {
    std::cout << std::setprecision(17);
    if(argc<2) throw std::invalid_argument("Use sample mode N seed Q bins; weights Q bins; angular; radial Q bins");
    const std::string command=argv[1];
    constexpr double mass=3727.3794066;
    if(command=="sample") {
      if(argc!=7) throw std::invalid_argument("sample requires mode N seed Q bins");
      const auto mode=static_cast<Mode>(std::stoi(argv[2]));
      const int count=std::stoi(argv[3]);
      if(count<1 || count>1000000) throw std::invalid_argument("N must be 1..1000000");
      std::mt19937_64 engine(std::stoull(argv[4]));
      auto uniform=[&] { return std::generate_canonical<double,53>(engine); };
      const RadialTable radial(std::stod(argv[5]),mass,{},std::stoi(argv[6]));
      for(int i=0;i<count;++i) {
        const auto sample=Generate(radial,mode,uniform);
        for(const auto& v:sample.event) std::cout << v.p[0] << ' ' << v.p[1] << ' ' << v.p[2] << ' ' << v.e-mass << ' ';
        const auto& w=sample.weights;
        std::cout << w.single << ' ' << w.incoherent << ' ' << w.coherent << ' ' << w.acceptance << ' ' << sample.attempts << '\n';
      }
    } else if(command=="weights") {
      if(argc!=4) throw std::invalid_argument("weights requires Q bins; stdin: px py pz kinetic for each alpha");
      const RadialTable radial(std::stod(argv[2]),mass,{},std::stoi(argv[3]));
      Event event;
      while(std::cin >> event[0].p[0]) {
        std::cin >> event[0].p[1] >> event[0].p[2] >> event[0].e;
        for(int i=1;i<3;++i) std::cin >> event[i].p[0] >> event[i].p[1] >> event[i].p[2] >> event[i].e;
        if(!std::cin) throw std::invalid_argument("Incomplete event");
        for(auto& v:event) v.e+=mass;
        const auto w=Evaluate(event,radial);
        std::cout << w.single << ' ' << w.incoherent << ' ' << w.coherent << ' ' << w.acceptance << '\n';
      }
    } else if(command=="angular") {
      Vector p,s;
      while(std::cin >> p[0] >> p[1] >> p[2] >> s[0] >> s[1] >> s[2]) {
        const auto values=AngularAmplitude(p,s);
        for(const auto& value:values) std::cout << value.real() << ' ' << value.imag() << ' ';
        std::cout << '\n';
      }
    } else if(command=="radial") {
      if(argc!=4) throw std::invalid_argument("radial requires Q bins; stdin: pair energy");
      const RadialTable radial(std::stod(argv[2]),mass,{},std::stoi(argv[3]));
      double energy;
      while(std::cin >> energy)
        std::cout << radial.Density(energy) << ' ' << radial.RawDensity(energy) << ' ' << radial.Width(energy) << '\n';
    } else if(command=="exhaustion-check") {
      // A deterministic sequence must encounter a rejected one-attempt proposal.
      const RadialTable radial(8.8314,mass);
      std::mt19937_64 engine(16219);
      auto uniform=[&] { return std::generate_canonical<double,53>(engine); };
      bool rejected=false;
      for(int i=0;i<100 && !rejected;++i) {
        try { (void)Generate(radial,Mode::Coherent,uniform,1); }
        catch(const std::runtime_error& error) {
          if(std::string(error.what()).find("rejection exhausted")==std::string::npos) throw;
          rejected=true;
        }
      }
      if(!rejected) throw std::runtime_error("Exhaustion check did not reach failure path");
      std::cout << "PASS: rejection exhaustion raises instead of returning a fallback event\n";
    } else throw std::invalid_argument("Unknown command");
  } catch(const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
