#include "H11B162BoseModel.hh"
#include "CoulombPenetrability.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace H11B162Bose {
namespace {
constexpr double pi = 3.14159265358979323846;
double Dot(const Vector& a, const Vector& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
Vector Scale(const Vector& a, double s) { return {a[0]*s, a[1]*s, a[2]*s}; }
Vector Add(const Vector& a, const Vector& b) { return {a[0]+b[0], a[1]+b[1], a[2]+b[2]}; }
Vector Cross(const Vector& a, const Vector& b)
{ return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]}; }
Vector Unit(const Vector& a)
{
  const double n = std::sqrt(Dot(a,a));
  if (!(n > 0.0) || !std::isfinite(n)) throw std::runtime_error("Invalid angular direction");
  return Scale(a, 1.0/n);
}
double Mass(const FourVector& a) { return std::sqrt(std::max(0.0, a.e*a.e-Dot(a.p,a.p))); }
double PrimaryMomentum(double total, double mass, double pair)
{
  return std::sqrt(std::max(0.0, (total-mass-pair)*(total+mass+pair)
                           *(total-mass+pair)*(total+mass-pair)))/(2*total);
}
double Factorial(int n)
{
  double out=1;
  for (int i=2;i<=n;++i) out*=i;
  return out;
}
double CG222(int m1,int m2,int m)
{
  if (m1+m2!=m) return 0.0;
  const double prefactor=std::sqrt(5*Factorial(2)*Factorial(2)*Factorial(2)/Factorial(7)
      *Factorial(2+m1)*Factorial(2-m1)*Factorial(2+m2)*Factorial(2-m2)
      *Factorial(2+m)*Factorial(2-m));
  double sum=0;
  for(int k=0;k<=6;++k) {
    const std::array<int,6> args={k,2-k,2-m1-k,2+m2-k,m1+k,-m2+k};
    if (*std::min_element(args.begin(),args.end())<0) continue;
    double denom=1;
    for(auto a:args) denom*=Factorial(a);
    sum+=(k%2 ? -1.0 : 1.0)/denom;
  }
  return prefactor*sum;
}
std::array<std::complex<double>,5> Y2(const Vector& direction)
{
  const auto u=Unit(direction);
  const std::complex<double> xy(u[0],u[1]);
  const auto y1=-std::sqrt(15/(8*pi))*u[2]*xy;
  const auto y2=std::sqrt(15/(32*pi))*xy*xy;
  return {std::conj(y2),-std::conj(y1),std::sqrt(5/(16*pi))*(3*u[2]*u[2]-1),y1,y2};
}
Vector Isotropic(const std::function<double()>& uniform)
{
  const double x=2*uniform()-1, phi=2*pi*uniform();
  const double r=std::sqrt(std::max(0.0,1-x*x));
  return {r*std::cos(phi),r*std::sin(phi),x};
}
Vector Correlated(const Vector& axis,const std::function<double()>& uniform)
{
  // max p_L2(x)=5/7; 0.25*(9*x^4-9*x^2+4) is its exact acceptance ratio.
  double x=0;
  bool accepted=false;
  for(int i=0;i<10000;++i) {
    x=2*uniform()-1;
    if(uniform() <= (9*x*x*x*x-9*x*x+4)/4) { accepted=true; break; }
  }
  if(!accepted) throw std::runtime_error("L2 angular rejection exhausted");
  const auto z=Unit(axis);
  const auto ex=Unit(Cross(std::abs(z[2])<0.9 ? Vector{0,0,1} : Vector{1,0,0},z));
  const auto ey=Cross(z,ex);
  const double phi=2*pi*uniform(), r=std::sqrt(std::max(0.0,1-x*x));
  return Add(Scale(z,x),Add(Scale(ex,r*std::cos(phi)),Scale(ey,r*std::sin(phi))));
}
} // namespace

RadialTable::RadialTable(double energy,double mass,Parameters params,int count)
    : available_energy(energy),alpha_mass(mass),parameters(params),bins(count),step(0)
{
  if (!std::isfinite(energy) || !(energy>params.pole) || !std::isfinite(mass) || !(mass>0)
      || !std::isfinite(params.pole) || !(params.pole>0) || !std::isfinite(params.width)
      || !(params.width>0) || count<64 || count>65536)
    throw std::invalid_argument("Invalid 162 Bose radial table parameters");
  step=energy/bins;
  density.resize(bins+1); cdf.resize(bins+1);
  for(int i=0;i<=bins;++i) {
    density[i]=RawDensity(i*step);
    if (!std::isfinite(density[i]) || density[i]<0) throw std::runtime_error("Invalid radial density");
    if(i) cdf[i]=cdf[i-1]+step*(density[i-1]+density[i])/2;
  }
  if(!(cdf.back()>0)) throw std::runtime_error("Empty radial probability");
}

double RadialTable::Width(double eaa) const
{
  if(!(eaa>0)) return 0;
  const double ratio=parameters.full_pair_penetrability
      ? CoulombPenetrability::AlphaAlphaL2Ratio(eaa*1000,parameters.pole*1000)
      : std::pow(eaa/parameters.pole,2.5);
  return parameters.width*ratio;
}

double RadialTable::RawDensity(double eaa) const
{
  if(!(eaa>0 && eaa<available_energy)) return 0;
  const double energy=available_energy-eaa, reference=available_energy-parameters.pole;
  const double first=parameters.full_primary_penetrability
      ? CoulombPenetrability::Alpha8BeRatio(energy*1000,reference*1000,2)
      : std::pow(energy/reference,2.5);
  const double width=Width(eaa), delta=parameters.pole-eaa;
  return first*width/(delta*delta+width*width/4);
}

double RadialTable::Density(double eaa) const
{
  if(!(eaa>0 && eaa<available_energy)) return 0;
  const double location=eaa/step;
  const int i=std::min(bins-1,static_cast<int>(location));
  return density[i]+(location-i)*(density[i+1]-density[i]);
}

double RadialTable::SampleEnergy(double uniform) const
{
  if(!std::isfinite(uniform) || uniform<0 || uniform>=1)
    throw std::invalid_argument("Uniform variate must lie in [0,1)");
  // The discrete RNG may return exactly zero. Move only that endpoint to a
  // half-ulp probability, avoiding unresolvable zero relative momentum.
  if(uniform==0) uniform=std::numeric_limits<double>::epsilon()/2;
  const double target=uniform*cdf.back();
  const int i=std::min(bins-1,static_cast<int>(std::upper_bound(cdf.begin(),cdf.end(),target)-cdf.begin())-1);
  const double area=target-cdf[i], slope=(density[i+1]-density[i])/step;
  const double discriminant=std::max(0.0,density[i]*density[i]+2*slope*area);
  const double denominator=density[i]+std::sqrt(discriminant);
  const double dx=denominator>0 ? 2*area/denominator : 0;
  return std::clamp(i*step+dx,std::nextafter(0.0,1.0),std::nextafter(available_energy,0.0));
}

FourVector Boost(const FourVector& v,const Vector& beta)
{
  const double b2=Dot(beta,beta);
  if(!std::isfinite(b2) || b2>=1) throw std::invalid_argument("Invalid boost");
  if(b2==0) return v;
  const double gamma=1/std::sqrt(1-b2), bp=Dot(beta,v.p);
  // Active boost to a frame in which the parent's velocity is +beta.
  return {Add(v.p,Scale(beta,((gamma-1)/b2)*bp+gamma*v.e)),gamma*(v.e+bp)};
}
Vector BoostMomentum(const FourVector& v,const Vector& beta) { return Boost(v,beta).p; }

std::array<std::complex<double>,5> AngularAmplitude(const Vector& primary,const Vector& secondary)
{
  const auto yp=Y2(primary),ys=Y2(secondary);
  static const auto cg=[] {
    std::array<std::array<double,5>,5> out{};
    for(int m=-2;m<=2;++m) for(int ms=-2;ms<=2;++ms)
      if(std::abs(m-ms)<=2) out[m+2][ms+2]=CG222(m-ms,ms,m);
    return out;
  }();
  std::array<std::complex<double>,5> out{};
  for(int m=-2;m<=2;++m) for(int ms=-2;ms<=2;++ms) {
    const int mp=m-ms;
    if(std::abs(mp)<=2) out[m+2]+=cg[m+2][ms+2]*yp[mp+2]*ys[ms+2];
  }
  return out;
}

Weights Evaluate(const Event& event,const RadialTable& radial)
{
  const double mass=radial.alpha_mass, total=3*mass+radial.available_energy;
  std::array<std::complex<double>,5> sum{};
  Weights out;
  for(int i=0;i<3;++i) {
    const int j=(i+1)%3,k=(i+2)%3;
    const FourVector pair{Add(event[j].p,event[k].p),event[j].e+event[k].e};
    const double pair_mass=Mass(pair), eaa=pair_mass-2*mass;
    const double density=radial.Density(eaa);
    if(!(density>0)) continue;
    const double p=PrimaryMomentum(total,mass,pair_mass);
    const double q=std::sqrt(std::max(0.0,eaa*(4*mass+eaa)))/2;
    if(!(p*q>0)) continue;
    const auto secondary=BoostMomentum(event[j],Scale(pair.p,-1/pair.e));
    const auto angular=AngularAmplitude(event[i].p,secondary);
    const double delta=radial.parameters.pole-eaa, half_width=radial.Width(eaa)/2;
    // |R|^2 = tabulated BW density / (p*q); 1/(delta-i*Gamma/2) supplies the phase.
    // The p*q factor removes the phase-space Jacobian already in the proposal.
    const auto phase=std::complex<double>(delta,half_width)/std::hypot(delta,half_width);
    const auto factor=std::sqrt(density/(p*q))*phase;
    double diagonal=0;
    for(int m=0;m<5;++m) {
      const auto amplitude=factor*angular[m];
      sum[m]+=amplitude;
      diagonal+=std::norm(amplitude)/5;
    }
    out.incoherent+=diagonal;
    if(i==0) out.single=diagonal;
  }
  for(const auto& value:sum) out.coherent+=std::norm(value)/5;
  if(!(out.incoherent>0) || !std::isfinite(out.incoherent) || !std::isfinite(out.coherent))
    throw std::runtime_error("Nonfinite or zero amplitude in proposal support");
  out.acceptance=out.coherent/(3*out.incoherent);
  if(out.acceptance<0 || out.acceptance>1+1e-12 || !std::isfinite(out.acceptance))
    throw std::runtime_error("Bose Cauchy bound violated; abort instead of clipping weights");
  out.acceptance=std::min(1.0,out.acceptance); // only possible roundoff <= 1e-12
  return out;
}

Sample Generate(const RadialTable& radial,Mode mode,const std::function<double()>& uniform,int max_attempts)
{
  if((mode!=Mode::SinglePath && mode!=Mode::Incoherent && mode!=Mode::Coherent) || max_attempts<1)
    throw std::invalid_argument("Invalid model mode or sampling attempt limit");
  const double mass=radial.alpha_mass,total=3*mass+radial.available_energy;
  const std::array<std::array<int,3>,6> permutations={{{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}}};
  for(int attempt=1;attempt<=max_attempts;++attempt) {
    const double eaa=radial.SampleEnergy(uniform()),pair_mass=2*mass+eaa;
    const double p=PrimaryMomentum(total,mass,pair_mass),q=std::sqrt(eaa*(4*mass+eaa))/2;
    const auto direction=Isotropic(uniform);
    const FourVector first{Scale(direction,p),std::sqrt(mass*mass+p*p)};
    const FourVector pair{Scale(first.p,-1),std::sqrt(pair_mass*pair_mass+p*p)};
    const auto beta=Scale(pair.p,1/pair.e);
    const auto secondary=Scale(Correlated(direction,uniform),q);
    Event event={first,Boost({secondary,pair_mass/2},beta),Boost({Scale(secondary,-1),pair_mass/2},beta)};
    if(mode!=Mode::SinglePath) {
      const auto perm=permutations.at(static_cast<int>(6*uniform()));
      event={event[perm[0]],event[perm[1]],event[perm[2]]};
    }
    const auto weights=Evaluate(event,radial);
    // A uniform mixture of the three diagonal path densities has density
    // I_inc/(3 Z). Cauchy gives I_coh <= 3 I_inc, hence no scanned envelope.
    if(mode!=Mode::Coherent || uniform()<=weights.acceptance)
      return {event,weights,attempt};
  }
  throw std::runtime_error("162 Bose rejection exhausted; no best-candidate fallback is permitted");
}
} // namespace H11B162Bose
