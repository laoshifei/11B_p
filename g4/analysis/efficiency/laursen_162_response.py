"""Conditional response controls for Laursen 2016, data set 9.

Published geometry and material dimensions are separated from assumed response
parameters. This is an analysis-layer CSDA calculation, not the authors' Monte
Carlo, electronics, or an experimental fit. Lengths: mm; energy: MeV;
material columns: g/cm2. Importing this module performs no calculation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import itertools

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator

from analysis.bayes.rmatrix_162_model import MASS, boost

CENTRES = np.array([[19.4, 0., -4.5], [-23.5, -.5, 4.9]])
BETAS = np.deg2rad([105., 285.])
NORMALS = np.column_stack([np.sin(BETAS), np.zeros(2), np.cos(BETAS)])
HORIZONTAL = np.column_stack([np.cos(BETAS), np.zeros(2), -np.sin(BETAS)])
VERTICAL = np.array([0., 1., 0.])
FOIL_NORMAL = np.array([np.sin(np.deg2rad(126.)), 0., np.cos(np.deg2rad(126.))])
HALF_SIZE, PITCH = 25., 50./16
PERMUTATIONS = tuple(itertools.permutations(range(3)))


@dataclass(frozen=True)
class Response:
    name: str = 'nominal_assumptions'
    boron_ug_cm2: float = 12.5
    adsorption_exit_ug_cm2: float = 10.
    backing_ug_cm2: float = 4.
    backing_side: int = 1
    dead_um: tuple = (.2, .7)
    adc_MeV: tuple = (.055, .150)
    trigger50_MeV: tuple = (.200, .350)
    trigger_width_MeV: float = .300
    trigger_mode: str = 'per_particle'
    fwhm_MeV: float = .040
    pixelise: bool = True
    kinematic_cuts: bool = True
    ambiguous_veto: bool = False
    radial_shift_mm: float = 0.
    geometry_only: bool = False

    def __post_init__(self):
        if self.trigger_mode not in ('none', 'per_particle', 'global_or'):
            raise ValueError('Unknown trigger assumption')
        if self.backing_side not in (-1, 1) or self.trigger_width_MeV <= 0:
            raise ValueError('Invalid backing side or trigger width')
        if min(self.boron_ug_cm2, self.adsorption_exit_ug_cm2, self.backing_ug_cm2,
               self.fwhm_MeV, *self.dead_um, *self.adc_MeV, *self.trigger50_MeV) < 0:
            raise ValueError('Negative material or response parameter')


def scenarios():
    base = Response()
    return [replace(base, name='geometry', geometry_only=True, kinematic_cuts=False),
            replace(base, name='ideal_readout', boron_ug_cm2=0, adsorption_exit_ug_cm2=0,
                    backing_ug_cm2=0, dead_um=(0., 0.), adc_MeV=(0., 0.), trigger_mode='none'),
            replace(base, name='material_adc', trigger_mode='none'), base,
            replace(base, name='low_threshold', adc_MeV=(.010,.100), trigger50_MeV=(.100,.200)),
            replace(base, name='high_threshold', adc_MeV=(.100,.200), trigger50_MeV=(.300,.500)),
            replace(base, name='trigger_width_200', trigger_width_MeV=.200),
            replace(base, name='trigger_width_400', trigger_width_MeV=.400),
            replace(base, name='boron_10', boron_ug_cm2=10.),
            replace(base, name='boron_15', boron_ug_cm2=15.),
            replace(base, name='carbon_5', adsorption_exit_ug_cm2=5.),
            replace(base, name='carbon_15', adsorption_exit_ug_cm2=15.),
            replace(base, name='no_adsorption', adsorption_exit_ug_cm2=0.),
            replace(base, name='backing_reversed', backing_side=-1),
            replace(base, name='global_trigger', trigger_mode='global_or'),
            replace(base, name='strip_ambiguity_veto', ambiguous_veto=True),
            replace(base, name='detectors_out_2mm', radial_shift_mm=2.),
            replace(base, name='detectors_in_2mm', radial_shift_mm=-2.)]


def unit(vectors):
    vectors = np.asarray(vectors, float)
    length = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, length, out=np.zeros_like(vectors), where=length > 0)


def trace(directions, origins, *, radial_shift=0.):
    """First positive intersection with either finite active rectangle."""
    directions, origins = np.asarray(directions, float), np.asarray(origins, float)
    if directions.ndim != 3 or directions.shape[1:] != (3,3) or origins.shape != (len(directions),3):
        raise ValueError('Expected N x 3 particles and N origins')
    if not np.isfinite(directions).all() or not np.allclose(np.linalg.norm(directions,axis=-1),1):
        raise ValueError('Directions must be finite unit vectors')
    shape = directions.shape[:2]
    distance, detector = np.full(shape,np.inf), np.full(shape,-1,int)
    pixel = np.full((*shape,2),-1,int)
    reconstructed = directions.copy()
    centres = CENTRES+radial_shift*unit(CENTRES)
    for d,(centre,normal,horizontal) in enumerate(zip(centres,NORMALS,HORIZONTAL)):
        denominator = directions@normal
        numerator = (centre-origins)@normal
        t = np.divide(numerator[:,None],denominator,out=np.full(shape,np.inf),where=abs(denominator)>1e-14)
        relative = origins[:,None,:]+np.where(np.isfinite(t),t,0.)[:,:,None]*directions-centre
        x,y = relative@horizontal, relative@VERTICAL
        hit = (t>0)&(t<distance)&(x>=-HALF_SIZE)&(x<HALF_SIZE)&(y>=-HALF_SIZE)&(y<HALF_SIZE)
        ix,iy = np.floor((x+HALF_SIZE)/PITCH).astype(int),np.floor((y+HALF_SIZE)/PITCH).astype(int)
        position = centre+( (ix+.5)*PITCH-HALF_SIZE)[:,:,None]*horizontal
        position += ((iy+.5)*PITCH-HALF_SIZE)[:,:,None]*VERTICAL
        distance[hit],detector[hit] = t[hit],d
        pixel[hit] = np.stack([ix,iy],axis=-1)[hit]
        # Reconstruction assumes the nominal beam-spot centre, not the true origin.
        reconstructed[hit] = unit(position)[hit]
    return dict(detector=detector,pixel=pixel,direction=reconstructed,distance=distance)


def rectangle_solid_angle(centre, normal, horizontal, origin=np.zeros(3)):
    """Independent analytic solid angle for a rectangle seen from a point."""
    relative = np.asarray(centre)-origin
    z = abs(relative@normal)
    x0,y0 = relative@horizontal,relative@VERTICAL
    result=0.
    for sx in (-1,1):
        for sy in (-1,1):
            x,y=x0+sx*HALF_SIZE,y0+sy*HALF_SIZE
            result += sx*sy*np.arctan2(x*y,z*np.sqrt(x*x+y*y+z*z))
    return float(abs(result))


class RangeTables:
    """CSDA from unrestricted Geant4 electronic+nuclear stopping powers.

    The 0--1 keV segment uses the 1-keV value. It is a declared numerical
    continuation, not validated very-low-energy stopping data. No straggling
    or scattering; silicon visible energy integrates electronic/total loss.
    """
    def __init__(self, table, stride=1):
        self.table, self.materials = table, {}
        for name, record in table['materials'].items():
            rows=np.asarray(record['rows'],float)[::stride]
            if (rows.ndim!=2 or rows.shape[1]!=3 or not np.isfinite(rows).all()
                    or np.any(rows[:,0]<=0) or np.any(np.diff(rows[:,0])<=0)
                    or np.any(rows[:,1]<=0) or np.any(rows[:,2]<0)):
                raise ValueError('Invalid stopping-power table')
            energy=np.r_[0.,rows[:,0]]
            total=np.r_[rows[0,1:].sum(),rows[:,1:].sum(axis=1)]
            electronic=np.r_[rows[0,1],rows[:,1]]
            ranges=cumulative_trapezoid(1/total,energy,initial=0)
            visible=cumulative_trapezoid(electronic/total,energy,initial=0)
            self.materials[name]=dict(energy=energy,range=ranges,visible=visible,
                forward=PchipInterpolator(energy,ranges,extrapolate=False),
                inverse=PchipInterpolator(ranges,energy,extrapolate=False),
                light=PchipInterpolator(energy,visible,extrapolate=False),
                inverse_light=PchipInterpolator(visible,energy,extrapolate=False))

    def remove(self, energy, column, material):
        if np.any(np.asarray(column)<0) or np.any(np.asarray(energy)<0):
            raise ValueError('Negative energy or mass column')
        row=self.materials[material]
        result=row['inverse'](np.maximum(0.,row['forward'](energy)-column))
        if not np.isfinite(result).all():raise ValueError('Energy outside stopping domain')
        return result

    def restore(self, energy, column, material):
        if np.any(np.asarray(column)<0):raise ValueError('Negative mass column')
        row=self.materials[material]
        # Invalid inversions stay NaN and are explicitly rejected, never clamped.
        return row['inverse'](row['forward'](energy)+column)


def columns(directions, detector, depth, response, tables):
    foil_dot=directions@FOIL_NORMAL
    foil_cos=np.maximum(abs(foil_dot),1e-14)
    boron=response.boron_ug_cm2*1e-6*np.where(foil_dot>=0,depth[:,None],1-depth[:,None])/foil_cos
    carbon=(response.adsorption_exit_ug_cm2+
            response.backing_ug_cm2*(response.backing_side*foil_dot>0))*1e-6/foil_cos
    d=np.maximum(0,detector)
    incidence=np.maximum(abs(np.sum(directions*NORMALS[d],axis=-1)),1e-14)
    silicon=np.asarray(response.dead_um)[d]*1e-4*tables.table['materials']['G4_Si']['density_g_cm3']/incidence
    return tuple(np.where(detector>=0,x,0.) for x in (boron,carbon,silicon))


def trigger_efficiency(energy, midpoint, width):
    if width<=0:raise ValueError('Trigger width must be positive')
    return np.clip(.5+(np.asarray(energy)-midpoint)/width,0.,1.)


def hit_pattern_probabilities(adc, trigger, mode):
    adc=np.asarray(adc,bool);trigger=np.asarray(trigger,float)
    if adc.shape != trigger.shape or adc.ndim!=2 or adc.shape[1]!=3 or np.any((trigger<0)|(trigger>1)):
        raise ValueError('Invalid hit probabilities')
    if mode not in ('none','per_particle','global_or'):raise ValueError('Unknown trigger model')
    p=adc*trigger if mode=='per_particle' else adc.astype(float)
    rows=[]
    for bit in range(8):
        pattern=np.array([(bit>>i)&1 for i in range(3)],bool)
        value=np.prod(np.where(pattern,p,1-p),axis=1)
        if mode=='global_or':value*=1-np.prod(1-trigger,axis=1)
        rows.append(value)
    result=np.stack(rows,axis=1)
    if mode=='global_or':result[:,0]+=np.prod(1-trigger,axis=1)
    if not np.allclose(result.sum(axis=1),1.,atol=1e-13):raise ArithmeticError('Lost event probability')
    return result


def reconstruct(energy, directions, parent_energy, parent_momentum, pattern):
    """Reconstruct a missing alpha by momentum conservation, not by energy closure."""
    momentum=np.sqrt(energy*(energy+2*MASS))[...,None]*directions
    energy=energy.copy()+MASS
    missing=np.flatnonzero(~pattern)
    if len(missing)==1:
        j=missing[0]
        momentum[:,j]=parent_momentum-momentum[:,pattern].sum(axis=1)
        energy[:,j]=np.sqrt(MASS*MASS+np.sum(momentum[:,j]**2,axis=1))
    elif len(missing):raise ValueError('Only multiplicity two or three is reconstructable')
    beta=-parent_momentum/parent_energy[:,None]
    ecm,pcm=boost(energy,momentum,beta[:,None,:])
    kinetic=np.maximum(ecm-MASS,0.)
    return kinetic,pcm


def triple_quality(momentum):
    direction=unit(momentum)
    angles=sum(np.arccos(np.clip(np.sum(direction[:,i]*direction[:,j],axis=1),-1,1))
               for i,j in ((0,1),(1,2),(2,0)))
    normal=unit(np.cross(direction[:,0],direction[:,1]))
    plane=np.arcsin(np.clip(abs(np.sum(normal*direction[:,2],axis=1)),0,1))
    return (np.linalg.norm(momentum.sum(axis=1),axis=1)<50.) & (abs(angles-2*np.pi)<=np.deg2rad(10.)) & (plane<=np.deg2rad(10.))


def ambiguous_pattern(geometry,pattern):
    ambiguous=np.zeros(len(geometry['detector']),bool)
    for i,j in itertools.combinations(np.flatnonzero(pattern),2):
        same=(geometry['detector'][:,i]==geometry['detector'][:,j])&(geometry['detector'][:,i]>=0)
        ambiguous |= same & np.any(geometry['pixel'][:,i]==geometry['pixel'][:,j],axis=1)
    return ambiguous


def fold(kinetic, directions, origins, depth, normal_noise, response, tables):
    """Return four reconstructable hit-pattern probabilities and observables."""
    geometry=trace(directions,origins,radial_shift=response.radial_shift_mm)
    detector=geometry['detector'];hit=detector>=0
    parent_momentum=(np.sqrt(kinetic*(kinetic+2*MASS))[...,None]*directions).sum(axis=1)
    parent_energy=(kinetic+MASS).sum(axis=1)
    if response.geometry_only:
        corrected=kinetic.copy();rec_direction=directions
        probabilities=hit_pattern_probabilities(hit,np.ones_like(kinetic),'none')
        invalid=np.zeros_like(hit)
    else:
        b,c,si=columns(directions,detector,depth,response,tables)
        entering=tables.remove(tables.remove(tables.remove(kinetic,b,'G4_B'),c,'G4_C'),si,'G4_Si')
        light=tables.materials['G4_Si']['light'](entering)
        measured=np.maximum(0.,light+response.fwhm_MeV/2.354820045*normal_noise)
        adc=hit&(entering>0)&(measured>=np.asarray(response.adc_MeV)[np.maximum(detector,0)])
        trigger=hit*(entering>0)*trigger_efficiency(measured,np.asarray(response.trigger50_MeV)[np.maximum(detector,0)],response.trigger_width_MeV)
        probabilities=hit_pattern_probabilities(adc,trigger,response.trigger_mode)
        rec_direction=geometry['direction'] if response.pixelise else directions
        b,c,si=columns(rec_direction,detector,np.full(len(depth),.5),response,tables)
        recovered=tables.materials['G4_Si']['inverse_light'](measured)
        corrected=tables.restore(tables.restore(tables.restore(recovered,si,'G4_Si'),c,'G4_C'),b,'G4_B')
        invalid=hit&~np.isfinite(corrected)
        corrected=np.nan_to_num(corrected,nan=0.)
    records=[]
    for bit in (3,5,6,7):
        pattern=np.array([(bit>>i)&1 for i in range(3)],bool)
        probability=probabilities[:,bit].copy()
        raw_probability=probability.copy()
        measured,pcm=reconstruct(corrected,rec_direction,parent_energy,parent_momentum,pattern)
        valid=~np.any(invalid[:,pattern],axis=1) & (measured.sum(axis=1)>0)
        if bit==7 and response.kinematic_cuts:valid &= triple_quality(pcm)
        ambiguity=ambiguous_pattern(geometry,pattern)
        if response.ambiguous_veto:valid &= ~ambiguity
        probability*=valid
        records.append(dict(multiplicity=int(pattern.sum()),pattern=pattern,probability=probability,
            preselection=raw_probability,ambiguous=ambiguity,cm_kinetic=measured,detector=detector))
    return records,dict(invalid_corrections=int(np.any(invalid,axis=1).sum()),geometry=geometry)


def symmetric_dalitz(energy,weights,edges):
    fractions=energy/energy.sum(axis=1)[:,None]
    output=np.zeros((len(edges)-1,len(edges)-1))
    for permutation in PERMUTATIONS:
        a,b,c=fractions[:,permutation].T
        x,y=(a+2*b-1)/np.sqrt(3),a-1/3
        output+=np.histogram2d(x,y,bins=(edges,edges),weights=weights/6)[0]
    return output
