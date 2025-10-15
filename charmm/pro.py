"""
JOB DESCRIPTION

Production run of MthK system
embedded in POPC membrane, ions and water.
With Gate restraints. PME, Constant Pressure, Temperature.

"""

import sys
from sys import stdout, exit, stderr

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

import numpy as np

#############################################################
## ADJUSTABLE PARAMETERS                                   ##
#############################################################

# Load CHARMM files
print("Loading parameters")
psf = CharmmPsfFile('mthk_sys_charmm.psf')  # psf file
pdb = PDBFile('mthk_sys_charmm.pdb')        # coordinate file
outputName = 'step4_pro'                # output name
saveXML = False                         # output a state or checkpoint file?

temp = 320                              # Temperature (K)

restartFile = ''                        # Restart file to continue a simulation 
                                        # in either xml or chk format

#############################################################
## SIMULATION PARAMETERS                                   ##
#############################################################

params = CharmmParameterSet('par_all36_prot.prm', 'top_all36_prot.rtf',
                            'par_all36_lipid.prm', 'top_all36_lipid.rtf',
                            'par_all36_carb.prm', 'top_all36_carb.rtf',
                            'par_all36_cgenff.prm', 'top_all36_cgenff.rtf',
                            'par_all36_na.prm', 'top_all36_na.rtf',
                            'toppar_water_ions_namd.str')          # Parameter files

psf.setBox(89.704*angstroms, 90.741*angstroms, 84.790*angstroms) # Box dimensions for PBC
                                                                 # If continuing from a chk or xml file 
                                                                 # the box is later updated but this call is needed
                                                                 # to set up the system, same is true for positions

# Build openmm system
system = psf.createSystem(params, nonbondedMethod=PME, 
                          nonbondedCutoff=1.2*nanometers, switchDistance=1.0*nanometers,
                          constraints=HBonds, ewaldErrorTolerance=0.0001)

# Constant Pressure Control (variable volume)
if True:
    system.addForce(MonteCarloMembraneBarostat(1.01325*bar, 0.0*bar*nanometers, 
                                               temp*kelvin, MonteCarloMembraneBarostat.XYIsotropic,
                                               MonteCarloMembraneBarostat.ZFree, 100))

if True:  # Turn on distance restraints
    # Gate distance restraints
    distance_restraint = CustomBondForce("(k/2)*(r-r0)^2")
    distance_restraint.setName('distance_restraint')
    distance_restraint.addGlobalParameter('k', 2.39 * kilocalorie_per_mole/angstrom**2)
    distance_restraint.addPerBondParameter('r0')
    # get atom indices for the distance restraints
    a = 0
    empty = []
    atom_dihedral_indices = np.array([[0,0,0,0]])
    atom_distance_indices = np.array([[0,0]])
    with open('dihed_and_distance_rest.inp') as f:
        for line in f:
            text = line.split()
            if 'harmonicWalls' in text:
                a += 1
            if 'atomNumbers' in text and a == 0:
                empty.append(int(text[3]))
                if len(empty) == 4:
                    atom_dihedral_indices = np.append(atom_dihedral_indices, [empty], axis=0)
                    empty.clear()
            if 'atomNumbers' in text and a == 1:
                empty.append(int(text[3]))
                if len(empty) == 2:
                    atom_distance_indices = np.append(atom_distance_indices, [empty], axis=0)
                    empty.clear()
        f.close()
    atom_dihedral_indices = atom_dihedral_indices - 1
    atom_distance_indices = atom_distance_indices - 1
    #print(atom_distance_indices)
    #print(psf.atom_list[atom_distance_indices[1][0]])
    # distance equilibrium postions
    distances = np.array([36, 36, 25.5, 25.5, 25.5, 25.5, 36.6, 36.6, 26, 26, 26, 26])
    # add distances
    for i,j in enumerate(atom_distance_indices[1:]):
        distance_restraint.addBond(*j, [distances[i % len(distances)] * angstrom])
    # add forces to the system
    system.addForce(distance_restraint)

if True: # Turn on COM restraints
    # the equilibrium centers
    x0, y0, z0 = (46.732 * angstrom, 46.459 * angstrom, 37.775 * angstrom)
    position_restraint = CustomCentroidBondForce(1, '(k/2)*((x1-x0)^2+(y1-y0)^2+(z1-z0)^2)')
    position_restraint.setName('position_restraint')
    position_restraint.addGlobalParameter('k', 2.39 * kilocalorie_per_mole/angstrom**2)
    position_restraint.addPerBondParameter('x0')
    position_restraint.addPerBondParameter('y0')
    position_restraint.addPerBondParameter('z0')
    # atom indices that define the COM
    particles = [1033, 2327, 3621, 4915]
    position_restraint.addGroup(particles)
    # add the COM
    position_restraint.addBond([0], [x0, y0, z0])
    # add the force to the system
    system.addForce(position_restraint)

for i, f in enumerate(system.getForces()):
    f.setForceGroup(i)

integrator = LangevinIntegrator(temp*kelvin, 1/picosecond, 0.002*picoseconds)                      # Constant Temperature Control
				                                                                   # Time-step (ps)

# Use GPU
platform = Platform.getPlatformByName('CUDA')
prop = {'Precision': 'mixed'}           # Precision mode can be single, mixed, or double
                                        # double give most accurate calculations, but
					# at a performance cost
# Build simulation context
simulation = Simulation(psf.topology, system, integrator, platform, prop)
simulation.context.setPositions(pdb.positions)

if 'xml' in restartFile:                     # Continue a job from a saved state
    simulation.loadState(restartFile)
if 'chk' in restartFile:                     # Continue a job from a checkpoint
    simulation.loadCheckpoint(restartFile)


#print("Parameters: ", dict(simulation.context.getParameters()))
# Calculate initial system energy
for i, f in enumerate(system.getForces()):
    state = simulation.context.getState(getEnergy=True, groups={i})
    print(f.getName() + ":\t", state.getPotentialEnergy().in_units_of(kilocalories_per_mole))
print("\nInitial system energy: " + str(simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilocalories_per_mole)))
print("The integrators parameters are: {}, {}, {}".format(
    simulation.integrator.getFriction(), 
    simulation.integrator.getTemperature(),
    simulation.integrator.getStepSize().in_units_of(femtosecond)))

# Energy minimization
if False:
    mini_nsteps = 100
    print("\nEnergy minimization: %s steps" % mini_nsteps)
    simulation.minimizeEnergy(tolerance=100.0*kilojoule/mole, maxIterations=mini_nsteps)
    print(simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilocalories_per_mole))

# Generate initial velocities
if True:
    print("\nGenerate initial velocities")
    simulation.context.setVelocitiesToTemperature(temp)

# Production
if True:
    simulation.currentStep = 0     # Needed to avoid the restart failure when steps exceed the max int of
                                   # 2147483647
    nstep = 250000000              # 500000 steps = 1 ns
    dcdfreq = 50000                # 5000 steps = every 0.01 ns
    restartfreq = 100000
    outputEnergies = 100000
    print("\nMD run: %s steps Traj Saving Freq %s steps" % (nstep, dcdfreq))
    simulation.reporters.append(DCDReporter(outputName+'.dcd', dcdfreq))
    simulation.reporters.append(StateDataReporter(sys.stdout, outputEnergies, step=True, time=True,
                                potentialEnergy=True, temperature=True, progress=True,
                                remainingTime=True, speed=True, totalSteps=nstep, separator='\t'))
    simulation.reporters.append(CheckpointReporter(outputName+'.restart.chk', restartfreq, writeState=False)) # Change writeState to True if you want xml format restart files
                                                                                                              # you should also change the name of the output to .xml in this case
    simulation.step(nstep)
    # Calculate final system energy
    print("\nFinal energy breakdown")
    for i, f in enumerate(system.getForces()):
        state = simulation.context.getState(getEnergy=True, groups={i})
        print(f.getName() + ":\t", state.getPotentialEnergy().in_units_of(kilocalories_per_mole))
    print("\nFinal system energy: " + str(simulation.context.getState(getEnergy=True).getPotentialEnergy().in_units_of(kilocalories_per_mole)))

# Write restart file
if saveXML:
    simulation.saveState(outputName+'.xml')
else:
    simulation.saveCheckpoint(outputName+'.chk')


