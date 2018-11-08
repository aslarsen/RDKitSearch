import numpy as np
import random
import math
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import PDBWriter
from rdkit.Chem import MolToPDBBlock
from rdkit.Chem import MolFromPDBBlock
import time
import multiprocessing
import multiprocessing.pool
import sys
import copy
try:
    import qmxtb
except:
    print ('qmxtb not available')
try:
    import qmgaussian
except:
    print ('qmgaussian not available')

def optimise_multiprocessing(jobs):
    count = 0
    new_conformers = []
    method = jobs[1]
    name = jobs[0]
    for conformer in jobs[2]:
        count += 1
        conformer.update_molecule(method, name)
        new_conformers.append(conformer)
    return new_conformers, count

# wrapper stuff to allow multiprocess to spawn another process
class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


class Conformer:
    def __init__(self, molecule):
        """ 
        Wrapper class for RDkit molecule
        """ 
        self.molecule = Chem.Mol(molecule)
        self.dihedrals = [ angle[4] for angle in self.get_dihedrals() ]
        self.energy = None

    def update_molecule(self, method="MMFF94", name="job"):
        """
        Optimise molecule calculate energy and set the dihedrals angles
        """
        if method == "MMFF94":
            self.optimise_MMFF94()
           #self.energy = self.calculate_energy_MMFF94() 
            self.dihedrals = [ angle[4] for angle in self.get_dihedrals() ]
        elif method == "gaussian": 
            self.optimise_gaussian(name) # energy is updated in optimise_gaussian()
            self.dihedrals = [ angle[4] for angle in self.get_dihedrals() ]
        elif method == "xtb":
            self.optimise_xtb(name)
            self.dihedrals = [ angle[4] for angle in self.get_dihedrals() ]
        else:
            print ('ERROR INVALID METHOD')

     
    def optimise_MMFF94(self):
        """
        Optimise molecule with MMFF94
        """
        mp = AllChem.MMFFGetMoleculeProperties(self.molecule, mmffVariant="MMFF94")
        ff = AllChem.MMFFGetMoleculeForceField(self.molecule, mp)
        opt_fail = ff.Minimize(maxIts=1000,forceTol=0.0001,energyTol=1e-06)

        self.energy = ff.CalcEnergy()

    def optimise_xtb(self, filename = "test"):
        """
        Optimise molecule with xtb
        """
        # first do minimisation with MMFF94 and dihedrals contrainted
        # this will hopefully fix clashes
        mp = AllChem.MMFFGetMoleculeProperties(self.molecule, mmffVariant="MMFF94")
        ff = AllChem.MMFFGetMoleculeForceField(self.molecule, mp)

        current_angles = self.get_dihedrals()
        N = 0
        for angle in current_angles:
            ff.MMFFAddTorsionConstraint(angle[0], angle[1], angle[2], angle[3], False, angle[4], angle[4], 500000.0)
            N += 1

        ff.Minimize(maxIts=500,forceTol=0.0001,energyTol=1e-06)

        # do gaussian opt
        self.molecule.SetProp("_Name", filename)
        output = qmxtb.xtb([self.molecule], method="opt", charge=0, mult=1, charged_fragments=True, cpus=1, keepFiles=True)

        if output[0] is not None:
            #print (output)

            atoms = output[0]['atomicPos']

            energykey = 'xTB_energy'
            self.energy = float(output[0][energykey])
            self.energy = self.energy * 627.15 # converts from hartree to kcal/mol
            conf = self.molecule.GetConformer()
            N = 0
            # This assumes that the atom ordering is the same!!!!
            for xyz in atoms: 
                conf.SetAtomPosition(N, xyz)
                N += 1
        elif output[0] is None:
            # xtb failed, set energy to a high number, this is kind of a hack
            self.energy = 9999.0

    def optimise_gaussian(self, filename = "test"):
        """
        Optimise molecule with gaussian
        """

        # first do minimisation with MMFF94 and dihedrals contrainted
        # this will hopefully fix clashes
        mp = AllChem.MMFFGetMoleculeProperties(self.molecule, mmffVariant="MMFF94")
        ff = AllChem.MMFFGetMoleculeForceField(self.molecule, mp)

        current_angles = self.get_dihedrals()
        N = 0
        for angle in current_angles:
            ff.MMFFAddTorsionConstraint(angle[0], angle[1], angle[2], angle[3], False, angle[4], angle[4], 500000.0)
            N += 1

        ff.Minimize(maxIts=500,forceTol=0.0001,energyTol=1e-06)

        # do gaussian opt
        self.molecule.SetProp("_Name", filename)
        Gmethod = "pm6"
        output = qmgaussian.gaussian([self.molecule], method="opt " + Gmethod, charge=0, mult=1, charged_fragments=True, cpus=1, keepFiles=True)

        if output[0] is not None:
            atoms = output[0]['atomicPos']

            energykey = 'g16:(' + Gmethod + ')_energy'
            self.energy = float(output[0][energykey])
            self.energy = self.energy * 627.15 # converts from hartree to kcal/mol
            conf = self.molecule.GetConformer()
            N = 0
            # This assumes that the atom ordering is the same!!!!
            for xyz in atoms: 
                conf.SetAtomPosition(N, xyz)
                N += 1
        elif output[0] is None:
            # gaussian failed, set energy to a high number, this is kind of a hack
            self.energy = 9999.0
 
    def get_dihedrals(self):
        """
        Get dihedral angles
        """
        raw_rot_bonds =  self.molecule.GetSubstructMatches(Chem.MolFromSmarts("[!#1]~[!$(*#*)&!D1]-!@[!$(*#*)&!D1]~[!#1]"))
        raw_rot_bonds += self.molecule.GetSubstructMatches(Chem.MolFromSmarts("[*]~[*]-[O,S]-[#1]"))
        raw_rot_bonds += self.molecule.GetSubstructMatches(Chem.MolFromSmarts("[*]~[*]-[NX3;H2]-[#1]"))
    
        bonds = []
        rot_bonds = []
        for k,i,j,l in raw_rot_bonds:
            if (i,j) not in bonds and (j,i) not in bonds: # makes sure that dihedrals are unique 
                bonds.append((i,j))
                rot_bonds.append((k,i,j,l))
    
        conf = self.molecule.GetConformer()
    
        angles = []
        for k,i,j,l in rot_bonds:
            theta = rdMolTransforms.GetDihedralDeg( conf, k,i,j,l )
       #     print k,i,j,l,theta
            angles.append([k,i,j,l,theta])

        return angles

    def get_dihedrals_withduplicates(self):

        raw_rot_bonds =  self.molecule.GetSubstructMatches(Chem.MolFromSmarts("[!#1]~[!$(*#*)&!D1]-!@[!$(*#*)&!D1]~[!#1]"))
        raw_rot_bonds += self.molecule.GetSubstructMatches(Chem.MolFromSmarts("[*]~[*]-[O,S]-[#1]"))
        raw_rot_bonds += self.molecule.GetSubstructMatches(Chem.MolFromSmarts("[*]~[*]-[NX3;H2]-[#1]"))
    
        conf = self.molecule.GetConformer()
    
        angles = []
        for k,i,j,l in raw_rot_bonds:
            theta = rdMolTransforms.GetDihedralDeg( conf, k,i,j,l )
            angles.append([k,i,j,l,theta])

        return angles
      
    def set_dihedrals(self, list_of_changes):
        """ 
        Change the dihedrals of the RDkit molecule. The molecule is not optimised
        """
        current_angles = self.get_dihedrals()
        conf = self.molecule.GetConformer()
        for change, current in zip(list_of_changes, current_angles):
          rdMolTransforms.SetDihedralDeg( conf, current[0], current[1], current[2], current[3], change )

        self.dihedrals = list_of_changes

    def sample_rings(self):

        current_dihedrals = [x[4] for x in self.get_dihedrals()]
        AllChem.EmbedMolecule(self.molecule, randomSeed=int(time.time()),  useRandomCoords=True)

        self.set_dihedrals(current_dihedrals)



class ConformerSearchBase:
    def __init__(self, smiles = 'CCCCC'):     
        """
        Base class for conformation search algorithms

        MAX_OPT_COUNT:
        maxumum number of optimisations that can be performed.

        cpu_nr:
        The number CPUs used. More than one will lead to multiprocessing.

        conformers:
        The list of low energy conformers

        pdb_tag:
        Name used to write the output files

        smiles:
        The smiles string of the molecule

        method:
        Determines method to evaultate energy currently MMFF94 or gaussian

        keep_stereochemistry:
        if true then structures that violate the original stereochemistry is discarded

        """
        self.OPT_COUNT = 0
        self.MAX_OPT_COUNT = 10000
        self.min_energy = 999999
        self.min_conformer = []
        self.max_conformers = 25
        self.conformers = []
        self.log = []
        self.cpu_nr = 1
        self.smiles = smiles
        self.method = 'MMFF94'
        self.jobname = 'conformationalsearch'
        self.dihedral_atom_id = []
        self.keep_stereochemistry = True
        self.original_conformer = None
        self.original_smiles = 'Empty'
        self.ring_atoms_nr = 0

    def check_uniqueness(self, test_conformer, conformers_tocheck_against, angle_tolerance = 0.2, energy_tolerance = 0.0000001):

        UNIQUE = True
        for known_conformer in conformers_tocheck_against:
            non_unique_dihedrals_nr = 0 
            test_conformer_dihedrals = test_conformer.get_dihedrals()
            known_conformer_dihedrals = known_conformer.get_dihedrals_withduplicates()

            # test if the conformers are identical in dihedrals
            for angle1 in test_conformer_dihedrals:
                for angle2 in known_conformer_dihedrals:
                    k1,i1,j1,l1,theta1 = angle1                    
                    k2,i2,j2,l2,theta2 = angle2                    

                    if (i1,j1) == (i2,j2) or (i1,j1) == (j2,i2):
                        angle_diff = 180.0 - abs(abs(theta1 - theta2) - 180.0)
                        if abs(angle_diff) < angle_tolerance:
                            non_unique_dihedrals_nr += 1
                            break    

            # test if conformers are mirror images of each other
            non_unique_mirror_dihedrals_nr = 0 
            for angle1 in test_conformer_dihedrals:
                for angle2 in known_conformer_dihedrals:
                    k1,i1,j1,l1,theta1 = angle1                    
                    k2,i2,j2,l2,theta2 = angle2                    

                    if (i1,j1) == (i2,j2) or (i1,j1) == (j2,i2):
                        angle_diff = 180.0 - abs(abs(abs(theta1) - abs(theta2)) - 180.0)
                        if abs(angle_diff) < angle_tolerance:
                            non_unique_mirror_dihedrals_nr += 1
                            break    

            if non_unique_dihedrals_nr >= len(test_conformer.dihedrals):
                UNIQUE = False
            if non_unique_mirror_dihedrals_nr >= len(test_conformer.dihedrals):
                UNIQUE = False
            
            # test if energy difference is within energy tolerance
            if abs(test_conformer.energy - known_conformer.energy) < energy_tolerance:
                UNIQUE = False

        return UNIQUE


    def get_smiles(self, molecule):
        """
        Get smiles fromt mol
        """ 
        
        molecule_copy = copy.deepcopy(molecule)
        test_molecule = Chem.Mol(molecule_copy)        
        test_molecule = Chem.RemoveHs(test_molecule)
        Chem.SanitizeMol(test_molecule)
        Chem.DetectBondStereochemistry(test_molecule,-1)
        Chem.AssignStereochemistry(test_molecule, flagPossibleStereoCenters=True, force=True)
        Chem.AssignAtomChiralTagsFromStructure(test_molecule,-1)
        
        smiles = Chem.MolToSmiles(test_molecule, isomericSmiles=True)

        return smiles

    def update_conformerslist(self, conformers, new_conformer, max_conformers):
        """
        Update conformer list and checks if the proposed structure is unique
        """ 

        LOWER_ENERGY = False
        UNIQUE = True

        for conformer in conformers:
            if conformer.energy > new_conformer.energy:
                LOWER_ENERGY = True
            UNIQUE = self.check_uniqueness(new_conformer, [conformer])
            if UNIQUE == False:
                break
 
        if LOWER_ENERGY == True and UNIQUE == True:
            if len(conformers) > max_conformers:
                removed_element = conformers.index(max(conformers, key = lambda t: t.energy))
                del conformers[removed_element]
                conformers.append(new_conformer)
            else:
                conformers.append(new_conformer)
        elif (len(conformers) < max_conformers) and UNIQUE == True:
            conformers.append(new_conformer)
 
        conformers.sort(key=lambda tup: tup.energy)

    def write_log(self, log, conformers, time, filename, dihedral_ids):
        """ 
        Write final log. The lowest energy for each optimisatin is saved and the list of conformers
        """
        LINE = 'RUN NAMED: ' + self.jobname + ' SMILES: ' + self.smiles + '\n'
        for point in log:
            LINE += 'POINT %7i %14.8f \n' % (point[0], point[1])
 
        N = 1
        for conformer in conformers:
            moves_string = ''
            for move in conformer.dihedrals:
                moves_string = moves_string + ' %7.2f' % move
            LINE += ('CONFORMER %4i ' % (N))  + ( "ENERGY: %14.8f " % (conformer.energy) ) + ' TORSION:     ' + moves_string + '\n'
            N += 1

        LINE += 'ATOM NUMBERS OF DIHEDRAL ANGLES\n'        
        for dihedral in dihedral_ids:
            LINE += 'DIHEDRAL ' + str(dihedral[0]) + ' ' + str(dihedral[1]) + ' ' + str(dihedral[2]) + ' ' + str(dihedral[3]) + '\n'

        LINE += 'TIME ' + str(time) + '\n'
 
        LINE += 'END\n'
 
        f = open(filename, 'w')
        f.write(LINE)
        f.close()


    def update_search(self, conformer):
        """
        Prints if a new energy minimum is found and update the log and list of conformers
        """

        if self.keep_stereochemistry == True:
            # checks if the stereochemistry of the molecule is conserved
            try:
                # somekind of bug here. Making the smiles from the mol object sometimes result in a wrong
                # stereochemistry, but writing to a pdb file and the reading it produces the correct result
                # this is a hack that should be fixed
                pdb_block = MolToPDBBlock(conformer.molecule)
                temp_mol = MolFromPDBBlock(pdb_block)
                test_smiles = self.get_smiles(temp_mol)
                if self.original_smiles != test_smiles:
                    angles = ''
                    for angle in conformer.dihedrals:
                        angles += " %7.2f" % angle
                    #print ('Structure with invalid stereochemistry found. Energy: %10.8f' % (conformer.energy), ' Torsions: ' + angles + " Smiles: " + test_smiles )
                    conformer.energy = 9999.0
            except:
                # this is need to prevent rdkit from crashing if it cant understand the molecule
                angles = ''
                for angle in conformer.dihedrals:
                    angles += " %7.2f" % angle
                #print ('Fail found. Energy: %10.8f' % (conformer.energy), ' Torsions: ' + angles )
                #temp_pdb_name = self.jobname + str(self.OPT_COUNT) + 'fail.pdb'
                #self.write_pdb(conformer.molecule, temp_pdb_name)
                conformer.energy = 9999.0
                

        if (conformer.energy < self.min_energy) and (self.check_uniqueness(conformer, self.conformers) == True):
            angles = ''
            for angle in conformer.dihedrals:
                angles += " %7.2f" % angle
            print ('OPT Count:',"%6i" % (self.OPT_COUNT), 'Energy:',"%14.8f" % (conformer.energy), 'Torsions:', angles)

            self.min_energy = conformer.energy
            self.min_conformer = conformer

            self.write_pdb(self.min_conformer.molecule, self.jobname + '_OPT_' + str(self.OPT_COUNT) + '.pdb')
 
        self.update_conformerslist(self.conformers, conformer, self.max_conformers)
        self.log.append((self.OPT_COUNT, self.min_energy))

    def count_ringatoms(self, molecule):

        ringbonds = molecule.GetSubstructMatches(Chem.MolFromSmarts("[r!a]~[r!a]"))
        ring_atoms = []
        for bond in ringbonds:
            if bond[0] not in ring_atoms:
                ring_atoms.append(bond[0])
            if bond[1] not in ring_atoms:
                ring_atoms.append(bond[1])
        ring_atoms_number = len(ring_atoms)
        return ring_atoms_number

    def init_search(self, smiles, algorithmname, algorithm_tag):

        print ('Conformational search with a ' + algorithmname + ' method')
        print ('Energy method: ' + self.method)
        print ('Smiles: ' + smiles)
        print ('Job name: ' + self.jobname)

        #init molecule and optimise
        mol = Chem.MolFromSmiles(self.smiles)
        mol = Chem.AddHs(mol,explicitOnly=False)
        AllChem.EmbedMolecule(mol, randomSeed=int(time.time()),  useRandomCoords=True)

        #write pre optimised molecule
        self.write_pdb(mol, self.jobname + algorithm_tag + 'preoptimised.pdb')

        #optimise start molecule
        AllChem.MMFFOptimizeMolecule(mol)
        mp = AllChem.MMFFGetMoleculeProperties(mol)
        ffm = AllChem.MMFFGetMoleculeForceField(mol, mp)

        #write optimised molecule
        self.write_pdb(mol, self.jobname + algorithm_tag + 'optimised.pdb')

        Chem.SanitizeMol(mol)
        Chem.DetectBondStereochemistry(mol,-1)
        Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
        Chem.AssignAtomChiralTagsFromStructure(mol,-1)

        self.ring_atoms_nr = self.count_ringatoms(mol)
        print ('Number of non aromatic ring atoms:',str(self.ring_atoms_nr))

        #create start conformer
        start_conformer = Conformer(mol)
        start_conformer.update_molecule('MMFF94')

        self.min_conformer = start_conformer
        self.min_energy = start_conformer.energy
        print ('Start energy:', self.min_energy)

        self.original_conformer = self.min_conformer
        self.original_smiles = self.get_smiles(self.original_conformer.molecule)

        start_angles = start_conformer.get_dihedrals()

        print ('Initial angles')
        start_angles_only = []
        for angle in start_angles:
            print (angle)
            self.dihedral_atom_id.append((angle[0], angle[1], angle[2], angle[3]))
            start_angles_only.append(angle)
        print ('Number of dihedrals:', len(start_angles))

        return start_conformer 

    def optimise_conformers_serial(self, conformers, method):
        """
        Optimise the list of conformers serially using one CPU
        """
        new_conformers = []
        for conformer in conformers:
            if method == "MMFF94":
                conformer.update_molecule(method)
                self.OPT_COUNT += 1        
            elif method in ["gaussian","xtb"]:
                p = MyPool(1)
       
                jobname = self.jobname
                job = [(jobname, method, [conformer])]
                parallel_output = p.imap(optimise_multiprocessing, job)

                p.close()
                p.join()

                output = parallel_output.next()
                conformer = output[0][0]
                self.OPT_COUNT += output[1]
            else:
                print ('ERROR IN METHOD')

            self.update_search(conformer)
            new_conformers.append(conformer)
    
        return new_conformers

 
    def optimise_conformers_parallel(self, conformers, cpu_nr, method):
        """
        Optimise the conformers in parallel using the multiprocessing module. 
        The conformers are divided into chunks for each CPU.
        """
        chunks = [ [] for x in range(cpu_nr) ]
        M = 0
        for conformer in conformers:
            chunks[M].append(conformer)
            M += 1
            if M == (cpu_nr):
                M = 0
        names = ['job'+str(i) for i in range(cpu_nr)]

        jobs = []
        for name, chunk in zip(names, chunks):
            jobs.append((name, method, chunk))

        p = MyPool(cpu_nr)        

        parallel_output = p.imap(optimise_multiprocessing, jobs)
        p.close()
        p.join()

        new_conformers = []
        N = 0
        for n in chunks:
            output = parallel_output.next()
            new_conformers.extend(output[0])
            self.OPT_COUNT += output[1]
            N += 1

        return new_conformers

    def write_pdb(self, molecule, filename):
        """
        Write PDB file
        """
        w = PDBWriter('./'+filename)
        w.write(molecule)

    

class MonteCarlo(ConformerSearchBase):
    def __init__(self, smiles = 'CCCCC'):
        """
        Conformational search with a Monte Carlo algorithm, inherits the ConformerSearchBase class

        generations:
        Sets the max number of generations

        mutation_rate:
        The chance that each dihedral will be mutated

        temperature:
        Used in the metropolis hastings acceptance criteria. A higher temperature will increase the 
        chance, that a new conformer with an energy higher than the current found energy minimum
        will be accepted.
        """

        ConformerSearchBase.__init__(self, smiles)
        self.mutation_rate = 0.20
        self.temperature = 300.0
        self.generations = 50000

    def sample_dihedrals(self, conformer, mutation_rate):
        """
        Goes through the dihedral angles of the child and mutate them
        acording to the mutation rate
        """
        for i in range(len(conformer.dihedrals)):
            if np.random.random() < mutation_rate:

                random_angle = random.uniform(-180,180)
                new_dihedrals = conformer.dihedrals 
                new_dihedrals[i] = random_angle
                conformer.set_dihedrals(new_dihedrals)
        return conformer

    def metropolis_hastings(self, conformer, current_energy_minimum, temperature):
        """
        metropolis hastings exp(dE/(R*T))
        """
        accept = False
        R = 8.314/1000.0
        e_diff = conformer.energy - current_energy_minimum
        metropolis_criterion = np.exp(-e_diff/(R*temperature))

        if e_diff < 0.0:
            accept = True
        elif metropolis_criterion > random.uniform(0,1):
            accept = True

        return accept

    def run(self):
        """
        Executes the conformational search
        """
        start = time.time()
       
        # init search and make initial conformer 
        current_conformer = self.init_search(self.smiles, 'Monte Carlo', '_mc_')
        current_energy_minimum = current_conformer.energy

        number_of_angles = len(current_conformer.dihedrals)

        for generatins in range(self.generations):
            if self.OPT_COUNT > self.MAX_OPT_COUNT:
                break
            new_conformers = []
            for i in range(0, self.cpu_nr):
                #new proposed conformer
                new_conformer = Conformer(Chem.Mol(current_conformer.molecule))

                #sample dihedral angles
                new_conformer = self.sample_dihedrals(new_conformer, self.mutation_rate)

                if self.ring_atoms_nr >= 5:
                    if np.random.uniform(0,1) < 0.10:
                        new_conformer.sample_rings()

                new_conformers.append(new_conformer)

            if self.cpu_nr == 1:
                #optimise molecule
                new_conformers = self.optimise_conformers_serial([new_conformer], self.method)
                new_conformer = new_conformers[0]
            elif self.cpu_nr > 1:
                new_conformers = self.optimise_conformers_parallel(new_conformers, self.cpu_nr, self.method)
                for conformer in new_conformers:
                    self.update_search(conformer)
            else:
                print ('WRONG CPU COUNT')

            new_conformers.sort(key=lambda x: x.energy, reverse=False)
            new_conformer = new_conformers[0]

            if self.metropolis_hastings(new_conformer, current_energy_minimum, self.temperature) == True:
                current_conformer = new_conformer
                current_energy_minimum = current_conformer.energy

      
        end = time.time()
        wall_clock_time = end - start

        # write final log
        logout = self.jobname + '_mc_search_results.txt'

        self.write_log(self.log, self.conformers, wall_clock_time, logout, self.dihedral_atom_id)

        N = 1
        for conformer in self.conformers:
            self.write_pdb(conformer.molecule, self.jobname + '_mc_conformer_' + str(N) + '.pdb')
            N += 1

        print ('DONE!')

class GeneticAlgorithm(ConformerSearchBase):
    def __init__(self, smiles = 'CCCCC'):
        """
        Conformational search with a genetic algorithm, inherits the ConformerSearchBase class

        population_size:
        Determines the number of children in each generation. It works best if the 
        population_size is devisable by the number of CPUs 
        
        generations:
        Sets the max number of generations

        mutation_rate:
        The chance that each dihedral will be mutated

        temperature:
        Determines the change that a conformer gets picked for the next generation.
        A low temperature will lead to a low diversity in the population. 
        High temperature will lead to higher energy structures in the population        
        """
        ConformerSearchBase.__init__(self, smiles)
        self.population_size = 20  
        self.generations = 50000
        self.mutation_rate = 0.20 
        self.temperature = 300.0

    def make_initial_population(self, population_size, length, start_molecule):
        """ 
        Makes the initial population and sets the dihedral angles to a random value 
        The if a bad child if found the population is remade.
        """
        population = []
        current_energies = []
    
        print ("Make initial population")
        remake_population = True
        while remake_population == True:
            for i in range(population_size):
                dihedrals = []
                for j in range(length):
                    dihedrals.append(random.uniform(-180,180))

                new_molecule = Chem.Mol(start_molecule)

                new_conformer = Conformer(new_molecule)
                new_conformer.set_dihedrals(dihedrals)

                # do a quick opt with MMFF94
                new_conformer.update_molecule("MMFF94")
                if self.method == "MMFF94": # only count if the main method is MMFF94
                    self.OPT_COUNT += 1    

                current_energies.append(new_conformer.energy)

                population.append(new_conformer)
   
            mean_energy = np.mean( np.array(current_energies) )
            remake_population = False

            # check if bad child
            for i in range(0, len(current_energies)):
                if current_energies[i] > (mean_energy + 20):
                    population = []
                    current_energies = []
                    remake_population = True
                    print ('Bad initial population remaking')
                    break 
            self.mean_energy = np.mean( np.array(current_energies) )
        print ("Initial population created") 
        return population

    def calculate_probabilities(self, energies, T):
        """
        Calculates the probabilities of the energies by taking the Boltzmann's factor 
        with exp(-E/(R*T)). The probabilities are then normed.
        The mean of the energies is subtracted and then added to make the probabilities
        nicer.
        """
        R = 8.314/1000.
        probabilities = []

        # np.float128 prevents overflow and underflow
        energies = np.array([np.float128(i) for i in energies]) 
    
        mean_energy = np.mean(energies)
        energies -= mean_energy
        
        probabilities = np.exp(-energies/(R*T))
        probabilities /= np.sum(probabilities)

        energies += mean_energy

        return list(probabilities)
 
    def calculate_normalized_fitness(self, population, Temperature):
        """
        Calculates the normalised fitness of the population.
        Very bad children that are 50 kcal/mol above the mean energy
        are removed.
        """
        energies = []

        for conformer in population:
            energies.append(conformer.energy)
   
        mean_energy = np.mean(np.array(energies))
        to_be_deleted = []
        N = 0
        for energy in energies:
            if energy > (mean_energy + 50):
                to_be_deleted.append(N)
            N += 1
              
        for index in sorted(to_be_deleted, reverse=True): # remove very bad children
            del energies[index]
            del population[index]

        #calculate probability
        normalized_fitness = self.calculate_probabilities(energies, Temperature)
          
        return normalized_fitness
    
    def make_mating_pool(self, population, fitness, population_size):
        """
        Makes a new mating pool by picking from the populatin according to children probability
        """

        mating_pool = []

        index = np.random.choice(range(len(population)), size=population_size, p=fitness)
        for i in index:
            mating_pool.append(population[i])
          
        return mating_pool
    
    def crossover(self, parent_A, parent_B):
        """
        Crosses the dihedral angles of parent A and parent B in order to produce a new child
        """

        if (len(parent_A.dihedrals)-1) > 1:

            cut = np.random.randint(0,len(parent_A.dihedrals))
            child_angles = parent_A.dihedrals[:cut] + parent_B.dihedrals[cut:]

            new_molecule = Chem.Mol(parent_A.molecule)

            child = Conformer(new_molecule)
            child.set_dihedrals(child_angles)

        else:
            child = parent_A
             
        return child
    
    def mutate(self, child, mutation_rate):
        """
        Goes through the dihedral angles of the child and mutate them
        acording to the mutation rate
        """
        for i in range(len(child.dihedrals)):
            if np.random.random() < mutation_rate:

                random_angle = random.uniform(-180,180)
                new_dihedrals = child.dihedrals 
                new_dihedrals[i] = random_angle
                child.set_dihedrals(new_dihedrals)

        return child

    def reproduce(self, mating_pool, population_size, mutation_rate):
        """
        Makes a new population by making a new child by crossover and then
        mutating it.
        """
        new_population = []
        for n in range(population_size):

            parent_A = random.choice(mating_pool)
            parent_B = random.choice(mating_pool)
            new_child = self.crossover(parent_A, parent_B)
            new_child = self.mutate(new_child, mutation_rate)
            # note that the child is still not optimised
            
            new_population.append(new_child)

        return new_population
    
    def run(self):
        """
        Executes the conformational search
        """
        start = time.time()

        # make initial conformer 
        current_conformer = self.init_search(self.smiles, 'genetic algorithm', '_ga_')

        number_of_angles = len(current_conformer.dihedrals)

        population = self.make_initial_population(self.population_size, number_of_angles, current_conformer.molecule)
        
        for generation in range(self.generations):
            if self.OPT_COUNT > self.MAX_OPT_COUNT:
                break

            fitness = self.calculate_normalized_fitness(population, self.temperature)

            mating_pool = self.make_mating_pool(population, fitness, self.population_size)

            population = self.reproduce(mating_pool, self.population_size, self.mutation_rate)

            if self.ring_atoms_nr >= 5:
                for conformer in population:
                    if np.random.uniform(0,1) < 0.10:
                        conformer.sample_rings()

            if self.cpu_nr == 1:
                population = self.optimise_conformers_serial(population, self.method)
            elif self.cpu_nr > 1:
                population = self.optimise_conformers_parallel(population, self.cpu_nr, self.method)
                # update search
                energies_conformer_pairs = []
                for conformer in population:
                    energies_conformer_pairs.append((conformer.energy, conformer))
                energies_conformer_pairs.sort(key=lambda tup: tup[0])                
                for pair in energies_conformer_pairs:
                    self.update_search(pair[1])
            else:
                print ('WRONG CPU COUNT')

        end = time.time()
        wall_clock_time = end - start

        # write final log
        logout = self.jobname + '_ga_search_results.txt'

        self.write_log(self.log, self.conformers, wall_clock_time, logout, self.dihedral_atom_id)

        N = 1
        for conformer in self.conformers:
            self.write_pdb(conformer.molecule, self.jobname + '_ga_conformer_' + str(N) + '.pdb')
            N += 1

        print ('DONE!')


if __name__ == "__main__":

    def do_search(INPUT):
        tag = INPUT[0]
        smiles = INPUT[1]
        total = 5
        for i in range(0, total):
            search = MonteCarlo(smiles)
            #search = GeneticAlgorithm(smiles)
            search.jobname = tag + '_' + str(i)
            search.MAX_OPT_COUNT = 10000
            search.temperature = 300
            search.mutation_rate = 0.3
            search.generations = 50000
            search.max_conformers = 25
            search.cpu_nr = 1
            search.method = "xtb"
            search.run()


    all_smiles = ( ["molecule_2" , r"c1(Cc2ccc(C#C[C@@H](N(C(N)=O)O)C)s2)ccc(F)cc1"],
                   ["molecule_7" , r"CC1C=C(C=CC=1OC1N=CC(Br)=CN=1)NC(=O)NC(=O)C1C=CC=CC=1N(C)C"],
                   ["molecule_13", r"C1C[C@](C)([C@@H]([C@@H](\C=C\[C@@H](C(C)C)C)C)CC2)[C@@H]2\C(=C\C=C(/C(=C)CC[C@@H]3O)\C3)\C1"],
                   ["molecule_69", r"COC1C=C(CNC(=O)CCCC/C=C/C(C)C)C=CC=1O"],
                   ["molecule_96", r"CC(C)/N=C(\N)/N=C(\N)/NOCCCOC1C=CC(=CC=1Cl)OC(F)(F)F"],
                   ["molecule_0" , r"CCCCCCCCCCC"],
                   ["molecule_100", r"CCCCC[C@H](O)/C=C/[C@@H]1[C@@H](C/C=C\CCCC(O)=O)[C@@H]2C[C@H]1OO2"],
                   ["molecule_87" , r"COC1=CC(/C=C/C(=O)CC(=O)/C=C/C2C=C(OC)C(O)=CC=2)=CC=C1O"],
                   ["molecule_73" , r"CC(C)(C)C1C=CC(=CC=1)C(=O)CCN1CCC(CC1)OC(C1C=CC=CC=1)C1C=CC=CC=1"],
                   ["molecule_57" , r"CNCC[C@H](OC1C=CC(=CC=1)C(F)(F)F)C1C=CC=CC=1"],
                   ["molecule_99" , r"COC1C=CC(CCN(CCC)CCC)=CC=1OCCC1C=CC=CC=1"],
                   ["molecule_98" , r"CC(C)C1N=C(C(C)C)C(COC)=C(C=1/C=C/[C@@H](O)C[C@@H](O)CC(O)=O)C1C=CC(F)=CC=1"],
                   ["molecule_76" , r"COCCCC/C(=N\OCCN)/C1=CC=C(C=C1)C(F)(F)F"],
                   ["molecule_74" , r"CNC(=O)C1C=C(C=CC=1)NCC(=O)NCCC1C=CC(OC)=C(C=1)OC"],
                   ["molecule_70" , r"CCCCCNC(=N)N/N=C/C1=CNC2C=CC(=CC1=2)OC"],
                   ["molecule_71" , r"COC1=CC(N)=C(Cl)C=C1C(=O)N[C@H]1CCN(C[C@H]1OC)CCCOC1C=CC(F)=CC=1"] )

    NUMBER_OF_CPU = 8
   # p = multiprocessing.Pool(NUMBER_OF_CPU)
   # p.map(do_search, all_smiles)

    p = MyPool(NUMBER_OF_CPU) # use multiprocessing wrapper
    p.map(do_search, all_smiles)

    #do_search(['test_0', r'CCCCCCCCCCC'])


    #do_search(['test_27', r'CC(C)C[C@@H]1NC(=O)[C@H](C)N(C)C(=O)CNC(=O)/C(=C\C2C=CC=CC=2)/N(C)C1=O'])

    #do_search(['test_13', r'C1C[C@](C)([C@@H]([C@@H](\C=C\[C@@H](C(C)C)C)C)CC2)[C@@H]2\C(=C\C=C(/C(=C)CC[C@@H]3O)\C3)\C1'])

    #do_search(['test_100', r'CCCCC[C@H](O)/C=C/[C@@H]1[C@@H](C/C=C\CCCC(O)=O)[C@@H]2C[C@H]1OO2'])

    #do_search(['test_71', r'COC1=CC(N)=C(Cl)C=C1C(=O)N[C@H]1CCN(C[C@H]1OC)CCCOC1C=CC(F)=CC=1'])

    #do_search(['test_110', r'C[C@H](Cn1cnc2c1ncnc2N)OCP(=O)(OCOC(=O)OC(C)C)OCOC(=O)OC(C)C'])

    #do_search(['test_105', r'CC(C)[C@@H](C(=O)N1CC2(CC2)C[C@H]1c3[nH]c(cn3)c4ccc-5c(c4)C(c6c5ccc(c6)c7ccc8c(c7)[nH]c(n8)[C@@H]9[C@H]1CC[C@H](C1)N9C(=O)[C@H](C(C)C)NC(=O)OC)(F)F)NC(=O)OC'])

