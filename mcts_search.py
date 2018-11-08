import random
import math
import hashlib
import logging
import argparse
import numpy as np
import time
import copy

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import PDBWriter
from rdkit.Chem import MolFromPDBFile

from conformational_search import Conformer
from conformational_search import ConformerSearchBase

"""
Takes a smiles string and perform monte carlo tree search on the molecule using geometry geometry optimisation with RDkit 
The monte carlo tree search code is heavily modified from https://github.com/haroldsultan/MCTS
"""

class State(Conformer):
    def __init__(self, molecule = None, dihedrals=[], turn = None, picked_dihedrals = [], parent_node = None):
        """
        Extends the Conformer class with functions relevant for MCTS
        """
        Conformer.__init__(self, molecule)
        self.turn = turn
        self.dihedrals = dihedrals
        self.dihedralsnumber = len(self.dihedrals)
        self.picked_dihedrals = list(picked_dihedrals)
        self.parent_node = parent_node 

    def next_state(self, mutation_rate = 0.20):
        """
        Samples the dihedrals angles and returns a new state with one turn less
        """
        new_moves = list(self.dihedrals) 
        new_picked_list = list(self.picked_dihedrals)
       # random.shuffle(new_picked_list)        
       # random_index = new_picked_list[0]
       # del new_picked_list[0]
              
        for i in range(0,len(self.dihedrals)):
            if np.random.random() < mutation_rate:
                change_angle = random.uniform(-180,180)
                new_moves[i] = change_angle

        new_molecule = Chem.Mol(self.molecule)
 
        next = State(new_molecule, new_moves, self.turn-1, new_picked_list)
        next.set_dihedrals(new_moves)

        return next

    def terminal(self):
        """
        Checks is there is any turns left
        """
        if self.turn < 1:
            return True
        return False

    def __hash__(self):
        return int(hashlib.md5(str(self.energy).encode('utf-8')).hexdigest(),16)

    def __eq__(self,other):
        if hash(self)==hash(other):
            return True
        return False

class Node():
    def __init__(self, state, parent = None, node_index = '0'):
        """
        The Node class contains a state object and have one parent node and a number of children
        """
        self.visits = 0
        self.reward = 0.0	
        self.state = state
        self.children = []
        self.parent = parent	
        self.node_index = node_index 
        self.rollout_energy = 99999.0

    def add_child(self, child_state):
        """
        Add a child to the node
        """
        child_index = self.node_index + ' ' + str(len(self.children))
        child = Node(child_state, self, child_index)
        self.children.append(child)
    
    def update(self, reward):
        """
        Update the reward and number of visits
        """ 
        self.reward += reward
        self.visits += 1
   
    def childless(self):
        if len(self.children) == 0:
            return True
        else:
            return False
 
    def fully_expanded(self):
        """
        The number of rotatable determies the number of children
        """
        if len(self.children) == self.state.dihedralsnumber:
            return True
        else:
            return False

    def kill_child(self, child_state):
        """
        Remove a child
        """
        index = self.children.index(child_state)
        del self.children[index]

    
class MCTSConformationSearch(ConformerSearchBase):
    def __init__(self, smiles = 'CCCCC'):
        """
        Performs a Monte Carlo Tree Search(MCTS) conformational search.
        
        mutation_rate: the chance that a dihedral angle is sampled at each new node/state in the tree

        budget: number of iterations of the MCTS algorithm

        nesting_level: the level of expansion of the front node. If 0 then the front node is expanded only once.
        If equal to 1, then each subnode is also expanded.

        reward_temperature: determines how much reward is given to nodes higher in energy than the currently known
        global minimum

        max_reward: The max posible reward 

        pruning_temperature: Determines how likely it is for a node to be removed if it is high in energy. 
        A higher temperature lowers this chance

        """
   
        ConformerSearchBase.__init__(self, smiles)
        self.smiles = smiles
        self.budget = 2000000
        self.nesting_level = 0
        self.mutation_rate = 0.2
        self.reward_temperature = 50.0
        self.max_reward = 15.0
        self.pruning_temperature = 1000.0

        #MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration. 
        self.SCALAR = 1/math.sqrt(2.0)


    def uctsearch(self, budget, root):
        """
        Performs the MCTS search
        """
        current_topnode = copy.deepcopy(root)

        for iter in range(int(budget)):
            if self.OPT_COUNT > self.MAX_OPT_COUNT:
                break
            front = self.treepolicy(current_topnode)

            # do recursive expansion
            new_children = self.recursive_expansion(front, self.nesting_level)

            current_energy_minimum = self.min_energy            
 
            # default policy loop
            rolledout_states = []                        
            for child in new_children:
                state_rollout = self.defaultpolicy(child.state)
                child.state = state_rollout
                rolledout_states.append(state_rollout)

            # optimise the States
            if self.cpu_nr == 1:
                optimised_states = self.optimise_conformers_serial(rolledout_states, self.method)               
            elif self.cpu_nr > 1:
                optimised_states = self.optimise_conformers_parallel(rolledout_states, self.cpu_nr, self.method)
                for state in optimised_states:
                    self.update_search(state)
           
            rolloutstate_nodes_pairs = zip(optimised_states, new_children)            
 
            # remove states that violate the stereochemistry
            if self.keep_stereochemistry == True:
                pairs_to_kill = []
                pairs_to_survive = []
                for pair in rolloutstate_nodes_pairs:
                    if pair[0].energy == 9999.0:                
                        pairs_to_kill.append(pair)
                    else:
                        pairs_to_survive.append(pair)

                for pair in pairs_to_kill:
                    pair[1].parent.kill_child(pair[1])

                rolloutstate_nodes_pairs = pairs_to_survive

            rolloutstate_nodes_pairs = sorted(rolloutstate_nodes_pairs, key=lambda x: x[0].energy, reverse=False)

            # give reward and update tree
            new_minimum_node_found, new_minimum_node = self.give_reward(rolloutstate_nodes_pairs, current_energy_minimum, self.reward_temperature, self.max_reward)

            # reset the tree if a new minima is found 
            if new_minimum_node_found == True:
                current_topnode.reward = 0
                current_topnode.visits = 0
                current_topnode.parent = None
                current_topnode.children = []
                current_topnode.state = new_minimum_node.state
                current_topnode.state.turn = len(new_minimum_node.state.dihedrals)
                current_topnode.nodex_index = '0'

            else:
                # kill bad children. Can be turned off   
                self.do_pruning(rolloutstate_nodes_pairs, self.pruning_temperature)

        return self.bestchild(root,0)

    def give_reward(self, rolloutstate_nodes_pairs, min_energy, temperature, max_reward):
        """
        Give reward to the node. If the node energy is lower than the currently known
        global minima then it recives the max_reward. If the energy is not lower then 
        the reward is given based on exp(-deltaE/(R*T))
        """      
        new_minimum_node = None
        new_minimum_node_found = False 
        for pair in rolloutstate_nodes_pairs:
            max_reward = 15
            reward = 0
            
            child_energy = pair[0].energy
            child = pair[1] 

            e_diff = child_energy - min_energy            
            R = 8.314/1000.0

            if child_energy < min_energy:
                reward = max_reward
                min_energy = child_energy
                new_minimum_node = child
                new_minimum_node_found = True
            else:
                metropolis_criterion = np.exp(-e_diff/(R*temperature))
                reward = max_reward*metropolis_criterion

            self.backup(child, reward)

        return new_minimum_node_found, new_minimum_node

    def do_pruning(self, rolloutstate_nodes_pairs, pruning_temperature):
        """
        Removes nodes if they fail the metropolis criterion
        """
        children_to_prune = []
        for pair in rolloutstate_nodes_pairs:
            child_energy = pair[0].energy
            child = pair[1] 

            e_diff = child_energy - self.min_energy
            R = 8.314/1000.0
            T = 1000.0
            metropolis_criterion = np.exp(-e_diff/(R*pruning_temperature))

            if e_diff <= 0.0:
                pass
            elif metropolis_criterion < random.uniform(0,1):
                children_to_prune.append(child)
        for child_to_kill in children_to_prune:
            child_to_kill.parent.kill_child(child_to_kill)

    
    def recursive_expansion(self, node, level):
        """
        Performs the expansion of the node.
        """
        
        all_children = [] 

        if node.state.terminal() == False:
            node = self.expand_all(node)        
            if level != 0:       
                for child in node.children: 
                    node = self.expand_all(node)        
                    next_children = self.recursive_expansion(child, level-1)
                    all_children.extend(next_children)
            else:
                return node.children
        else:
            # end of tree reached 
            self.backup_visits(node)
            return []

        return all_children
        
    def defaultpolicy(self, state):
        # do rollout
        state = state
        state = state.next_state(self.mutation_rate)
        
        if self.ring_atoms_nr >= 5:
            if np.random.uniform(0,1) < 0.25:
                state.sample_rings()

        return state
 
    def treepolicy(self, node):
      #a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
        while node.state.terminal() == False and node.childless() == False :
            node = self.bestchild(node,self.SCALAR)
            if random.uniform(0,1)<(0.05):
                # small chance that random node is picked
                break
        if node.state.terminal():
            return node
        else:
            return node
        
    def expand_all(self, node):
        while node.fully_expanded()==False:
            node = self.expand(node)
        return node
    
    def expand(self, node):
        tried_children = [c.state for c in node.children]
        new_state = node.state.next_state(self.mutation_rate)
        node.add_child(new_state)
        return node

    #current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
    def bestchild(self, node, scalar):
        bestscore=0.0
        bestchildren=[]
        for c in node.children:
            #print 'best', '%10.3f' % (c.reward), c.visits, 'node index:', c.node_index, 'state', c.state.turn, c.state.picked_dihedrals
            exploit=c.reward/c.visits

            explore=math.sqrt(2.0*math.log(node.visits)/float(c.visits))	

            score=exploit+scalar*explore
            if score==bestscore:
                bestchildren.append(c)
               # print 'score', '%10.3f' % (score), '%10.3f' % (c.reward), c.visits, 'node index:', c.node_index
            if score>bestscore:
                bestchildren=[c]
                bestscore=score
               # print 'score', '%10.3f' % (score), '%10.3f' % (c.reward), c.visits, 'node index:', c.node_index
        if len(bestchildren)==0:
            print ("OOPS: no best child found, probably fatal")
            best = node.parent
            self.backup_visits(node)
        else:
            best = random.choice(bestchildren)
        return best


    def backup(self, node, reward):
        while node != None:
            node.visits +=  1
            node.reward += reward
            node = node.parent
        return

    def backup_visits(self, node):
        while node != None:
            node.visits+=1
            node = node.parent
        return

    def run(self):

        start = time.time()

        # init start state
        start_conformer = self.init_search(self.smiles, 'MCTS algorithm', '_mcts_')
        moves = [ 0.0 for x in range(len(start_conformer.dihedrals)) ] 
        turns = len(start_conformer.dihedrals)
        remaning_dihedrals = range(0,len(start_conformer.dihedrals))
        start_state = State( start_conformer.molecule, moves, turns, remaning_dihedrals) 

        # init start node
        genesis_node = Node(start_state)
        
        final_node = self.uctsearch(self.budget, genesis_node)
        
        # write energy minimum
        self.write_pdb(self.min_conformer.molecule, 'molecule_mcts_' + self.jobname + '_minimum.pdb')

        end = time.time()
        wall_clock_time = end - start

        # write final log
        logout = self.jobname + '_mcts_search_results.txt'

        self.write_log(self.log, self.conformers, wall_clock_time, logout, self.dihedral_atom_id)

        N = 1 
        for conformer in self.conformers:
            self.write_pdb(conformer.molecule, self.jobname + '_conformer_' + str(N) + '.pdb')
            N += 1
        print ('TOTAL OPT:', self.OPT_COUNT)
        print ('DONE!')

import multiprocessing

def do_mcts(INPUT):
    tag = INPUT[0]
    smiles = INPUT[1]
    total = 25
    for i in range(0, total):
        search = MCTSConformationSearch(INPUT[1])
        search.jobname = tag + '_' + str(i)
        search.MAX_OPT_COUNT = 10000
        search.budget = 5000000
        search.max_conformers = 25
        search.cpu_nr = 1
        search.method = "MMFF94"
        search.mutation_rate = 0.2

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



#do_mcts(('lol', all_smiles[9][1]))
 
#do_mcts(('lol', all_smiles[5][1]))

p = multiprocessing.Pool(16)
p.map(do_mcts, all_smiles)

