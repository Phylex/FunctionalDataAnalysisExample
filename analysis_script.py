#!/usr/bin/python3
import itertools as itt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Background simulation
bkgnd_sim_file_1 = "./data/MC_2012_ZZ_to_4L_2el2mu.csv"
bkgnd_sim_file_2 = "./data/MC_2012_ZZ_to_4L_4mu.csv"
bkgnd_sim_file_3 = "./data/MC_2012_ZZ_to_4L_4el.csv"

# Signal simulation
sig_sim_file_1 = "./data/MC_2012_H_to_ZZ_to_4L_2el2mu.csv"
sig_sim_file_2 = "./data/MC_2012_H_to_ZZ_to_4L_4mu.csv"
sig_sim_file_3 = "./data/MC_2012_H_to_ZZ_to_4L_4el.csv"

bkgnd_sim_2el2mu = pd.read_csv(bkgnd_sim_file_1, delimiter=";")
bkgnd_sim_4mu = pd.read_csv(bkgnd_sim_file_2, delimiter=";")
bkgnd_sim_4el = pd.read_csv(bkgnd_sim_file_3, delimiter=";")
sig_sim_2el2mu = pd.read_csv(sig_sim_file_1, delimiter=';')
sig_sim_4mu = pd.read_csv(sig_sim_file_2, delimiter=';')
sig_sim_4el = pd.read_csv(sig_sim_file_3, delimiter=';')


# First let's define the functions needed. If we get them out of the way here,
# we can see the whole of the functional approach in (hopefully) comparatively
# few lines
def add_electron_entries_to_raw_data(event):
    """ Add the entries for the electrons to the events that only contain
    muons (4mu channel)"""
    data = event[1]
    data['electron_energy'] = ''
    data['electron_px'] = ''
    data['electron_py'] = ''
    data['electron_pz'] = ''
    data['electron_charge'] = ''
    data['electron_relPFIso'] = ''
    data['electron_dxy'] = ''
    data['electron_dz'] = ''
    data['electron_SIP3d'] = ''
    return event


def add_muon_entries_to_raw_data(event):
    """ Add the entries for the muons to the events that only contain electrons
    (4el channel)"""
    data = event[1]
    data['muon_energy'] = ''
    data['muon_px'] = ''
    data['muon_py'] = ''
    data['muon_pz'] = ''
    data['muon_charge'] = ''
    data['muon_relPFIso'] = ''
    data['muon_dxy'] = ''
    data['muon_dz'] = ''
    data['muon_SIP3d'] = ''
    return event


def parse_strings(event):
    """ The events that are read in from the pandas dataframe contain only strings and we need
    numbers so convert all the strings to numbers/arrays where appropriate """
    ed = event[1]
    ed['muon_energy'] = np.array([float(e) for e in ed['muon_energy'].split(',') if e != ''])
    ed['muon_px'] = np.array([float(e) for e in ed['muon_px'].split(',') if e != ''])
    ed['muon_py'] = np.array([float(e) for e in ed['muon_py'].split(',') if e != ''])
    ed['muon_pz'] = np.array([float(e) for e in ed['muon_pz'].split(',') if e != ''])
    ed['muon_charge'] = np.array([float(e) for e in ed['muon_charge'].split(',') if e != ''])
    ed['muon_relPFIso'] = np.array([float(e) for e in ed['muon_relPFIso'].split(',') if e != ''])
    ed['muon_dxy'] = np.array([float(e) for e in ed['muon_dxy'].split(',') if e != ''])
    ed['muon_dz'] = np.array([float(e) for e in ed['muon_dz'].split(',') if e != ''])
    ed['muon_SIP3d'] = np.array([float(e) for e in ed['muon_SIP3d'].split(',') if e != ''])
    ed['electron_energy'] = np.array([float(e) for e in ed['electron_energy'].split(',') if e != ''])
    ed['electron_px'] = np.array([float(e) for e in ed['electron_px'].split(',') if e != ''])
    ed['electron_py'] = np.array([float(e) for e in ed['electron_py'].split(',') if e != ''])
    ed['electron_pz'] = np.array([float(e) for e in ed['electron_pz'].split(',') if e != ''])
    ed['electron_charge'] = np.array([float(e) for e in ed['electron_charge'].split(',') if e != ''])
    ed['electron_relPFIso'] = np.array([float(e) for e in ed['electron_relPFIso'].split(',') if e != ''])
    ed['electron_dxy'] = np.array([float(e) for e in ed['electron_dxy'].split(',') if e != ''])
    ed['electron_dz'] = np.array([float(e) for e in ed['electron_dz'].split(',') if e != ''])
    ed['electron_SIP3d'] = np.array([float(e) for e in ed['electron_SIP3d'].split(',') if e != ''])
    return (event[0], ed)


def build_4_vectors(event):
    """ from the various impulse numbers construct an energy-momentum fourvector """
    # ed stands for event-data
    ed = event[1]
    # transformed event-data (this will contain the data in the structure that we want)
    ted = {}
    e_keys = ['electron_energy', 'electron_px', 'electron_py', 'electron_pz']
    mu_keys = ['muon_energy', 'muon_px', 'muon_py', 'muon_pz']
    mu_vecs = np.array([[me, mpx, mpy, mpz] for me, mpx, mpy, mpz in zip(ed[mu_keys[0]], ed[mu_keys[1]], ed[mu_keys[2]], ed[mu_keys[3]])])
    e_vecs = np.array([[ee, epx, epy, epz] for ee, epx, epy, epz in zip(ed[e_keys[0]], ed[e_keys[1]], ed[e_keys[2]], ed[e_keys[3]])])
    ted['e_fourvector'] = e_vecs
    ted['mu_fourvector'] = mu_vecs
    
    # copy the rest of the values as long as they are not redundant
    for key in ed.keys():
        if key not in e_keys and key not in mu_keys:
            ted[key] = ed[key]
        
    return (event[0], ted)


def to_leptons(event):
    """ organize the data such that the information is grouped by the lepton that it belongs to. This will make the code
    a *lot* more readable later on """
    data = event[1]
    leptons = []
    e_quantities = [data['e_fourvector'],
                    data['electron_charge'],
                    data['electron_relPFIso'],
                    data['electron_dxy'],
                    data['electron_dz'],
                    data['electron_SIP3d']]
    mu_quantities = [data['mu_fourvector'],
                     data['muon_charge'],
                     data['muon_relPFIso'],
                     data['muon_dxy'],
                     data['muon_dz'],
                     data['muon_SIP3d']]
    for fourvec, charge, relPFIso, dxy, dz, SIP3d in zip(*e_quantities):
        lepton_dict = {'p': fourvec, 'type': 'e', 'charge': charge,
                       'dxy': dxy, 'dz': dz, 'relPFIso': relPFIso, 'SIP3d': SIP3d}
        leptons.append(lepton_dict)
    for fourvec, charge, relPFIso, dxy, dz, SIP3d in zip(*mu_quantities):
        lepton_dict = {'p': fourvec, 'type': 'mu', 'charge': charge,
                       'dxy': dxy, 'dz': dz, 'relPFIso': relPFIso, 'SIP3d': SIP3d}
        leptons.append(lepton_dict)
    return (event[0], {'run': data['run'], 'event':data['event'], 'leptons': leptons})


def calculate_transverse_momentum(fourvector):
    """ given a four vector this function simply calculates the corresponding
    transverse momentum The four vector should follow this convention:
    [e, px, py, pz]"""
    return np.sqrt(fourvector[1]**2 + fourvector[2]**2)


def calc_pt(event):
    """ calculate the transverse momentum for each lepton of an event using the
    calculate_transverse_momentum function to do the actual calculation. This
    is a 'wrapper' type function """
    leptons = event[1]['leptons']
    for l in leptons:
        l.update({'pt': calculate_transverse_momentum(l['p'])})
    return event


# this is the first 'filter' function as it returns True/False for an event given a criteris
# this function needs a simple wrapper in the form of a 'lambda' function.
def pt_min(event, min_pt):
    """ function to be used in conjunction with a filter. filters out all
    leptons in an event with a transverse momentum below the min_pt
    threshold"""
    data = event[1]
    leptons = filter(lambda lepton: lepton['pt'] > min_pt, data['leptons'])
    return True if len(list(leptons)) > 4 else False


def pseudorapidity(pt, pz):
    """calculate the pseudorapidity (eta) for a given four vector"""
    # remember opposite/adjacent = tan(theta)
    return -np.log(np.abs(pt/(pz*2)))


def phi(fourvector):
    """ calculate the direction of the lepton in the plane vertical
    to the beam """
    vec = np.array([fourvector[1], fourvector[2]])
    normed_vec = vec/np.linalg.norm(vec)
    return np.arctan2(normed_vec[1], normed_vec[0])


# the next two functions are more wrapper functions for eta and phi calculations
def calc_pseudorapidity(event):
    """ calculates pt for all leptons in an event """
    leptons = event[1]['leptons']
    for l in leptons:
        l.update({'eta': pseudorapidity(l['pt'], l['p'][3])})
    return event


def calc_phi(event):
    """ calculates phi for all leptons in an event """
    leptons = event[1]['leptons']
    for l in leptons:
        l.update({'phi': phi(l['p'])})
    return event


# the next four functions only apply to leptons that are extracted from the event.
# they check that the leptons satisfy some basic requirements so that they can be considered
# in the signal reconstruction. A lot of the parameters have to do with the detector and how
# well it can measure data in various areas.
def lepton_pt_and_eta_requirements(lepton, pt_min, eta_max):
    return lepton['pt'] >= pt_min and np.abs(lepton['eta']) <= eta_max


def impact_parameter_requirements(lepton, min_SIP, max_dxy, min_dz):
    return lepton['SIP3d'] >= min_SIP and lepton['dxy'] <= max_dxy and lepton['dz'] <= max_dz


def relative_isolation_requirements(lepton, min_relPFIso):
    return lepton['relPFIso'] >= min_relPFIso


def remove_unfit_leptons(event, lepton_type, pt_min, eta_max, SIP_min, dxy_max, dz_max, rel_PFIso_min):
    """ remove all the leptons from an event that don't meet the requirements specified by the parameters
    that are passed into this function. This function is only a filter for the leptons in an event, but should be used
    with a map() function as it takes in an event and should also return an event (with the leptons filtered) """
    leptons = event[1]['leptons']
    other_leptons = filter(lambda l: l['type'] != lepton_type, leptons)
    leptons_of_specified_type = filter(lambda l: l['type'] == lepton_type, leptons)
    event[1]['leptons'] = list(itt.chain(filter(lambda l: lepton_pt_and_eta_requirements(l, pt_min, eta_max),
                                                filter(lambda l: impact_parameter_requirements(l, SIP_min, dxy_max, dz_max), 
                                                       filter(lambda l: relative_isolation_requirements(l, rel_PFIso_min),
                                                              leptons_of_specified_type))),
                                         other_leptons))
    return event


# now that the leptons have been filtered out for each event, the event needs to have enough leptons left to do a propper
# reconstruction, so here we filter out the events with to few leptons.
def at_least_n_leptons_of_type(t, n, leptons):
    """function that can be used to filter out leptons of type t"""
    type_t_leptons = list(filter(lambda l: l['type'] == t, leptons))
    return len(type_t_leptons) >= n


# these are functions that are handy to have for the plotting
def extract_leptons(events):
    """ turn an iterator over the events into an iterator over the leptons of all events"""
    return itt.chain.from_iterable(map(lambda e: e[1]['leptons'], events))

def extract_plot_data_for_leptons(leptons, plot_keys):
    """ map an iterator over leptons to the quantiy/quantities of interest.
    Be aware that this function does NOT return an iterator but a list of lists.
    There is one list per plot key provided (the plot_keys must be iterable) """
    plotable_quantities = map(lambda l: [l[key] for key in plot_keys], leptons)
    return list(zip(*list(plotable_quantities)))


def plot(electron_energies, muon_energies, title):
    """ Plot a histogram of the energies of the """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim((0, 20))
    hist, bins, bars = ax.hist([electron_energies, muon_energies], bins=1000, stacked=True, color=['blue', 'darkblue'], label=['electrons', 'muons'])
    plt.legend()
    plt.title(title)
    plt.show()
    

def combined_charge_criteria(leptons, num_of_leptons_considered):
    """function checks if a neutral charge combination can be built from a (sub)set of leptons with length
    num_of_leptons_considered (! the sequence of leptons in the event does not matter !)
    """
    for combination in itt.combinations(leptons, num_of_leptons_considered): # itertools to the rescue (again) (you should know where to find the docs by now)
        if sum([l['charge'] for l in leptons]) == 0:
            return True
    return False


# this is a specialized version of the function above, so it can be applied to the data in the 2el 2mu channel.
def combined_charge_criteria_for_2el2mu_channel(event):
    electrons, muons = filter(lambda l: l['type'] == 'e', event[1]['leptons']), filter(lambda l: l['type'] == 'mu', event[1]['leptons'])
    return combined_charge_criteria(electrons, 2) and combined_charge_criteria(muons, 2)


# this function, though it acts on leptons produces for True or False for an
# entire sequence of leptons and can be used to filter out events in turn.
def strict_pt(leptons):
    """applies strict pt requirements to the leptons"""
    pt = [l['pt'] for l in leptons]
    pt = sorted(pt)
    if pt[-1] < 20:
        return False
    if pt[-2] < 10:
        return False
    for momentum in pt[:-2]:
        if momentum < 4:
            return False
    return True


# we need a function to generate the combinations of two pairs from two iterators
def combinations_of_two_pairs(it1, it2):
    product = list(itt.product(it1, it2))
    combinations_of_two_pairs = []
    for i, e1 in enumerate(product):
        for e2 in product[i:]:
            if hash(str(e1[0])) != hash(str(e2[0])) and\
               hash(str(e1[0])) != hash(str(e2[1])) and\
               hash(str(e1[1])) != hash(str(e2[0])) and\
               hash(str(e1[1])) != hash(str(e2[1])):
                combinations_of_two_pairs.append((e1, e2))
    return combinations_of_two_pairs


# This function is the one that we have defined most recently, it takes a
# list of leptons sorts it according to type and charge of the lepton into four
# lists and then build all possible combinations that could represent a decay
# of two Z bosons.
def build_valid_combinations(leptons):
    leptons1, leptons2 = itt.tee(leptons, 2)
    muons = filter(lambda l: l['type'] == 'mu', leptons1)
    electrons = filter(lambda l: l['type'] == 'e', leptons2)
    es1, es2 = itt.tee(electrons)
    mus1, mus2 = itt.tee(muons)
    positrons = list(filter(lambda l: l['charge'] > 0, es1))
    electrons = list(filter(lambda l: l['charge'] < 0, es2))
    muons = list(filter(lambda l: l['charge'] < 0, mus1))
    antimuons = list(filter(lambda l: l['charge'] > 0, mus2))
    if len(positrons) == 1 and len(electrons) == 1 and\
       len(muons) == 1 and len(antimuons) == 1:
        return [((positrons[0], electrons[0]), (antimuons[0], muons[0]))]
    return list(itt.chain(combinations_of_two_pairs(positrons, electrons),
                          combinations_of_two_pairs(antimuons, muons)))


# another wrapper function, this time for the build_valid_combinations
def calculate_combinations(event):
    leptons = event[1]['leptons']
    combinations = build_valid_combinations(leptons)
    data = event[1]
    data.pop('leptons')
    data['lepton_combinations'] = combinations
    return (event[0], data)


# calculate the energy in the rest frame of the particles
# with the mentioned four vector
def mandelstam_s(fourvectors):
    fourvectors = np.array(fourvectors)
    vec = np.add.reduce(fourvectors)
    minkovski_metric = np.diag([1, -1, -1, -1])
    return vec @ (minkovski_metric @ vec)


# yet another wrapper function, this time we insert the energy of the rest frame into the
# each lepton pair of each combination
def calc_rest_enrgy_for_lepton_pairs(event):
    possible_combinations = event[1]['lepton_combinations']
    event[1]['lepton_combinations'] = [list(map(lambda lp: (lp[0],
                                                            lp[1],
                                                            np.sqrt(mandelstam_s([lp[0]['p'], lp[1]['p']]))),
                                                combination))
                                       for combination in possible_combinations]
    return event


# Here we filter out the combination of leptons (we know that the combinations conserve charge and group the leptons right)
# to the combination that is closest to a signal like event with the rest energy of one lepton pair being close to the
# rest mass of a Z boson.
def pick_combination_closest_to_z_mass(event):
    combinations = event[1]['lepton_combinations']
    closest_combination = combinations[0]
    for c in combinations:
        if abs(c[0][2]-90) < abs(closest_combination[0][2]-90) or abs(c[1][2]-90) < abs(closest_combination[1][2]):
            closest_combination = c
    event[1]['lepton_combinations'] = c
    return event


# this will be an important parameter to distinguish signal from background
def one_decay_close_to_z_mass_shell(event, spread):
    pairs = event[1]['lepton_combinations']
    for p in pairs:
        if abs(p[2] - 90) < spread:
            return True
    return False


# the last thing to do is to specify global parameters that should be used with this analysis
electron_pt_min = 7
electron_max_eta = 2.5
muon_pt_min = 5
muon_max_eta = 2.4
min_impact_param = .1
max_dxy = .1
max_dz = .1
min_rel_iso = 0


def higgs_to_4leptons_via_zz_analysis(raw_data, ept_min, eeta_max, mupt_min, mueta_max, impactparam_min, dxy_max, dz_max, reliso_min, spread):
    # parse and reformat data
    p_and_r_data = map(to_leptons, map(build_4_vectors, map(parse_strings, raw_data)))
    
    pt_and_eta_data = map(calc_pseudorapidity, map(calc_pt, p_and_r_data))
    # filter the leptons of each event
    # remove all muons that dont meet the requirements
    data_with_filtered_muons = map(lambda e: remove_unfit_leptons(e, 'mu',
                                                                  mupt_min,
                                                                  mueta_max,
                                                                  impactparam_min,
                                                                  dxy_max,
                                                                  dz_max,
                                                                  reliso_min),
                                     pt_and_eta_data)
    
    #remove all electrons that dont meet the requirements
    data_with_filtered_leptons = map(lambda e: remove_unfit_leptons(e, 'e',
                                                                    ept_min,
                                                                    eeta_max,
                                                                    impactparam_min,
                                                                    dxy_max,
                                                                    dz_max,
                                                                    reliso_min),
                                     data_with_filtered_muons)
    
    # now split the events into the three streams, one for each channel
    c4mu_raw, c4el_raw, c2el2mu_raw = itt.tee(data_with_filtered_leptons, 3)
    
    # run the data from the unfiltered source through the filters that will only leave the data
    # that should be in the channel
    c4mu = filter(lambda e: at_least_n_leptons_of_type('mu', 4, e[1]['leptons']), c4mu_raw)
    c4el = filter(lambda e: at_least_n_leptons_of_type('e', 4, e[1]['leptons']), c4el_raw)
    c2el2mu = filter(lambda e: at_least_n_leptons_of_type('mu', 2, e[1]['leptons']),
                     filter(lambda e: at_least_n_leptons_of_type('e', 2, e[1]['leptons']), c2el2mu_raw))
    
    # now all events get filtered out where no neutral charge can be constructed from the leptons
    c4mu_f1 = filter(lambda e: combined_charge_criteria(e, 4), c4mu)
    c4el_f1 = filter(lambda e: combined_charge_criteria(e, 4), c4el)
    c2el2mu_f1 = filter(lambda e: combined_charge_criteria_for_2el2mu_channel, c2el2mu)
    
    # now we apply our strict_pt_requirements
    c4mu_f2 = filter(lambda e: strict_pt(e[1]['leptons']), c4mu_f1)
    c4el_f2 = filter(lambda e: strict_pt(e[1]['leptons']), c4el_f1)
    c2el2mu_f2 = filter(lambda e: strict_pt(e[1]['leptons']), c2el2mu_f1)
    
    # now we determine possible combinations of the leptons into two Z decays
    c4mu_c = map(calculate_combinations, c4mu_f2)
    c4el_c = map(calculate_combinations, c4el_f2)
    c2el2mu_c = map(calculate_combinations, c2el2mu_f2)

    # remove all events that don't have any pairs
    c4mu_f3 = filter(lambda e: len(e[1]['lepton_combinations']) > 0, c4mu_c)
    c4el_f3 = filter(lambda e: len(e[1]['lepton_combinations']) > 0, c4el_c)
    c2el2mu_f3 = filter(lambda e: len(e[1]['lepton_combinations']) > 0, c2el2mu_c)

    # calculate the rest energy for every pair
    c4mu_re = map(calc_rest_enrgy_for_lepton_pairs, c4mu_f3)
    c4el_re = map(calc_rest_enrgy_for_lepton_pairs, c4el_f3)
    c2el2mu_re = map(calc_rest_enrgy_for_lepton_pairs, c2el2mu_f3)
    
    # chose the combination form all combinations, where the rest energy of
    # one of the
    # pairs is closest to the rest mass of a Z boson
    c4mu_s = map(pick_combination_closest_to_z_mass, c4mu_re)
    c4el_s = map(pick_combination_closest_to_z_mass, c4el_re)
    c2el2mu_s = map(pick_combination_closest_to_z_mass, c2el2mu_re)
    
    # now all that's left is to filter out all the events that don't have one real and
    # one virtual Z boson
    c4mu_wz = filter(lambda e: one_decay_close_to_z_mass_shell(e, spread),
                     c4mu_s)
    c4el_wz = filter(lambda e: one_decay_close_to_z_mass_shell(e, spread),
                     c4el_s)
    c2el2mu_wz = filter(lambda e: one_decay_close_to_z_mass_shell(e, spread),
                        c2el2mu_s)
    
    # finally we calculate the invariant mass for every event that is left.
    c4mu_energies = map(lambda e:
                        sum(map(lambda pair: pair[2],
                                e[1]['lepton_combinations'])),
                        c4mu_wz)
    c4el_energies = map(lambda e: sum(map(lambda pair: pair[2], e[1]['lepton_combinations'])), c4el_wz)
    c2el2mu_energies = map(lambda e: sum(map(lambda pair: pair[2], e[1]['lepton_combinations'])), c2el2mu_wz)

    return list(c4mu_energies), list(c4el_energies), list(c2el2mu_energies)


if __name__ == "__main__":
    testbkgnd = bkgnd_sim_2el2mu.iterrows()
    testsig = sig_sim_2el2mu.iterrows()
    print("Background")
    testout = higgs_to_4leptons_via_zz_analysis(testbkgnd, electron_pt_min,
                                                electron_max_eta, muon_pt_min,
                                                muon_max_eta, min_impact_param,
                                                max_dxy, max_dz,
                                                min_rel_iso, 5)
    print(len(testout[0]))
    print(len(testout[1]))
    print(len(testout[2]))
    print("Signal")
    testout = higgs_to_4leptons_via_zz_analysis(testsig, electron_pt_min,
                                                electron_max_eta, muon_pt_min,
                                                muon_max_eta, min_impact_param,
                                                max_dxy, max_dz,
                                                min_rel_iso, 5)
    print(len(testout[0]))
    print(len(testout[1]))
    print(len(testout[2]))
