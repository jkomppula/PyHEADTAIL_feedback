import numpy as np

import PyHEADTAIL
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning



def generate_one_turn_map(original_map, list_of_optics_elements,
                          list_of_PyHT_objects, optics_file,
                          phase_x_col, beta_x_col, phase_y_col, beta_y_col, 
                          circumference, alpha_x=None, beta_x=None, D_x=None,
                          alpha_y=None, beta_y=None, D_y=None, accQ_x=None, 
                          accQ_y=None, Qp_x=None, Qp_y=None, app_x=0, app_y=0, 
                          app_xy=0, other_detuners=[], use_cython=False):
    
    with open(optics_file) as f:
        content = f.readlines()
        content = [x.strip() for x in content]


    element_coordinates = np.zeros((len(list_of_optics_elements),5))
    for j, element in enumerate(list_of_optics_elements):
        element_name = element[0]
        line_found = False
        for i, l in enumerate(content):
            s = l.split()
            
            if s[0] == ('"' + element_name + '"'):
                line_found = True
                element_coordinates[j,0] = j
                element_coordinates[j,1] = float(s[phase_x_col])
                element_coordinates[j,2] = float(s[beta_x_col])
                element_coordinates[j,3] = float(s[phase_y_col])
                element_coordinates[j,4] = float(s[beta_y_col])
                
                
        if line_found == False:
            raise ValueError('Element ' + element_name + ' not found from the optics file ' + optics_file)
        
    element_coordinates[:,3] = element_coordinates[:,3]*accQ_x/accQ_y
    
    
    
    map_locations = []
    
    for j, element in enumerate(list_of_optics_elements):
        if element[1] == 'x':     
            map_locations.append(element_coordinates[j,1])
        elif element[1] == 'y':     
            map_locations.append(element_coordinates[j,3])
        else:
            raise ValueError('Unknown plane')
    
    map_locations = np.array(map_locations)
    
    order = np.argsort(map_locations)
    print('ORDER: ' + str(order))
    
    s = []
    
    for i, j in enumerate(order):
        if i == 0:
            if map_locations[j]/accQ_x*circumference != 0.:
                    s.append(0.)
        s.append(map_locations[j]/accQ_x*circumference)
    
    s.append(circumference)
    print 'S: ' + str(s)
    s = np.array(s)
#    s = (np.arange(0, n_segments + 1) * circumference / n_segments)
    alpha_x = 0.*s
    beta_x = 0.*s+beta_x
    D_x = 0.*s+D_x
    alpha_y = 0.*s
    beta_y = 0.*s+beta_y
    D_y = 0.*s+D_y
    
    print 'beta_x: ' + str(beta_x)
    print 'beta_y: ' + str(beta_y)
    
    
    detuners = []
    if any(np.atleast_1d(Qp_x) != 0) or \
            any(np.atleast_1d(Qp_y) != 0):
        detuners.append(Chromaticity(Qp_x, Qp_y))
    if app_x != 0 or app_y != 0 or app_xy != 0:
        detuners.append(AmplitudeDetuning(app_x, app_y, app_xy))
    detuners += other_detuners

    transverse_map = TransverseMap(
        s=s,
        alpha_x=alpha_x, beta_x=beta_x, D_x=D_x,
        alpha_y=alpha_y, beta_y=beta_y, D_y=D_y,
        accQ_x=accQ_x, accQ_y=accQ_y, detuners=detuners)

    transverse_map.n_segments = len(s)-1

    transverse_map.name = ['P_%d' % ip for ip in range(len(s)-1)]
    transverse_map.name.append('end_ring')
    

    for i_seg, m in enumerate(transverse_map):
        m.i0 = i_seg
        m.i1 = i_seg+1
        print('transverse_map.s[i_seg]: ' + str(transverse_map.s[i_seg]))
        print('transverse_map.s[i_seg+1]: ' + str(transverse_map.s[i_seg+1]))
        m.s0 = transverse_map.s[i_seg]
        m.s1 = transverse_map.s[i_seg+1]
        m.name0 = transverse_map.name[i_seg]
        m.name1 = transverse_map.name[i_seg+1]
        m.beta_x0 = transverse_map.beta_x[i_seg]
        m.beta_x1 = transverse_map.beta_x[i_seg+1]
        m.beta_y0 = transverse_map.beta_y[i_seg]
        m.beta_y1 = transverse_map.beta_y[i_seg+1]

    # insert transverse map in the ring
    one_turn_map = []
    
    for m in original_map:
        if type(m) != PyHEADTAIL.trackers.transverse_tracking.TransverseSegmentMap:
            one_turn_map.append(m)
        
    
    for i, m in enumerate(transverse_map):
        one_turn_map.append(m)
        if i < len(order):
            element_idx = order[i]
            one_turn_map.append(list_of_PyHT_objects[element_idx])
    
    return one_turn_map


def find_object_locations(list_of_objects, optics_file,
                          phase_x_col, beta_x_col,
                          phase_y_col, beta_y_col):
    
    with open(optics_file) as f:
        content = f.readlines()
        content = [x.strip() for x in content]


    object_locations = []
    for j, element in enumerate(list_of_objects):
        element_name = element[0]
        line_found = False
        for i, l in enumerate(content):
            s = l.split()
            
            if s[0] == ('"' + element_name + '"'):
                line_found = True
                object_locations.append((element[0],
                                 float(s[phase_x_col]),
                                 float(s[beta_x_col]),
                                 float(s[phase_y_col]),
                                 float(s[beta_y_col])))
                
                
        if line_found == False:
            raise ValueError('Element ' + element_name + ' not found from the optics file ' + optics_file)
        
    
    
    return object_locations
    