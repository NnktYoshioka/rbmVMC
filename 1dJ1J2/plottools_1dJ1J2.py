import yaml
import datetime
import matplotlib.pyplot as plt
import numpy as np
from ExactWaveFunc_1dJ1J2 import *

# The following three lines are neccesary when you save the file.
##d = datetime.datetime.today()
##directory = 'DIII_Lambda_scaling_cyl'
##filename = '/W_%2d_'%W + d.strftime("%Y%m%d_%H%M%S")
def PredateName(vmc, compflag, Tsymmflag):
    if compflag == 0:
        if Tsymmflag == 0:
            return '1dJ1J2_real_' + '%d_%d_'%(vmc.n_vis, vmc.alpha)
        if Tsymmflag == 1:
            return '1dJ1J2_real_Tsymm_' + '%d_%d_'%(vmc.n_vis, vmc.alpha)

    if compflag == 1:
        if Tsymmflag == 0:
            return '1dJ1J2_complex_' + '%d_%d_'%(vmc.n_vis, vmc.alpha)
        if Tsymmflag == 1:
            return '1dJ1J2_complex_Tsymm_' + '%d_%d_'%(vmc.n_vis, vmc.alpha)

def save_yaml(vmc, EnPerSite, params, param_contents, directory, predatename, d):

    print '\nThe date is ' + str(d.strftime("%Y%m%d_%H%M%S"))

    filename = predatename + d.strftime("%Y%m%d_%H%M%S")
    if len(vmc.rbm.b) == vmc.alpha:
        obj = { 'overlaps': vmc.overlaps, 'alpha': vmc.alpha, 'n_vis': vmc.n_vis,'EnPerSite':EnPerSite,  \
            'date_time':d.strftime("%Y%m%d_%H%M%S"), 'params':params, 'param_contents':param_contents,  \
            'a':vmc.rbm.a, 'bf':vmc.rbm.bf, 'Wf':vmc.rbm.Wf, 'J1':vmc.J1, 'J2':vmc.J2}
    else:
        obj = { 'overlaps': vmc.overlaps, 'alpha': vmc.alpha, 'n_vis': vmc.n_vis,'EnPerSite':EnPerSite,  \
            'date_time':d.strftime("%Y%m%d_%H%M%S"), 'params':params, 'param_contents':param_contents,  \
            'a':vmc.rbm.a, 'b':vmc.rbm.b, 'W':vmc.rbm.W, 'J1':vmc.J1, 'J2':vmc.J2}

    with open(directory + filename + '.yml', 'w') as file:
        yaml.dump(obj,file,default_flow_style=False)

    print 'Successfully saved as ' + directory + filename + '.yml\n'


def plot_yaml(yamlname):
    print "Loading " + yamlname + '.yml...'
    f =open(yamlname + '.yml','r+')
    yamlfile = yaml.load(f)
    overlaps, EnPerSite = [yamlfile['overlaps'], yamlfile['EnPerSite']]

    J1_, J2_ = [yamlfile['params'][3], yamlfile['params'][4]]

    # Plot the learning iteration
    plt.plot(np.real(EnPerSite))
    if yamlfile['n_vis'] <= 12:
        enpersite_exact = EnPerSite_exact(yamlfile['n_vis'], PBCflag = 1, J1 = J1_, J2 = J2_, hz = 0, hx = 0)/(yamlfile['n_vis']*1.0)
    else:
        enpersite_exact = None
#    EnPerSite_exact = 4 * (-np.log(2) + 0.25)

    plt.axhline(enpersite_exact ,linestyle = '--', color = 'black')
    plt.title("GS Energy of the 1d J1J2 model with J1 = %.2f, J2 = %.2f"%(J1_, J2_))
    plt.xlabel("#iteration")
    plt.ylabel("GS  Energy")
    plt.ylim(enpersite_exact - 0.3, np.real(EnPerSite[0]) + 0.3)
    plt.show()

    plt.plot(np.array(overlaps))
    plt.ylim(ymax =1)
    plt.title("The overlap between the ansatz and the exact GS wave function")
    plt.xlabel("# iteration")
    plt.ylabel("Overlap")
    plt.show()
