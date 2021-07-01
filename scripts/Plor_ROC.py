import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import os
import h5py
from sklearn import metrics

parser = OptionParser(usage="%prog [opt]  inputFiles")
parser.add_option("--folder", type="string", default='../h5', help="Specify the folder to load. Default: %default")
parser.add_option("--plot", type="string", default='../plots', help="Folder to save plots. Default: %default")
parser.add_option("--file", type="string", default='doca_tau_10N.h5', help="File to load. Default: %default")
(opt, args) = parser.parse_args()

titles = ['Against all','Against other tau','Against other B','Against PU']

fig, base = plt.subplots(2,2,dpi=150,sharex=True,sharey=True)
fig.tight_layout()
ititle = 0
for axis in base:
    for plot in axis:
        plot.set_xlabel("True Postive Rate")
        plot.set_ylabel("False Postive Rate")
        plot.set_title(titles[ititle])
        ititle+=1

plt.figure(figsize=(6, 10))
h5file = h5py.File(os.path.join(opt.folder,opt.file),'r')
pid = h5file['pid'][:]
doca = h5file['doca'][:]
dnn = h5file['DNN'][:]
dnn = dnn[:,:,1]

#Take zero padded out
dnn = dnn[pid>0]
doca = -doca[pid>0]
pid = pid[pid>0]


counter = 0
for xplot in range(2):
    for yplot in range(2):
        add_sel = (pid == counter+1)
        if counter==0:
            add_sel = (pid > 0)
        dnn_cat = dnn[(pid==1) | add_sel ]
        doca_cat = doca[(pid==1) | add_sel]
        pid_cat = pid[(pid==1) | add_sel]
        binary_pid = pid_cat==1
        

        fpr_dnn, tpr_dnn, thresholds = metrics.roc_curve(binary_pid.flatten(),dnn_cat.flatten(), pos_label=1)   
        fpr_doca, tpr_doca, thresholds = metrics.roc_curve(binary_pid.flatten(),doca_cat.flatten(), pos_label=1)
        with open("roc_dnn_lab{}.txt".format(counter),'w') as f:
            merged = np.concatenate((np.expand_dims(fpr_dnn,-1),np.expand_dims(tpr_dnn,-1)),axis=1)
            for value in merged:
                f.write(str(value)+'\n')

        with open("roc_doca_lab{}.txt".format(counter),'w') as f:
            merged = np.concatenate((np.expand_dims(fpr_doca,-1),np.expand_dims(tpr_doca,-1)),axis=1)
            for value in merged:
                f.write(str(value)+'\n')

        print(titles[counter])
        print("AUC DNN: {}".format(metrics.auc(fpr_dnn, tpr_dnn)))
        print("AUC DOCA: {}".format(metrics.auc(fpr_doca, tpr_doca)))
        p = base[xplot,yplot].semilogy(tpr_dnn, fpr_dnn,label='DNN')
        p = base[xplot,yplot].semilogy(tpr_doca, fpr_doca,label='DOCA3D')
        base[xplot,yplot].legend(loc="lower right")
        counter+=1

#plt.show()  




