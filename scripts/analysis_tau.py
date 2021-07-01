from coffea import hist
import uproot
from coffea.util import awkward
from coffea.analysis_objects import JaggedCandidateArray
import coffea.processor as processor
from coffea.processor import run_parsl_job
from coffea.processor.parsl.parsl_executor import parsl_executor
from coffea.lookup_tools import extractor
from coffea.lumi_tools import LumiMask
import numpy as np
import sys
import re
import h5py as h5
from optparse import OptionParser
import parsl
import os
from parsl.configs.local_threads import config
from parsl.providers import LocalProvider,SlurmProvider
from parsl.channels import LocalChannel,SSHChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_hostname
np.set_printoptions(threshold=sys.maxsize)

def Writeh5(output, name,folder):
    if not os.path.exists(folder):
        os.makedirs(folder)    
    with h5.File(os.path.join(folder,'{0}.h5'.format(name)), "w") as fh5:
        dset = fh5.create_dataset("data", data=output['data'])
        dset = fh5.create_dataset("label", data=output['label'])
        dset = fh5.create_dataset("pid", data=output['pid'])

class Channel():
    def __init__(self, name):
        self.name = name
        self.pfs=None
        self.df=None
        self.decay = None

    
        
    def datah5(self):         
        ''' Data for training'''
        points = opt.points -2 #Muons first
        nparts = self.pfs.counts        

        data = np.zeros((self.pfs.size,points,13))
        data_muon = np.zeros((self.pfs.size,2,13))
        label = np.zeros((self.pfs.size,points))
        label_muon = 5*np.ones((self.pfs.size,2))
        
        # pid = np.array(
        #     (1*((self.decay==10) | (self.decay==11)) + 
        #      2*((self.decay==-2) | (self.decay==-1)) + 
        #      3*((self.decay==0) | (self.decay==1))
        #  ).tolist()
        # )



        data_muon[:,0,0]+= np.array((self.df['JpsiTau_mu1_eta']).tolist())
        data_muon[:,0,1]+= np.array((self.df['JpsiTau_mu1_phi']).tolist())
        data_muon[:,0,2]+= np.array((np.log(self.df['JpsiTau_mu1_pt'])).tolist())
        data_muon[:,0,3]+= np.array((self.df['JpsiTau_mu1_q']).tolist())
        data_muon[:,0,12]+= np.array((self.df['JpsiTau_mu2_q']>-2).tolist()) #true for muons

        data_muon[:,1,0]+= np.array((self.df['JpsiTau_mu2_eta']).tolist())
        data_muon[:,1,1]+= np.array((self.df['JpsiTau_mu2_phi']).tolist())
        data_muon[:,1,2]+= np.array((np.log(self.df['JpsiTau_mu2_pt'])).tolist())
        data_muon[:,1,3]+= np.array((self.df['JpsiTau_mu2_q']).tolist())
        
        data_muon[:,1,12]+= np.array((self.df['JpsiTau_mu2_q']>-2).tolist()) #true for muons




        data[:,:,0]+=self.pfs.eta.pad(points, clip=True).fillna(0).regular()
        data[:,:,1]+=self.pfs.phi.pad(points, clip=True).fillna(0).regular()
        data[:,:,2]+=np.log(self.pfs.pt).pad(points, clip=True).fillna(0).regular()
        data[:,:,3]+=self.pfs.charge.pad(points, clip=True).fillna(0).regular()
        data[:,:,4]+=self.pfs.pvAssociationQuality.pad(points, clip=True).fillna(0).regular()
        data[:,:,5]+=self.pfs.doca3d.pad(points, clip=True).fillna(0).regular()
        data[:,:,6]+=self.pfs.doca2d.pad(points, clip=True).fillna(0).regular()
        data[:,:,7]+=self.pfs.doca3de.pad(points, clip=True).fillna(0).regular()
        data[:,:,8]+=self.pfs.doca2de.pad(points, clip=True).fillna(0).regular()
        data[:,:,9]+=self.pfs.dz.pad(points, clip=True).fillna(0).regular()
        data[:,:,10]+=self.pfs.isAssociate.pad(points, clip=True).fillna(0).regular()
        data[:,:,11]+=self.pfs.nearDz.pad(points, clip=True).fillna(0).regular()
        # data[:,:,12]+=self.pfs.dnn_sig.pad(points, clip=True).fillna(0).regular()
        # data[:,:,13]+=self.pfs.dnn_1prong.pad(points, clip=True).fillna(0).regular()
        # data[:,:,14]+=self.pfs.dnn_otherB.pad(points, clip=True).fillna(0).regular()
        # data[:,:,15]+=self.pfs.dnn_pu.pad(points, clip=True).fillna(0).regular()


            
        label[:]+=np.array(
            (
                ((self.pfs.nprong==3) &  (self.pfs.isSignal==1)) + 
                2*((self.pfs.nprong!=3) & (self.pfs.isSignal==1)) + 
                3*((self.pfs.isBdecay==1) & (self.pfs.isSignal==0)) + 
                4*(self.pfs.isBdecay==0)
            ).pad(points, clip=True).fillna(0).regular()
        )
            
        return np.concatenate((data_muon,data),axis=1).tolist(), np.concatenate((label_muon,label),axis=1).tolist()


class SignalProcessor(processor.ProcessorABC):

    def __init__(self):
        dataset_axis = hist.Cat("dataset", "")        
        nparts_axis = hist.Bin("nparts", "N$_{particles}$ ", 100, 0, 20000)        

        axis_list = {
            'nparts':nparts_axis, 
        }

        self.channels = ['pfs']
        dict_accumulator = {}
        ML_accumulator = {
            'data': processor.list_accumulator(),
            'label': processor.list_accumulator(),
            'pid': processor.list_accumulator(),
            
        }
        dict_accumulator['ML'] = processor.dict_accumulator(ML_accumulator)

        for axis in axis_list:
            dict_accumulator["{}".format(axis)] = hist.Hist("Events", dataset_axis, axis_list[axis])
            
        

        for axis in axis_list:
            for channel in self.channels:
                dict_accumulator["{}_{}".format(axis,channel)] = hist.Hist("Events", dataset_axis, axis_list[axis])        
               
        dict_accumulator['cutflow']= processor.defaultdict_accumulator(int)

        self._accumulator = processor.dict_accumulator( dict_accumulator )

    @property
    def accumulator(self):
        return self._accumulator
    
        
    def process(self, df):

        output = self.accumulator.identity()
        dataset = df["dataset"]
        
        npf = df['JpsiTau_st_pt'] > 0
        npf = npf.sum()




        PFCands = JaggedCandidateArray.candidatesfromcounts(
            npf,
            pt = df['JpsiTau_st_pt'].content,
            eta = df['JpsiTau_st_eta'].content,
            phi = df['JpsiTau_st_phi'].content,
            mass = df['JpsiTau_st_mass'].content,
            charge = df['JpsiTau_st_charge'].content,
            nprong=df['JpsiTau_st_nprong'].content,
            isBdecay=df['JpsiTau_st_isBdecay'].content,
            isSignal=df['JpsiTau_st_isSignal'].content,
            doca3d=df['JpsiTau_st_doca3d'].content, 
            doca2d=df['JpsiTau_st_doca2d'].content,                                    
            doca3de = df['JpsiTau_st_doca3de'].content,
            doca2de = df['JpsiTau_st_doca2de'].content,
            dz = df['JpsiTau_st_dz'].content,
            isAssociate = df['JpsiTau_st_isAssociate'].content,
            nearDz = df['JpsiTau_st_near_dz'].content,
            pvAssociationQuality = df['JpsiTau_st_pvAssociationQuality'].content,

            # dnn_sig = df['JpsiTau_st_dnn'].content,
            # dnn_1prong = df['JpsiTau_st_dnn_1prong'].content,
            # dnn_otherB = df['JpsiTau_st_dnn_otherB'].content,
            # dnn_pu = df['JpsiTau_st_dnn_pu'].content,
            
            )
        output['cutflow']['all events'] += PFCands.size
        output['cutflow']['all PFcands'] += PFCands.counts.sum()
        
        
        #MC only
        # if 'data' not in dataset:
        #     decayid = df['JpsiTau_st_decayid']
        #     # noccurance = df['JpsiTau_st_noccurance'].content 
        # else:
        #     decayid = PFCands.pt[:,0] == -9
            
             
            
        

        pfs = Channel("pfs")
        for channel in [pfs]:
            channel.pfs = PFCands
            channel.df = df
            # channel.decay = decayid
            output['nparts_{}'.format(channel.name)].fill(dataset=dataset, nparts=channel.pfs.counts.flatten()) 



        for channel in [pfs]:
            data,label = channel.datah5()                    
                
            output['ML']['data']+= data
            output['ML']['label']+=label 
            output['ML']['pid']+=['TAU' in dataset]*len(label)
                
        
        return output

    def postprocess(self, accumulator):
        return accumulator

        

parser = OptionParser(usage="%prog [opt]  inputFiles")

parser.add_option("--samples",dest="samples", type="string", default='train', help="Specify which default samples to run [train/test/eval]. Default: data")
parser.add_option("-q", "--queue",  dest="queue", type="string", default="quick", help="Which queue to send the jobs. Default: %default")
parser.add_option("-p", "--nproc",  dest="nproc", type="long", default=1, help="Number of processes to use. Default: %default")
parser.add_option("--mem",  dest="mem", type="long", default=5000, help="Memory required in mb")
parser.add_option("--points", type="int", default=14, help="max number of particles to be used. Default %default")
parser.add_option("--cpu",  dest="cpu", type="long", default=4, help="Number of cpus to use. Default %default")
parser.add_option("--blocks",  dest="blocks", type="long", default=100, help="number of blocks. Default %default")
parser.add_option("--walltime",  dest="walltime", type="string", default="0:59:50", help="Max time for job run. Default %default")
parser.add_option("--chunk",  dest="chunk", type="long",  default=10000, help="Chunk size. Default %default")
parser.add_option("--maxchunk",  dest="maxchunk", type="long",  default=2e6, help="Maximum number of chunks. Default %default")
parser.add_option("--parsl",  dest="parsl", action="store_true",  default=False, help="Run without parsl. Default: False")
parser.add_option("--data",  action="store_true",  default=False, help="Use data as background. Default: False")

parser.add_option("--h5folder", type="string", default="../h5", help="Folder to store the h5 files. Default: %default")
parser.add_option("--year", type="int", default=17, help="max number of particles to be used. Default %default")

(opt, args) = parser.parse_args()
samples = opt.samples


if len(args) < 1:    

    if samples == 'train': 
        #file_name = os.path.join("../h5",'train_UL{}.txt'.format(opt.year))
        file_name = os.path.join("../h5",'train_UL{}_BKG.txt'.format(opt.year))
        if opt.data:
            file_name = os.path.join("../h5",'train_UL{}_data.txt'.format(opt.year))
    elif samples == 'test': 
        #file_name = os.path.join("../h5",'test_UL{}.txt'.format(opt.year))
        file_name = os.path.join("../h5",'test_UL{}_BKG.txt'.format(opt.year))
        if opt.data:
            file_name = os.path.join("../h5",'test_UL{}_data.txt'.format(opt.year))
    elif samples == 'eval':  #Don't really use it
        file_name = os.path.join('eval.txt')
    else:
        sys.exit("ERROR: You must specify what kind of dataset you want to run [--samples]")
    
    print('Loading sets from file')    
    files = []
    with open(os.path.join(file_name),'r') as fread:
        files = fread.readlines()
        files = [x.strip() for x in files] 
        idx = np.arange(len(files))
        np.random.shuffle(idx)
        files = np.array(files)[idx]
        
    #files = ['/pnfs/psi.ch/cms/trivcat/store/user/ytakahas/forVini/Signal_v2/0000/' + s for s in files]

else:
    files = args


fileset={}
for f in files:
    if 'sig' in f:
        name = 'TAU'
    elif 'BG' in f:
        name = 'BKG'
    elif 'Data' in f:
        name = 'DATA'
    elif '2017' in f:
        name = 'TAU_UL17'
    elif '2018' in f:
        name = 'TAU_UL18'
    else:
        print(f)
        sys.exit("ERROR: CANNOT IDENTIFY DATASET")
    if name in fileset:
        fileset[name].append(f)
    else:
        fileset[name] = [f]


nproc = opt.nproc
sched_options = '''
#SBATCH --cpus-per-task=%d
#SBATCH --mem=%d
''' % (opt.cpu,opt.mem) 

x509_proxy = '.x509up_u%s'%(os.getuid())
wrk_init = '''
export XRD_RUNFORKHANDLER=1
export X509_USER_PROXY=/t3home/%s/%s
'''%(os.environ['USER'],x509_proxy)



if not opt.parsl:
    output = processor.run_uproot_job(fileset,
                                      treename='ntuplizer/tree',
                                      processor_instance=SignalProcessor(),
                                      executor=processor.futures_executor,
                                      executor_args={'workers':nproc},
                                      maxchunks =opt.maxchunk,
                                      chunksize = opt.chunk
    )


else:
    print("id: ",os.getuid())
    print(wrk_init)
    #sched_opts = ''
    #wrk_init = ''


    slurm_htex = Config(
        executors=[
            HighThroughputExecutor(
                label="coffea_parsl_slurm",
                #worker_debug=True,
                address=address_by_hostname(),
                prefetch_capacity=0,  
                heartbeat_threshold=60,
                #cores_per_worker=1,
                #cores_per_worker=opt.cpu,
                max_workers=opt.cpu,
                provider=SlurmProvider(
                    channel=LocalChannel(),
                    launcher=SrunLauncher(),
                    init_blocks=opt.blocks,
                    min_blocks = opt.blocks-20,                     
                    max_blocks=opt.blocks+20,
                    exclusive  = False,
                    parallelism=1,
                    nodes_per_block=1,
                    #cores_per_node = opt.cpu,
                    partition=opt.queue,
                    scheduler_options=sched_options,   # Enter scheduler_opt if needed
                    worker_init=wrk_init,         # Enter worker_init if needed
                    walltime=opt.walltime
                ),
            )
        ],
        initialize_logging=False,
        #app_cache = True,
        retries=5,
        strategy=None,
    )

    dfk = parsl.load(slurm_htex)


    output = processor.run_uproot_job(fileset,
                                      treename='ntuplizer/tree',
                                      processor_instance=SignalProcessor(),
                                      executor=processor.parsl_executor,
                                      executor_args={'config':None, 'flatten': False,'compression':4},
                                      chunksize = opt.chunk,
                                      maxchunks =opt.maxchunk,
    )

    #np.save('test.npy', output)
for flow in output['cutflow']:
    print(flow, output['cutflow'][flow])

Writeh5(output['ML'],"{}_{}".format(opt.samples,name),os.path.join(opt.h5folder))
    
if opt.parsl:
    parsl.dfk().cleanup()
    parsl.clear()
