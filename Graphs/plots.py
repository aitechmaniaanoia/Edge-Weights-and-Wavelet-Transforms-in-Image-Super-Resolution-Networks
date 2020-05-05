import pandas
import numpy as np
import matplotlib.pyplot as plt
import os.path


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

dir_path = os.path.abspath(os.path.dirname(__file__))
save_dir = 'ALL2/'

# dwt_csv = 'experiments/SRFBN_WV_dwt/records/train_records.csv'
# swt_csv = 'experiments/SRFBN_WV_swt/records/train_records.csv'
# dwt_ycbcr_csv = 'experiments/SRFBN_WV_dwt_with_Ycbcr/records/train_records.csv'
# swt_ycbcr_csv = 'experiments/SRFBN_WV_swt_with_Ycbcr/records/train_records.csv'



# dwt_csv = os.path.join(dir_path,'../experiments/SRFBN_WV_dwt/records/train_records.csv')
# swt_csv = os.path.join(dir_path,'../experiments/SRFBN_WV_swt/records/train_records.csv')
# dwt_ycbcr_csv = os.path.join(dir_path,'../experiments/SRFBN_WV_dwt_with_Ycbcr/records/train_records.csv')
# swt_ycbcr_csv = os.path.join(dir_path,'../experiments/SRFBN_WV_swt_with_Ycbcr/records/train_records.csv')
res_v2 = os.path.join(dir_path,'../experiments/SRFBN_WL_in9f32_x4/records/train_records.csv')


# path = os.path.join(my_path, "../data/test.csv")
# print(my_path)

original_csv = os.path.join(dir_path,'../experiments/SRFBN_in3f32_x4/records/train_records.csv')
t005_w10_00_30 = os.path.join(dir_path,'../experiments/SRFBN_WL_in3f32_x4_t005_w10-00-30/records/train_records.csv')
t01_w07_00_03 = os.path.join(dir_path,'../experiments/SRFBN_WL_in3f32_x4_t01_w07-00-03/records/train_records.csv')
res = os.path.join(dir_path,'../experiments/SRFBN_WL_in12f32_x4_lr_1e-4/records/train_records.csv')

# files=[original_csv,dwt_csv,swt_csv,dwt_ycbcr_csv,swt_ycbcr_csv]
files=[original_csv,t005_w10_00_30,t01_w07_00_03,res,res_v2]#,dwt_csv,swt_csv,dwt_ycbcr_csv,swt_ycbcr_csv]
labels= ['original','wl, th=0.1,   w=0.7,0.3','wl, th=0.05, w=1.0,3.0','res','res-v2']#,'dwt','swt','dwt_ycbcr','swt_ycbcr']
# labels= ['original','dwt','swt','dwt_ycbcr','swt_ycbcr']
length = 100

psnr= np.zeros((len(files),length))
ssim= np.zeros((len(files),length))
val_loss= np.zeros((len(files),length))
train_loss= np.zeros((len(files),length))
x= np.linspace(0,100,100)
metrics = [psnr,ssim,val_loss,train_loss]
# files = np.asarray(names

for f in range(len(files)):
    for i in range (length):
        df = pandas.read_csv(files[f])
        psnr[f,i] = df.iloc[i]['psnr']
        ssim[f,i] = df.iloc[i]['ssim']
        val_loss[f,i] = df.iloc[i]['val_loss']
        train_loss[f,i] = df.iloc[i]['train_loss']


legend_prop = {'weight':'bold'}
for j in range(4):
    figure,ax = plt.subplots()
    for f in range(len(files)):
        y = metrics[j][f,:]
        ax.plot(x,y,label=labels[f])

    metric_name = namestr(metrics[j],globals())[0]

    ax.set_xlim(left=0)

    if (metric_name =='psnr' or metric_name == 'ssim'):
        plt.legend(prop=legend_prop,loc=4)
    else:
        plt.legend(prop=legend_prop,loc=1)  

    plt.xlabel('epochs',fontweight = 'bold' ,fontsize=12 )
    plt.ylabel(metric_name,fontweight = 'bold' ,fontsize=12)
    # plt.axis('equal')
    plt.grid()
    # plt.show()
    
    plt.savefig(os.path.join(dir_path,save_dir,namestr(metrics[j],globals())[0]))
    a=1

