from sklearn.preprocessing import StandardScaler
import numpy as np 
import pandas as pd
from tensorflow.keras.utils import to_categorical

'''
columns= AA_score,Domain,dF,diso,[ASA_std,P(C),P(H),P(E)],[Sp,Polar,Vol,Hydrophobicity,Iso,P(Helix),P(Sheet)]
'''

aa_index = {'A':0,'R':1,'N':2,'D':3,'C':4,'Q':5,'E':6,'G':7,'H':8,'I':9,'L':10,'K':11,'M':12,'F':13,
           'P':14,'S':15,'T':16,'W':17,'Y':18,'V':19}

# 计算每个样本的aa_score
def aa_score(mutsites):
    aa_list = [[0.5,-0.5,89.3,115,0.305,1.29,1.08],
               [0,3,190.3,225,0.227,1,1.05],
               [0,0.2,122.4,160,0.322,0.81,0.85],
               [0,3,114.4,150,0.335,1.1,0.85],
               [0,-1,102.5,135,0.339,0.79,0.95],
               [0,0.2,146.9,180,0.306,1.07,0.95],
               [0,3,138.8,190,0.282,1.49,1.15],
               [0,0,63.8,75,0.352,0.63,0.55],
               [0.5,-0.5,157.5,195,0.215,1.33,1],
               [1.8,-1.8,163,175,0.278,1.05,1.05],
               [1.8,-1.8,163.1,170,0.262,1.31,1.25],
               [0,3,165.1,200,0.391,1.33,1.15],
               [1.3,-1.3,165.8,185,0.28,1.54,1.15],
               [2.5,-2.5,190.8,210,0.195,1.13,1.1],
               [0,0,121.6,145,0.346,0.63,0.71],
               [0,0.3,94.2,115,0.326,0.78,0.75],
               [0.4,-0.4,119.6,140,0.251,0.77,0.75],
               [3.4,-3.4,226.4,255,0.291,1.18,1.1],
               [2.3,-2.3,194.6,230,0.293,0.71,1.1],
               [1.5,-1.5,138.2,155,0.291,0.81,0.95]]
    
    ss = StandardScaler()
    aa_score = ss.fit_transform(aa_list) # -mean/std 标准化
    sum_dl = 0
    get_aa_score = []
    for mutsite in mutsites:
        deleted = aa_index[mutsite[0:1]]
        induced = aa_index[mutsite[-1:]]
        sum_dl = 0
        for j in range(len(aa_list[0])):
            d = aa_score[deleted][j] - aa_score[induced][j] # deleted - induced
            dlj = np.square(d)
            sum_dl += dlj
        get_aa_score.append(sum_dl/7)
    return np.array(get_aa_score)


def daaph7(mutsites):
    amino_dict = {
        'A': [-1.04, -1.17, -1.46, -0.17, -0.06,  1.57, -0.91],
        'C': [-2.39, -1.66, -1.99, -0.48, -0.09, -1.75, -1.79],
        'D': [ 1.46, -0.29, -0.4 ,  0.74, -0.11, -0.15,  1.96],
        'E': [ 0.33,  0.2 ,  0.13,  1.21, -0.1 ,  1.22, -0.03],
        'F': [ 2.01,  0.2 ,  0.13,  1.31, -0.1 ,  0.19,  1.52],
        'G': [ 0.7 ,  1.17,  1.13,  1.3 , -0.32,  0.19,  0.74],
        'H': [ 0.7 ,  1.27,  1.43,  0.48, -0.32, -0.38,  1.08],
        'I': [ 0.98,  2.35,  2.29,  1.76, -0.16,  0.42,  1.19],
        'K': [ 0.79, -0.59, -0.61, -0.22, -0.36, -0.83,  0.52],
        'L': [-1.01, -1.08, -1.14, -0.52, -0.3 , -0.95, -0.36],
        'M': [ 0.07,  1.17,  1.25, -1.49,  2.62,  0.88, -0.69],
        'N': [-0.4 ,  0.49,  0.53, -1.47,  2.18,  0.42, -0.47],
        'P': [ 0.75,  0.59,  0.48, -0.35,  0.85, -0.15, -0.14],
        'Q': [-0.71, -0.59, -0.52, -1.25, -1.89, -0.38, -1.24],
        'R': [-0.75, -0.2 ,  0.01, -1.12, -1.81,  1.57, -1.13],
        'S': [-0.71, -0.39, -0.43, -1.08,  0.18, -0.83, -1.02],
        'T': [-0.75,  0.1 ,  0.1 , -0.7 , -0.33,  0.88, -0.69],
        'V': [ 0.08,  0.49,  0.35,  0.75, -0.29,  1.11,  0.08],
        'W': [ 0.42, -1.66, -0.55,  0.24,  0.34, -1.75,  0.3 ],
        'Y': [-0.53, -0.39, -0.7 ,  1.05,  0.08, -1.29,  1.08]
    }


    daaph7_list = []
    for mutsite in mutsites:
        deleted = mutsite[0:1]
        induced = mutsite[-1:]
        aapmt = np.array(list(amino_dict[induced]),dtype=np.float)
        aapwt = np.array(list(amino_dict[deleted]),dtype=np.float)
        daaph7_list.append(aapmt-aapwt) #突变-野生

    return np.array(daaph7_list)



# 结构域特征 （共16个结构域）
def domain(mutsites):
    domain= {0 : np.arange(1,23), # 'SP'         
            1 : np.arange(23,386), # 'D1'
            2 : np.arange(386,764), # 'D2'
            3 : np.arange(764,865), # 'Dd'
            4 : np.arange(865,1271), # 'D3'
            5 : np.arange(1271,1480), # 'A1'
            6 : np.arange(1480,1673), # 'A2'
            7 : np.arange(1673,1873), # 'A3'
            8 : np.arange(1873,2255), # 'D4'
            9 : np.arange(2255,2334), # 'C1'
            10: np.arange(2334,2429), # 'C2'
            11: np.arange(2429,2497), # 'C3'
            12: np.arange(2497,2578), # 'C4'
            13: np.arange(2578,2647), # 'C5'
            14: np.arange(2647,2723), # 'C6'
            15: np.arange(2723,2814)} # 'CK'

    get_domain = []
    for mutsite in mutsites:
        site = mutsite[1:-1]  
        for key,value in domain.items():
            if int(site) in value:
                get_domain.append(key)
    
    oh_domain = to_categorical(get_domain)
    return np.array(oh_domain)

# 共进化特征
def get_blosum_line(deleted):
    f=open('blosum62.iij','r')
    lines=f.read().split('\n')
    line_idx=aa_index[wt]+2
    line=lines[line_idx]
    line_sp=line.split()
    blosum_line=list(map(float,line_sp[1:21]))
    blosum_array=np.array(blosum_line)
    return blosum_array

def get_dF(pssm_path,mutsites):
    dF_list = []
    for mutsite in mutsites:
        deleted = aa_index[mutsite[0:1]]
        site = int(mutsite[1:-1])   
        induced = aa_index[mutsite[-1:]]  
        with open(pssm_path,'r') as opf:
             lines=opf.read().split('\n')
             line_idx= site + 2 #pos is 1-based
             line=lines[line_idx].strip()
             line_sp=line.split()
             F_array=np.array(list(map(float, line_sp[22:42])))
             if np.all(F_array==0):
                 F_array=get_blosum_line(deleted)
                 F_array=F_array/10
             else:
                 F_array=F_array/100
             Fwt=F_array[deleted]
             Fmt=F_array[induced]
             dF=Fmt-Fwt
             dF_list.append(dF)
    return np.array(dF_list)

# 突变信息（SPOTD）
def get_spotd(mutsites,sptdir_path):
    get_diso = []
    for mutsite in mutsites:
        deleted = aa_index[mutsite[0:1]]
        site = int(mutsite[1:-1])
        induced = aa_index[mutsite[-1:]]
        sptod_path = sptdir_path + '/vWF_' + mutsite + '.spotds'
        with open(sptod_path, 'r') as opf:
            lines=opf.read().split('\n')
            mut_line=lines[site+1].rstrip() #pos is 2-baesd
            mut_line_sp=mut_line.split()
            diso = float(mut_line_sp[2])
            get_diso.append(diso)
    return np.array(get_diso)

# spd33
def get_spd33(mutsites,spd33_path):
    rnam1_std = "ACDEFGHIKLMNPQRSTVWY"
    ASA_std = (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
               185, 160, 145, 180, 225, 115, 140, 155, 255, 230)
    dict_rnam1_ASA = dict(zip(rnam1_std, ASA_std))
    ASA_SS_list = []
    for mutsite in mutsites:
        deleted = mutsite[0:1]
        site = int(mutsite[1:-1]) 
        with open(spd33_path, 'r') as opf:
            lines=opf.read().split('\n')
            mut_line=lines[site].rstrip() #pos is 1-baesd
            mut_line_sp=mut_line.split()
            line = list(map(float, mut_line_sp[3:13]))
            temp_ASA_std = dict_rnam1_ASA[deleted]
            ASA = [line[0]/temp_ASA_std]
            HCEprob = np.array(line[7:10])
            ASA_SS = np.concatenate([ASA,HCEprob])
            ASA_SS_list.append(ASA_SS)
    return np.array(ASA_SS_list)

def mix_feature(mutsites,pssm_path,sptdir_path,spd33_path):
    mix_fet = np.column_stack((aa_score(mutsites), \
              domain(mutsites),\
              get_dF(pssm_path,mutsites),\
              get_spotd(mutsites,sptdir_path),\
              get_spd33(mutsites,spd33_path),\
              daaph7(mutsites)))
    return mix_fet




