import numpy as np
from sympy import *
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy import signal
import scipy as sy
from numpy.fft import fft, fftshift
import seaborn as sns



def plot_with_labels(upxvalues, downxvalues, up, down, user, index):
    
    '''
    Plots the X, Y, Z axis from the accelerometer
            Parameters:
                    upxvalues (list): ordenadas para impressão de up (labels a colocar em cima)
                    downxvalues (list): ordenadas para impressão de down (labels a colocar em baixo)
                    up (list): lista com as labels das atividades
                    down (list): list com as labels das atividades
                    user (pandas.dataFrame): dataFrame com os dados do user a imprimir.
                    index (int): numero da experiência, de modo a imprimir o Numero do Sujeito, bem como da sua experiência
    '''
    int_to_label = {1: "W",       
                    2: "W_U",  
                    3: "W_D",
                    4: "SIT",           
                    5: "STAND",         
                    6: "LAY",           
                    7: "STAND_TO_SIT",     
                    8: "SIT_TO_STAND",      
                    9: "SIT_TO_LIE",       
                    10: "LIE_TO_SIT",        
                    11: "STAND_TO_LIE",     
                    12: "LIE_TO_STAND",  
    }
    
    figure, subplots = plt.subplots(nrows=3, ncols=1, figsize=(20,8))
    figure.suptitle(f"Valores Obtidos Pelo Acelerómetro na Experiência {index%2+1} do Sujeito {index//2+1}", fontsize = 'xx-large')

    #X
    subplots[0].plot( user['Time (min)'], user['X'], lw = 0.3)
    subplots[0].set_xlabel("Time (min)")
    subplots[0].set_ylabel("ACC_X")
    subplots[0].set_xlim(0, max(user['Time (min)']))
    
    #Y
    subplots[1].plot( user['Time (min)'], user['Y'], lw = 0.3)
    subplots[1].set_xlabel("Time (min)")
    subplots[1].set_ylabel("ACC_Y")
    subplots[1].set_xlim(0, max(user['Time (min)']))
    
    #Z
    subplots[2].plot( user['Time (min)'], user['Z'], lw = 0.3)
    subplots[2].set_xlabel("Time (min)")
    subplots[2].set_ylabel("ACC_Z")
    subplots[2].set_xlim(0, max(user['Time (min)']))
        
    for i in range(len(upxvalues)):
        yv = 0.95 if i % 2 == 0 else 0.90
        frase = int_to_label[up[i]]
        xv = upxvalues[i]/len(user)
        
        subplots[0].annotate(xycoords = 'axes fraction', text = frase, xy = (xv,yv), textcoords = subplots[0].transAxes, xytext = (xv,yv), fontweight='bold')
        subplots[1].annotate(xycoords = 'axes fraction', text = frase, xy = (xv,yv), textcoords = subplots[1].transAxes, xytext = (xv,yv), fontweight='bold')
        subplots[2].annotate(xycoords = 'axes fraction', text = frase, xy = (xv,yv), textcoords = subplots[2].transAxes, xytext = (xv,yv), fontweight='bold')        
    yv = 0.01 
    for i in range(len(downxvalues)):
        frase = int_to_label[down[i]]
        xv = downxvalues[i]/len(user)
        
        subplots[0].annotate(xycoords = 'axes fraction', text = frase, xy = (xv,yv), textcoords = subplots[0].transAxes, xytext = (xv,yv), fontweight='bold')
        subplots[1].annotate(xycoords = 'axes fraction', text = frase, xy = (xv,yv), textcoords = subplots[1].transAxes, xytext = (xv,yv), fontweight='bold')
        subplots[2].annotate(xycoords = 'axes fraction', text = frase, xy = (xv,yv), textcoords = subplots[2].transAxes, xytext =(xv,yv), fontweight='bold')
    figure.tight_layout()


#uma segmentação das informações do user, num intervalo [*xi*, *xf*[
def ufrag(user, ni, nf):
    return user.iloc[ni:nf].reset_index().drop("index", axis = 1)


### Dá plot às informações dos 3 eixos de uma fragmentação do utilizador nos subplots *subplots*, com a cor *c*
## Para ajudar a análise também dá plot a uma linha com a média dos valores, bem como da média +/- 3* desvio padrão
def plot_walk(subplots, user, c):
    # subplots -- array com 3 subplots
    # c -- string com a cor
    # user -- DataFrame com o user
    xmean = user['X'].mean()
    xstd = user['X'].std()
    user.plot( x = 'Time (min)', y = 'X', ax = subplots[0], color = c )
    subplots[0].plot( user['Time (min)'], np.full( (len(user['X']), ), xmean ), 'k:')
    subplots[0].plot( user['Time (min)'], np.full( (len(user['X']), ), xmean + 3*xstd ), 'r--')
    subplots[0].plot( user['Time (min)'], np.full( (len(user['X']), ), xmean - 3*xstd ), 'r--')
    subplots[0].set_xlabel("Time (min)")
    subplots[0].set_ylabel("ACC_X")
    
    
    ymean = user['Y'].mean()
    ystd = user['Y'].std()
    user.plot( x = 'Time (min)', y = 'Y', ax = subplots[1], color = c )
    subplots[1].plot( user['Time (min)'], np.full( (len(user['Y']), ), ymean ), 'k:')
    subplots[1].plot( user['Time (min)'], np.full( (len(user['Y']), ), ymean + 3*ystd ), 'r--')
    subplots[1].plot( user['Time (min)'], np.full( (len(user['Y']), ), ymean - 3*ystd ), 'r--')
    subplots[1].set_xlabel("Time (min)")
    subplots[1].set_ylabel("ACC_Y")
    
    zmean = user['Z'].mean()
    zstd = user['Z'].std()
    user.plot( x = 'Time (min)', y = 'Z', ax = subplots[2], color = c )
    subplots[2].plot( user['Time (min)'], np.full( (len(user['Z']), ), zmean ), 'k:')
    subplots[2].plot( user['Time (min)'], np.full( (len(user['Z']), ), zmean + 3*zstd ), 'r--')
    subplots[2].plot( user['Time (min)'], np.full( (len(user['Z']), ), zmean - 3*zstd ), 'r--')
    subplots[2].set_xlabel("Time (min)")
    subplots[2].set_ylabel("ACC_Z")


### Dá plot às informações nos eixos das fragmentações do utilizador
def plot_user_activity(user_walks, title):
    # user_walks -- array com as fragmentações dos dados do utilizador // não deve exceder 4 fragmentações.
    figure, subplots = plt.subplots(nrows = 6, ncols= 2, figsize = (15,10))
    colors = ['royalblue', 'maroon', 'darkorange', 'olive']
    
    for i in range(len(user_walks)):
        if i < 2:
            plot_walk(subplots[:3,i], user_walks[i], colors[i] )
            subplots[0,i].set_title(f"{title} {i+1}")
        else:
            plot_walk(subplots[3:,i-2], user_walks[i], colors[i] )
            subplots[3,i-2].set_title(f"{title} {i+1}")
    figure.tight_layout()


def set_between_std(user):
    new = user.copy()
    xmean = new['X'].mean()
    xstd = new['X'].std()
    new['X'] = new['X'].map(lambda x: (xmean + 2*xstd) if (x > xmean + 3*xstd) 
                              else ( (xmean - 2*xstd) if (x < xmean - 3*xstd) 
                                    else x ) )

    ymean = new['Y'].mean()
    ystd = new['Y'].std()
    new['Y'] = new['Y'].map(lambda x: (ymean + 2*ystd) if (x > ymean + 3*ystd) 
                              else ( (ymean - 2*ystd) if (x < ymean - 3*ystd) 
                                    else x ) )
    
    zmean = new['Z'].mean()
    zstd = new['Z'].std()
    new['Z'] = new['Z'].map(lambda x: (zmean + 2*zstd) if (x > zmean + 3*zstd) 
                              else ( (zmean - 2*zstd) if (x < zmean - 3*zstd) 
                                    else x ) )
    return new


def detrend_user_walk(user):
    new = user.copy()
    new['X'] = signal.detrend(user['X'])
    new['Y'] = signal.detrend(user['Y'])
    new['Z'] = signal.detrend(user['Z'])
    return new


def plot_activity_dft(user_frag_act, N, user, exp, title, zeros, percent, wind_title, plotit):
    fs = 50
    if N%2==0:
        f = np.linspace( -fs/2, fs/2 - fs/N, N) 
    else:
        f = np.linspace( -fs/2 + fs/2/N, fs/2 - fs/2/N, N)
    
    if wind_title.lower()=='blackman':
        window = np.blackman(N)
    elif wind_title.lower()=='hamming':
        window = np.hamming(N)
    elif wind_title.lower()=='hann':
        window = signal.windows.hann(N)
    elif wind_title.lower()=='rect':
        window = np.ones(f.shape)
    elif wind_title.lower()=='black_harris':
        window = signal.windows.blackmanharris(N)
    
    dfts = []
    dfts.append( np.abs( fftshift( fft( user_frag_act['X'] * window ) ) ) ) 
    dfts.append( np.abs( fftshift( fft( user_frag_act['Y'] * window ) ) ) )
    dfts.append( np.abs( fftshift( fft( user_frag_act['Z'] * window ) ) ) )
    
    if plotit:
        figure, subplots = plt.subplots(nrows = 1, ncols= 3, figsize = (20,5))
        figure.suptitle(f'DFT de {title} - Experiência {exp} do User {user} -- Window: {wind_title}', fontsize = 'xx-large')
    
        if zeros:
            for i in range(3):
                dfts[i][ dfts[i]<np.max(dfts[i])*percent ] = 0
        else:
            for i in range(3):
                subplots[i].plot(f, np.full((len(f),),np.max(dfts[i])*percent), 'r')
        
        for i in range(3):
            marker, stem, base = subplots[i].stem(f, dfts[i])
            stem.set_linewidth(0.8)
            plt.setp(marker, markersize = 2)

        subplots[0].set_xlabel("Axis X Frequency (Hz)")
        subplots[1].set_xlabel("Axis Y Frequency (Hz)")
        subplots[2].set_xlabel("Axis Z Frequency (Hz)")
        subplots[0].set_ylabel("Magnitude")
        subplots[1].set_ylabel("Magnitude")
        subplots[2].set_ylabel("Magnitude")
    
        figure.tight_layout()

    if not plotit:
        for i in range(3):
            dfts[i][ dfts[i]<np.max(dfts[i])*percent ] = 0
            
    #Calcular o Cm para determinar se a atividade é estática pu dinâmica
    #atividade dinâmica -> valores > 0.01
    #amplitude     
    #Cm =
    aux = {'X':[], 'Y':[], 'Z':[]}
    
    for eixo in range(3):
        freqs = np.unique(np.round(np.abs(f[np.where(dfts[eixo])]), 8))
        magnitude = np.unique(np.round(dfts[eixo][dfts[eixo]>0], 8))
        for i in range(len(freqs)):
            if freqs[i] == 0:
                Cm = magnitude[i] / N
            else:
                Cm = 2*magnitude[i] / N
            if eixo == 0:
                aux['X'].append(Cm)
            if eixo == 1:
                aux['Y'].append(Cm)
            else:
                aux['Z'].append(Cm)
    #print(aux)
    Cms = pd.DataFrame.from_dict(aux, orient='index').transpose()
    #print(Cms)
    #print(Cms.describe())
    
    if zeros and not plotit:
        print(f'user{user}_{exp}')
        print(f'X: Freqs - {np.unique(np.round(np.abs(f[np.where(dfts[0])]), 8))}, Magnitude - {np.unique(np.round(dfts[0][dfts[0]>0], 8))}')
        print(f'Y: Freqs - {np.unique(np.round(np.abs(f[np.where(dfts[1])]), 8))}, Magnitude - {np.unique(np.round(dfts[0][dfts[0]>0], 8))}')
        print(f'Z: Freqs - {np.unique(np.round(np.abs(f[np.where(dfts[2])]), 8))}, Magnitude - {np.unique(np.round(dfts[0][dfts[0]>0], 8))}')
        print('------------------')
    
    return np.unique(np.round(np.abs(f[np.where(dfts[0])]), 8)), np.unique(np.round(np.abs(f[np.where(dfts[1])]), 8)), np.unique(np.round(np.abs(f[np.where(dfts[2])]), 8))


def get_frequencies_from_activities(activity_array, percent, window):
    x_total = []
    y_total = []
    z_total = []
    aux = {}
    
    for a in activity_array:
        if len(a)<5:
            for i in range(len(a)):
                x, y, z = plot_activity_dft(a[i], len(a[i]), 0, 0, '', False, percent, window, False)
                for j in x:
                    x_total.append(j)
                for j in y:
                    y_total.append(j)
                for j in z:
                    z_total.append(j)
        else:
            x, y, z = plot_activity_dft(a, len(a), 0, 0, '', False, percent, window, False)
            for j in x:
                x_total.append(j)
            for j in y:
                y_total.append(j)
            for j in z:
                z_total.append(j)
    
    aux = {'X':x_total, 'Y':y_total, 'Z':z_total}
    Hz_pd = pd.DataFrame.from_dict(aux, orient='index').transpose()
    Hz_pd = Hz_pd[::][Hz_pd[::]<2.5]
    Hz_pd = Hz_pd[::][Hz_pd[::]>0]

    return Hz_pd


def stft(user, janela, media = 150):
    miniNs = janela
    overlap = miniNs//2

    fs = 50
    if miniNs%2==0:
        f = np.linspace( -fs/2, fs/2 - fs/2/miniNs, miniNs) 
    else:
        f = np.linspace( -fs/2 + fs/2/miniNs, fs/2 - fs/2/miniNs, miniNs)
    
    freqs = np.array([])
    window = np.hanning(miniNs)
    times = np.array([])
    cms_max = np.array([])
    mean_var = np.array([])
    
    N = len(user['Z'])
    
    for i in range(0, N-miniNs + 1, miniNs - overlap):
        Cms = np.array([])
        
        seccao = signal.detrend(user['Z'][i:i+miniNs])
        
        media_movel = get_media_movel(user['Z'], [i,i+miniNs],media)
        
        var_in_interval = abs(media_movel[-1] - media_movel[0])
        
        mean_var = np.append( mean_var, [var_in_interval])
        
        dft = np.abs(fftshift(fft(seccao*window)))           
        
        for j in range(len(dft[f>=0])):
            if f[f>=0][j]==0:
                Cms = np.append(Cms, [dft[f>=0][j]/miniNs])
            else:
                Cms = np.append(Cms, [dft[f>=0][j]*2/miniNs])

        cms_max = np.append( cms_max, [max(Cms)])
    
        dft[np.abs(f)>2.5] = 0
        dft[dft != max(dft)] = 0
        max_freq = np.unique( np.round( np.abs(f[ dft==dft.max() ]), 8 ) )
        
        freqs = np.append( freqs, [max_freq[0]])
        times = np.append(times, user['Time (min)'][(2*i+miniNs)//2])

    return times, freqs, cms_max, mean_var


#funcao para predicao da classe
def get_sample_predict(user, percent, n_interval = []):
    
    if n_interval == []:
        n_interval.append(0)
        n_interval.append(len(user))
    N = n_interval[1] - n_interval[0]
    
    #----------------#
    fs = 50
    if N%2==0:
        f = np.linspace( -fs/2, fs/2 - fs/2/N, N) 
    else:
        f = np.linspace( -fs/2 + fs/2/N, fs/2 - fs/2/N, N)
    
    #----------------#
    window = np.hanning(N)        
    seccao = detrend_user_walk(user.iloc[n_interval[0]:n_interval[1]])
    
    dfts = pd.DataFrame()
    dfts['X'] = np.abs(fftshift(fft( np.array(seccao['X']) * window)))
    dfts['Y'] = np.abs(fftshift(fft( np.array(seccao['Y']) * window)))
    dfts['Z'] = np.abs(fftshift(fft( np.array(seccao['Z']) * window)))

    dfts['X'][(np.abs(f)>=3) ] = 0
    dfts['Y'][(np.abs(f)>=3) ] = 0
    dfts['Z'][(np.abs(f)>=3) ] = 0
    
    x_max_freq = np.unique( np.round( np.abs(f[ dfts['X']==dfts['X'].max() ]), 8 ) )
    y_max_freq = np.unique( np.round( np.abs(f[ dfts['Y']==dfts['Y'].max() ]), 8 ) )
    z_max_freq = np.unique( np.round( np.abs(f[ dfts['Z']==dfts['Z'].max() ]), 8 ) )
    
    #----------------#
    if N < 250:
        if y_max_freq[0]>=1 or x_max_freq[0] >=1.2:
            return 3
        else:
            return 2
    
    if z_max_freq[0] >= 0.6:
        return 3
    else:
        return 1

# funcao para conversao das atividades (recebidas de input) para a classe da atividade
def convert_to_class( n ):
    if n[2] < 4:
        return 3  #Dinâmica
    elif 3<n[2]<7:
        return 1 # Estática
    return 2 #Transição

#obter o valor da média móvel do user no intervalo, utilizando os n_points anteriores
def get_media_movel(user_Z, interval, n_points):
    media_movel = []
    for i in range(interval[0], interval[1]+1):
        if interval[0] >= n_points:
            media_movel.append( (user_Z[i-n_points:i]/n_points).sum() )
        else:
            media_movel.append(np.mean(user_Z[interval[0]:interval[1]]))
    return media_movel


def get_user(user,exp):
    u = pd.DataFrame(data = np.loadtxt(f"Datasets/acc_exp0{exp}_user0{user}.txt"), columns = ['X','Y','Z'])
    fs = 50
    T = 1/fs
    u['Time (min)'] = np.arange(0, len(u) * T, T)/60
    
    return u