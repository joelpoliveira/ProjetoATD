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
    
    xmin,xmax = subplots[0].get_xlim()
    
    for i in range(len(upxvalues)):
        yv = 0.95 if i % 2 == 0 else 0.90
        frase = up[i]
        xv = upxvalues[i]/xmax
        
        subplots[0].annotate(xycoords = 'axes fraction', text = frase, xy = (xv,yv), textcoords = subplots[0].transAxes, xytext = (xv,yv), fontweight='bold')
        subplots[1].annotate(xycoords = 'axes fraction', text = frase, xy = (xv,yv), textcoords = subplots[1].transAxes, xytext = (xv,yv), fontweight='bold')
        subplots[2].annotate(xycoords = 'axes fraction', text = frase, xy = (xv,yv), textcoords = subplots[2].transAxes, xytext = (xv,yv), fontweight='bold')
        
    yv = 0.01 
    for i in range(len(downxvalues)):
        frase = down[i]
        xv = downxvalues[i]/xmax
        
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
    new['X'] = new['X'].map(lambda x: (xmean + 2.5*xstd) if (x > xmean + 3*xstd) 
                              else ( (xmean - 2.5*xstd) if (x < xmean - 3*xstd) 
                                    else x ) )
    
    ymean = new['Y'].mean()
    ystd = new['Y'].std()
    new['Y'] = new['Y'].map(lambda x: (ymean + 2.5*ystd) if (x > ymean + 3*ystd) 
                              else ( (ymean - 2.5*ystd) if (x < ymean - 3*ystd) 
                                    else x ) )
    
    zmean = new['Z'].mean()
    zstd = new['Z'].std()
    new['Z'] = new['Z'].map(lambda x: (zmean + 2.5*zstd) if (x > zmean + 3*zstd) 
                              else ( (zmean - 2.5*zstd) if (x < zmean - 3*zstd) 
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
    
    """
    if zeros:
        #print(f'user{user}_{exp}')
        #print('X: Hz-', data[0][0], 'Magnitude:' , np.unique(np.round(dfts[0][dfts[0]>0], 8)))
        #print('Y: Hz-', data[1][0], 'Magnitude:' , np.unique(np.round(dfts[1][dfts[1]>0], 8)))
        #print('Z: Hz-', data[2][0], 'Magnitude:' , np.unique(np.round(dfts[2][dfts[2]>0], 8)))
        print('------------------')
    """
    return np.unique(np.round(np.abs(f[np.where(dfts[0])]), 8)), np.unique(np.round(np.abs(f[np.where(dfts[1])]), 8)), np.unique(np.round(np.abs(f[np.where(dfts[2])]), 8))


def get_frequencies_from_activities(activity_array, percent, window):
    x_total = []
    y_total = []
    z_total = []
    aux = {}
    
    for a in activity_array:
        if len(a)<5:
            for i in range(len(a)):
                x, y, z = plot_activity_dft(a[i], len(a[i]), 0, 0, '', True, percent, window, False)
                for j in x:
                    x_total.append(j)
                for j in y:
                    y_total.append(j)
                for j in z:
                    z_total.append(j)
        else:
            x, y, z = plot_activity_dft(a, len(a), 0, 0, '', True, percent, window, False)
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


def stft(user, janela, percent, flag_max_freq = 3):
    miniNs = janela
    overlap = miniNs//2

    fs = 50
    if miniNs%2==0:
        f = np.linspace( -fs/2, fs/2 - fs/2/miniNs, miniNs) 
    else:
        f = np.linspace( -fs/2 + fs/2/miniNs, fs/2 - fs/2/miniNs, miniNs)
    
    freqs = [np.array([])] * flag_max_freq
    window = np.hanning(miniNs)
    times = np.array([])
    cms_max = np.array([])
    
    N = len(user['Z'])
    
    for i in range(0, N-miniNs + 1, miniNs - overlap):
        Cms = np.array([])
        seccao = signal.detrend(user['Z'][i:i+miniNs])
        dft = np.abs(fftshift(fft(seccao)))
        
        for j in range(len(dft[f>=0])):
            if f[f>=0][j]==0:
                Cms = np.append(Cms, [dft[f>=0][j]/miniNs])
            else:
                Cms = np.append(Cms, [dft[f>=0][j]*2/miniNs])

        cms_max = np.append( cms_max, [max(Cms)])
        #c_max = max(dft)*(2 if np.any(f[dft==max(dft)]!=0) else 1)/miniNs
        #print(c_max)

        dft = np.abs(fftshift(fft(seccao*window)))
        dft[dft<np.max(dft)*percent] = 0
        
        j = -1
        while j >= -3 and abs(j)-1 < len(np.unique(np.round(np.abs(f[dft>0]), 8))):
            freqs[abs(j)-1] = np.append( freqs[abs(j)-1], np.unique(np.round(np.abs(f[dft>0]), 8))[j] )
            j-=1
        if j>=-3:
            while j >= -3:
                freqs[abs(j)-1] = np.append(freqs[abs(j)-1], [0])
                j-=1
        
        """
        for j in range(len(np.unique(np.round(np.abs(f[ dft>0 ]), 8)))):
            if j == flag_max_freq:
                break
            if np.unique(np.round(np.abs(f[ dft>0 ]), 8))[j] > 2.5:
                continue
            else:
                freqs[j] = np.append( freqs[j], np.unique(np.round(np.abs(f[ dft>0 ]), 8))[j] )
        

        max_len = max(map(lambda x: len(x), freqs))
        for j in range(len(freqs)):
            while len(freqs[j]) < max_len:
                freqs[j] = np.append(freqs[j], [-1])
        """

        times = np.append(times, user['Time (min)'][(2*i+miniNs)//2])

    return times, freqs, cms_max
