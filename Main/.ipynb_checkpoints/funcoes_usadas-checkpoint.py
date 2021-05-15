import numpy as np
from sympy import *
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy import signal
import scipy as sy
from numpy.fft import fft, fftshift
import seaborn as sns
import matplotlib
from scipy.stats import entropy

dark_background = False

if dark_background:
    sns.set_palette("Paired")
    sns.set_style("whitegrid")
else:
    sns.set_palette("deep")
    sns.set_style("darkgrid")

"""
\\\\
    DIVISÃO DE FUNÇÃO PARA SE ENCONTRAR INICÍO/FIM SEM DIFICULDADE
////
"""    

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
    figure, subplots = plt.subplots(nrows=3, ncols=1, figsize=(20,8), facecolor = 'black' if dark_background else 'white')
    figure.suptitle(f"Valores Obtidos Pelo Acelerómetro na Experiência {index%2+1} do Sujeito {index//2+1}", fontsize = 'xx-large', c='white' if dark_background else 'black')
    
    if dark_background:
        subplots[0].set_facecolor('k')
        subplots[1].set_facecolor('k')
        subplots[2].set_facecolor('k')
    #X
    subplots[0].plot( user['Time (min)'], user['X'], lw = 0.3)
    subplots[0].set_xlabel("Time (min)", c = 'white' if dark_background else 'black')
    subplots[0].set_ylabel("ACC_X", c = 'white' if dark_background else 'black')
    subplots[0].set_xlim(0, max(user['Time (min)']))
    
    subplots[0].tick_params(axis='x', colors='white' if dark_background else 'black')
    subplots[0].tick_params(axis='y', colors='white' if dark_background else 'black')
    
    #Y
    subplots[1].plot( user['Time (min)'], user['Y'], lw = 0.3)
    subplots[1].set_xlabel("Time (min)", c = 'white' if dark_background else 'black')
    subplots[1].set_ylabel("ACC_Y", c = 'white' if dark_background else 'black')
    subplots[1].set_xlim(0, max(user['Time (min)']))
    
    subplots[1].tick_params(axis='x', colors='white' if dark_background else 'black')
    subplots[1].tick_params(axis='y', colors='white' if dark_background else 'black')
    
    #Z
    subplots[2].plot( user['Time (min)'], user['Z'], lw = 0.3)
    subplots[2].set_xlabel("Time (min)", c = 'white' if dark_background else 'black')
    subplots[2].set_ylabel("ACC_Z", c = 'white' if dark_background else 'black')
    subplots[2].set_xlim(0, max(user['Time (min)']))
    
    subplots[2].tick_params(axis='x', colors='white' if dark_background else 'black')
    subplots[2].tick_params(axis='y', colors='white' if dark_background else 'black')
    
    xmin,xmax = subplots[0].get_xlim()
    
    for i in range(len(upxvalues)):
        yv = 0.95 if i % 2 == 0 else 0.90
        frase = up[i]
        xv = upxvalues[i]/xmax
        
        subplots[0].annotate(xycoords = 'axes fraction', text = frase, xy = (xv,yv), textcoords = subplots[0].transAxes, xytext = (xv,yv), fontweight='bold', c = 'white' if dark_background else 'black')
        subplots[1].annotate(xycoords = 'axes fraction', text = frase, xy = (xv,yv), textcoords = subplots[1].transAxes, xytext = (xv,yv), fontweight='bold', c = 'white' if dark_background else 'black')
        subplots[2].annotate(xycoords = 'axes fraction', text = frase, xy = (xv,yv), textcoords = subplots[2].transAxes, xytext = (xv,yv), fontweight='bold', c = 'white' if dark_background else 'black')
        
    yv = 0.01 
    for i in range(len(downxvalues)):
        frase = down[i]
        xv = downxvalues[i]/xmax
        
        subplots[0].annotate(xycoords = 'axes fraction', text = frase, xy = (xv,yv), textcoords = subplots[0].transAxes, xytext = (xv,yv), fontweight='bold', c = 'white' if dark_background else 'black')
        subplots[1].annotate(xycoords = 'axes fraction', text = frase, xy = (xv,yv), textcoords = subplots[1].transAxes, xytext = (xv,yv), fontweight='bold', c = 'white' if dark_background else 'black')
        subplots[2].annotate(xycoords = 'axes fraction', text = frase, xy = (xv,yv), textcoords = subplots[2].transAxes, xytext =(xv,yv), fontweight='bold', c = 'white' if dark_background else 'black')
    figure.tight_layout()

    
"""
\\\\
    DIVISÃO DE FUNÇÃO PARA SE ENCONTRAR INICÍO/FIM SEM DIFICULDADE
////
"""    

#uma segmentação das informações do user, num intervalo [*xi*, *xf*[
def ufrag(user, ni, nf):
    return user.iloc[ni:nf].reset_index().drop("index", axis = 1)


"""
\\\\
    DIVISÃO DE FUNÇÃO PARA SE ENCONTRAR INICÍO/FIM SEM DIFICULDADE
////
"""    

### Dá plot às informações dos 3 eixos de uma fragmentação do utilizador nos subplots *subplots*, com a cor *c*
## Para ajudar a análise também dá plot a uma linha com a média dos valores, bem como da média +/- 3* desvio padrão
def plot_walk(subplots, user, c, windowed):
    # subplots -- array com 3 subplots
    # c -- string com a cor
    # user -- DataFrame com o user
    if windowed:
        window = np.hanning(len(user['X']))
    else:
        window = np.ones((len(user)))
        
    xmean = user['X'].mean()
    user = user.copy()
    user['X']*=window
    user.plot( x = 'Time (min)', y = 'X', ax = subplots[0], color = c )
    subplots[0].plot( user['Time (min)'], np.full( (len(user['X']), ), xmean ), 'k:', label = "Média")
    subplots[0].set_xlabel("Time (min)", c = 'white' if dark_background else 'black')
    subplots[0].set_ylabel("ACC_X", c = 'white' if dark_background else 'black')
    subplots[0].tick_params(axis='x', colors='white' if dark_background else 'black')
    subplots[0].tick_params(axis='y', colors='white' if dark_background else 'black')
    subplots[0].legend()
    
    ymean = user['Y'].mean()
    user['Y']*=window
    user.plot( x = 'Time (min)', y = 'Y', ax = subplots[1], color = c )
    subplots[1].plot( user['Time (min)'], np.full( (len(user['Y']), ), ymean ), 'k:', label = "Média")
    subplots[1].set_xlabel("Time (min)", c = 'white' if dark_background else 'black')
    subplots[1].set_ylabel("ACC_Y", c = 'white' if dark_background else 'black')
    subplots[1].tick_params(axis='x', colors='white' if dark_background else 'black')
    subplots[1].tick_params(axis='y', colors='white' if dark_background else 'black')
    subplots[1].legend()

    
    zmean = user['Z'].mean()
    user['Z']*=window
    user.plot( x = 'Time (min)', y = 'Z', ax = subplots[2], color = c )
    subplots[2].plot( user['Time (min)'], np.full( (len(user['Z']), ), zmean ), 'k:', label = "Média")
    subplots[2].set_xlabel("Time (min)", c = 'white' if dark_background else 'black')
    subplots[2].set_ylabel("ACC_Z", c = 'white' if dark_background else 'black')
    subplots[2].tick_params(axis='x', colors='white' if dark_background else 'black')
    subplots[2].tick_params(axis='y', colors='white' if dark_background else 'black')
    subplots[2].legend()

    
"""
\\\\
    DIVISÃO DE FUNÇÃO PARA SE ENCONTRAR INICÍO/FIM SEM DIFICULDADE
////
"""    


### Dá plot às informações nos eixos das fragmentações do utilizador
def plot_user_activity(user_walks, title, windowed):
    # user_walks -- array com as fragmentações dos dados do utilizador // não deve exceder 4 fragmentações.
    figure, subplots = plt.subplots(nrows = 6 if len(user_walks) > 2 else 3, ncols= 2, figsize = (15,10), facecolor = 'black' if dark_background else 'white')
    colors = ['royalblue', 'maroon', 'darkorange', 'olive']
    
    if dark_background:
        for i in range(6 if len(user_walks) > 2 else 3):
            for j in range(2):
                subplots[i,j].set_facecolor('k')
    
    for i in range(len(user_walks)):
        if i < 2:
            plot_walk(subplots[:3,i], user_walks[i], colors[i], windowed )
            subplots[0,i].set_title(f"{title} {i+1}", color = 'white' if dark_background else 'black')
        else:
            plot_walk(subplots[3:,i-2], user_walks[i], colors[i], windowed )
            subplots[3,i-2].set_title(f"{title} {i+1}", color = 'white' if dark_background else 'black')
    figure.tight_layout()

    

"""
\\\\
    DIVISÃO DE FUNÇÃO PARA SE ENCONTRAR INICÍO/FIM SEM DIFICULDADE
////
"""    


def detrend_user_walk(user):
    new = user.copy()
    new['X'] = signal.detrend(user['X'])
    new['Y'] = signal.detrend(user['Y'])
    new['Z'] = signal.detrend(user['Z'])
    return new


"""
\\\\
    DIVISÃO DE FUNÇÃO PARA SE ENCONTRAR INICÍO/FIM SEM DIFICULDADE
////
"""    


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
    else:
        return 
    
    dfts = []
    dfts.append( np.abs( fftshift( fft( user_frag_act['X'] * window ) ) ) ) 
    dfts.append( np.abs( fftshift( fft( user_frag_act['Y'] * window ) ) ) )
    dfts.append( np.abs( fftshift( fft( user_frag_act['Z'] * window ) ) ) )
    
    if plotit:
        figure, subplots = plt.subplots(nrows = 1, ncols= 3, figsize = (20,5))
        figure.suptitle(f'DFT de {title} - Experiência {exp} do User {user} -- Window: {wind_title}', fontsize = 'xx-large')
        
        if dark_background:
            subplots[0].set_facecolor('k')
            subplots[1].set_facecolor('k')
            subplots[2].set_facecolor('k')
        
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
        

        subplots[0].set_xlabel("Axis X Frequency (Hz)", c = 'white'  if dark_background else 'black')
        subplots[1].set_xlabel("Axis Y Frequency (Hz)", c = 'white'  if dark_background else 'black')
        subplots[2].set_xlabel("Axis Z Frequency (Hz)", c = 'white'  if dark_background else 'black')
        
        subplots[0].set_ylabel("Magnitude", c = 'white'  if dark_background else 'black')
        subplots[1].set_ylabel("Magnitude", c = 'white'  if dark_background else 'black')
        subplots[2].set_ylabel("Magnitude", c = 'white'  if dark_background else 'black')
        
        subplots[0].tick_params(axis='x', colors='white' if dark_background else 'black')
        subplots[1].tick_params(axis='x', colors='white' if dark_background else 'black')
        subplots[2].tick_params(axis='x', colors='white' if dark_background else 'black')
        
        subplots[0].tick_params(axis='y', colors='white' if dark_background else 'black')
        subplots[1].tick_params(axis='y', colors='white' if dark_background else 'black')
        subplots[2].tick_params(axis='y', colors='white' if dark_background else 'black')
    
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
                Cm = (2*magnitude[i]) / N
            if eixo == 0:
                aux['X'].append(Cm)
            if eixo == 1:
                aux['Y'].append(Cm)
            else:
                aux['Z'].append(Cm)
    #print(aux)
    Cms = pd.DataFrame.from_dict(aux, orient='index').transpose()
    #print(Cms)
    print(Cms.describe())
    print(f"\n\nENTROPIAAAA:::{entropy(dfts[2])}")
    
    if zeros:
        #print(f'user{user}_{exp}')
        print('X: Hz -', np.unique(np.round(np.abs(f[np.where(dfts[0])]), 8)))
        print('Y: Hz-', np.unique(np.round(np.abs(f[np.where(dfts[1])]), 8)))
        print('Z: Hz-', np.unique(np.round(np.abs(f[np.where(dfts[2])]), 8)))
        print('------------------')
    
    return np.unique(np.round(np.abs(f[np.where(dfts[0])]), 8)), np.unique(np.round(np.abs(f[np.where(dfts[1])]), 8)), np.unique(np.round(np.abs(f[np.where(dfts[2])]), 8))

"""
\\\\
    DIVISÃO DE FUNÇÃO PARA SE ENCONTRAR INICÍO/FIM SEM DIFICULDADE
////
"""    


def get_frequencies_from_activities(activity_array, percent, window, user=0, exp=0):
    x_total = []
    y_total = []
    z_total = []
    aux = {}
    #print(activity_array)
    if type(activity_array)!=list:
            activity_array = [activity_array]
   
    
    for a in activity_array:
        if type(a)!=list:
            a = [a]
        for i in range(len(a)):
            x_a, y_a, z_a = plot_activity_dft(a[i], len(a[i]), 0, 0, '', True, percent, window, False)
        
            x_total = x_total + x_a.tolist()
            y_total = y_total + y_a.tolist()
            z_total = z_total + z_a.tolist()
        
            maxlen = np.max([len(x_total), len(y_total), len(z_total)])
        
            while (len(x_total)< maxlen): x_total.append(np.nan)
            while (len(y_total)< maxlen): y_total.append(np.nan)
            while (len(z_total)< maxlen): z_total.append(np.nan)
    aux = {'X':x_total, 'Y':y_total, 'Z':z_total}
    Hz_pd = pd.DataFrame(aux)
    Hz_pd = Hz_pd[::][Hz_pd[::]<2.5]
    #print("\n", Hz_pd)
    print(Hz_pd.describe())

    
"""
\\\\
    DIVISÃO DE FUNÇÃO PARA SE ENCONTRAR INICÍO/FIM SEM DIFICULDADE
////
"""        

def get_dft(user_frag_act, N, zeros, percent, plotit, wind_flag):
    fs = 50
    if N%2==0:
        f = np.linspace( -fs/2, fs/2 - fs/2/N, N) 
    else:
        f = np.linspace( -fs/2 + fs/2/N, fs/2 - fs/2/N, N)
    
    if wind_flag=='hann':
        window = signal.windows.hann(N)
    elif wind_flag=='hamming':
        window = signal.windows.hamming(N)
    elif wind_flag=='blackman':
        window = signal.windows.blackman(N)
    else:
        window = np.ones((N,))
        
    dfts = pd.DataFrame()
    dfts['Frequency (Hz)'] = f
    dfts['X'] = np.abs( fftshift( fft( user_frag_act['X'] * window ) ) )
    dfts['Y'] = np.abs( fftshift( fft( user_frag_act['Y'] * window ) ) )
    dfts['Z'] = np.abs( fftshift( fft( user_frag_act['Z'] * window ) ) )
    
    print(entropy(dfts['X']), entropy(dfts['Y']), entropy(dfts['Z']), sep = '\t'*7)
    
    figure, subplots = plt.subplots(nrows = 1, ncols= 3, figsize = (20,5), facecolor = 'black' if dark_background else 'white')
        
    if dark_background:
        subplots[0].set_facecolor('k')
        subplots[1].set_facecolor('k')
        subplots[2].set_facecolor('k')
        
    if zeros:
        for i in ['X', 'Y', 'Z']:
            dfts[i][ dfts[i]<np.max(dfts[i])*percent ] = 0
    else:
        j=0
        for i in ['X', 'Y', 'Z']:
            subplots[j].plot(f, np.full((N,), np.max(dfts[i])*percent), 'r')
            j+=1
    j=0
    for i in ['X','Y','Z']:
        marker, stem, base = subplots[j].stem(dfts['Frequency (Hz)'], dfts[i])
        stem.set_linewidth(0.8)
        plt.setp(marker, markersize = 2)
        j+=1
        

    subplots[0].set_xlabel("Axis X Frequency (Hz)", c = 'white'  if dark_background else 'black')
    subplots[1].set_xlabel("Axis Y Frequency (Hz)", c = 'white'  if dark_background else 'black')
    subplots[2].set_xlabel("Axis Z Frequency (Hz)", c = 'white'  if dark_background else 'black')
        
    subplots[0].set_ylabel("Magnitude", c = 'white'  if dark_background else 'black')
    subplots[1].set_ylabel("Magnitude", c = 'white'  if dark_background else 'black')
    subplots[2].set_ylabel("Magnitude", c = 'white'  if dark_background else 'black')
        
    subplots[0].tick_params(axis='x', colors='white' if dark_background else 'black')
    subplots[1].tick_params(axis='x', colors='white' if dark_background else 'black')
    subplots[2].tick_params(axis='x', colors='white' if dark_background else 'black')
        
    subplots[0].tick_params(axis='y', colors='white' if dark_background else 'black')
    subplots[1].tick_params(axis='y', colors='white' if dark_background else 'black')
    subplots[2].tick_params(axis='y', colors='white' if dark_background else 'black')
    
    figure.tight_layout()
    
    for i in ['X', 'Y', 'Z']:
        dfts[i][ dfts[i]<np.max(dfts[i])*percent ] = 0
    
    return dfts

"""
\\\\
    DIVISÃO DE FUNÇÃO PARA SE ENCONTRAR INICÍO/FIM SEM DIFICULDADE
////
"""    

def get_amps(user_dfts, N):            
    figure, subplots = plt.subplots(nrows = 1, ncols= 3, figsize = (25,5), facecolor = 'black' if dark_background else 'white')
        
    if dark_background:
        subplots[0].set_facecolor('k')
        subplots[1].set_facecolor('k')
        subplots[2].set_facecolor('k')
    
    amps = pd.DataFrame()
    print(N)
    amps['m'] = np.arange( (N//2) if N%2==0 else (N//2 + 1) )
    f = lambda x: x if x==0 else 2*x
    print(len(amps))
    amps['X'] = np.array(list(map( f, user_dfts['X'][ user_dfts['Frequency (Hz)'] >=0 ] )))/N
    amps['Y'] = np.array(list(map( f, user_dfts['Y'][ user_dfts['Frequency (Hz)'] >=0 ] )))/N
    amps['Z'] = np.array(list(map( f, user_dfts['Z'][ user_dfts['Frequency (Hz)'] >=0 ] )))/N
    
    j=0
    for i in ['X','Y','Z']:
        marker, stem, base = subplots[j].stem( amps['m'], amps[i])
        stem.set_linewidth(0.8)
        plt.setp(marker, markersize = 2)
        j+=1
    
    subplots[0].set_xlabel("Axis X Frequency (Hz)", c = 'white'  if dark_background else 'black')
    subplots[1].set_xlabel("Axis Y Frequency (Hz)", c = 'white'  if dark_background else 'black')
    subplots[2].set_xlabel("Axis Z Frequency (Hz)", c = 'white'  if dark_background else 'black')
        
    subplots[0].set_ylabel("Magnitude", c = 'white'  if dark_background else 'black')
    subplots[1].set_ylabel("Magnitude", c = 'white'  if dark_background else 'black')
    subplots[2].set_ylabel("Magnitude", c = 'white'  if dark_background else 'black')
        
    subplots[0].tick_params(axis='x', colors='white' if dark_background else 'black')
    subplots[1].tick_params(axis='x', colors='white' if dark_background else 'black')
    subplots[2].tick_params(axis='x', colors='white' if dark_background else 'black')
        
    subplots[0].tick_params(axis='y', colors='white' if dark_background else 'black')
    subplots[1].tick_params(axis='y', colors='white' if dark_background else 'black')
    subplots[2].tick_params(axis='y', colors='white' if dark_background else 'black')
    return amps