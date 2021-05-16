import numpy as np
import pandas as pd
from functions_tomas import ufrag, detrend_user_walk


user1_1 = pd.DataFrame(data = np.loadtxt("Datasets/acc_exp01_user01.txt"), columns = ['X','Y','Z'])
user1_2 = pd.DataFrame(data = np.loadtxt("Datasets/acc_exp02_user01.txt"), columns = ['X','Y','Z'])
user2_1 = pd.DataFrame(data = np.loadtxt("Datasets/acc_exp03_user02.txt"), columns = ['X','Y','Z'])
user2_2 = pd.DataFrame(data = np.loadtxt("Datasets/acc_exp04_user02.txt"), columns = ['X','Y','Z'])
user3_1 = pd.DataFrame(data = np.loadtxt("Datasets/acc_exp05_user03.txt"), columns = ['X','Y','Z'])
user3_2 = pd.DataFrame(data = np.loadtxt("Datasets/acc_exp06_user03.txt"), columns = ['X','Y','Z'])
user4_1 = pd.DataFrame(data = np.loadtxt("Datasets/acc_exp07_user04.txt"), columns = ['X','Y','Z'])
user4_2 = pd.DataFrame(data = np.loadtxt("Datasets/acc_exp08_user04.txt"), columns = ['X','Y','Z'])

fs = 50 # Hz
T = 1/fs # sec

user1_1['Time (min)'] = np.arange(0, len(user1_1['X']) * T, T)/60
user1_2['Time (min)'] = np.arange(0, len(user1_2['X']) * T, T)/60 
user2_1['Time (min)'] = np.arange(0, len(user2_1['X']) * T, T)/60 
user2_2['Time (min)'] = np.arange(0, len(user2_2['X']) * T, T)/60 
user3_1['Time (min)'] = np.arange(0, len(user3_1['X']) * T, T)/60 
user3_2['Time (min)'] = np.arange(0, len(user3_2['X']) * T, T)/60 
user4_1['Time (min)'] = np.arange(0, len(user4_1['X']) * T, T)/60 
user4_2['Time (min)'] = np.arange(0, len(user4_2['X']) * T, T)/60


#---------------------------------------------------------------#
#------------------Fragment all activities----------------------#
#---------------------------------------------------------------#

#--------Walk--------#
unow = user1_1
walks_user1_1 = [ ufrag(unow, 7496, 8078), ufrag(unow, 8356, 9250), ufrag(unow, 9657, 10567), ufrag(unow, 10750, 11714)]

unow = user1_2
walks_user1_2 = [  ufrag(unow, 7624, 8252), ufrag(unow, 8618, 9576), ufrag(unow, 9991, 10927), ufrag(unow, 11311, 12282)]

unow = user2_1
walks_user2_1 = [ ufrag(unow, 8434, 9501), ufrag(unow, 9854, 10926)] 

unow = user2_2
walks_user2_2 = [ ufrag(unow, 7306, 8343), ufrag(unow, 8720, 9686) ]

unow = user3_1
walks_user3_1 = [ ufrag(unow, 8687, 9837), ufrag(unow, 10240, 11305) ]

unow = user3_2
walks_user3_2 = [ ufrag(unow, 8700, 9666), ufrag(unow, 10005, 10940) ]

unow = user4_1
walks_user4_1 = [ ufrag(unow, 8125, 9269), ufrag(unow, 9443, 10483) ]

unow = user4_2
walks_user4_2 = [ ufrag(unow, 7873, 8907), ufrag(unow, 9045, 10016) ]

walks_user1_1_detrended = []
for i in walks_user1_1:
    walks_user1_1_detrended.append( detrend_user_walk(i) )
    
walks_user1_2_detrended = []
for i in walks_user1_2:
    walks_user1_2_detrended.append( detrend_user_walk(i) )
    
walks_user2_1_detrended = []
for i in walks_user2_1:
    walks_user2_1_detrended.append( detrend_user_walk(i) )
    
walks_user2_2_detrended = []
for i in walks_user2_2:
    walks_user2_2_detrended.append( detrend_user_walk(i) )
    
walks_user3_1_detrended = []
for i in walks_user3_1:
    walks_user3_1_detrended.append( detrend_user_walk(i) )

walks_user3_2_detrended = []
for i in walks_user3_2:
    walks_user3_2_detrended.append( detrend_user_walk(i) )

walks_user4_1_detrended = []
for i in walks_user4_1:
    walks_user4_1_detrended.append( detrend_user_walk(i) )
    
walks_user4_2_detrended = []
for i in walks_user4_2:
    walks_user4_2_detrended.append( detrend_user_walk(i) )


#--------Walk up--------#
unow = user1_1
walks_up_user1_1 = [ ufrag(unow, 14069, 14699), ufrag(unow, 15712, 16377), ufrag(unow, 17298, 17970) ]

unow = user1_2
walks_up_user1_2 = [  ufrag(unow, 14128, 14783), ufrag(unow, 15920, 16598), ufrag(unow, 17725, 18425)]

unow = user2_1
walks_up_user2_1 = [ ufrag(unow, 12631, 13306), ufrag(unow, 14528, 15166), ufrag(unow, 16259, 16870) ]

unow = user2_2
walks_up_user2_2 = [ ufrag(unow, 11294, 11928), ufrag(unow, 12986, 13602), ufrag(unow, 14705, 15274) ]

unow = user3_1
walks_up_user3_1 = [ ufrag(unow, 14018, 14694), ufrag(unow, 15985, 16611), ufrag(unow, 17811, 18477), ufrag(unow, 19536, 20152) ]

unow = user3_2
walks_up_user3_2 = [ ufrag(unow, 12703, 13384), ufrag(unow, 14587, 15213), ufrag(unow, 16146, 16779) ]

unow = user4_1
walks_up_user4_1 = [ ufrag(unow, 12653, 13437), ufrag(unow, 14548, 15230), ufrag(unow, 16178, 16814) ]

unow = user4_2
walks_up_user4_2 = [ ufrag(unow, 11392, 11991), ufrag(unow, 12939, 13565), ufrag(unow, 14391, 15007) ]

walks_up_user1_1_detrended = []
for i in walks_up_user1_1:
    walks_up_user1_1_detrended.append( detrend_user_walk(i) )
    
walks_up_user1_2_detrended = []
for i in walks_up_user1_2:
    walks_up_user1_2_detrended.append( detrend_user_walk(i) )
    
walks_up_user2_1_detrended = []
for i in walks_up_user2_1:
    walks_up_user2_1_detrended.append( detrend_user_walk(i) )
    
walks_up_user2_2_detrended = []
for i in walks_up_user2_2:
    walks_up_user2_2_detrended.append( detrend_user_walk(i) )
    
walks_up_user3_1_detrended = []
for i in walks_up_user3_1:
    walks_up_user3_1_detrended.append( detrend_user_walk(i) )

walks_up_user3_2_detrended = []
for i in walks_up_user3_2:
    walks_up_user3_2_detrended.append( detrend_user_walk(i) )

walks_up_user4_1_detrended = []
for i in walks_up_user4_1:
    walks_up_user4_1_detrended.append( detrend_user_walk(i) )
    
walks_up_user4_2_detrended = []
for i in walks_up_user4_2:
    walks_up_user4_2_detrended.append( detrend_user_walk(i) )


#--------Walk down--------#
unow = user1_1
walks_down_user1_1 = [ ufrag(unow, 13191, 13846), ufrag(unow, 14869, 15492), ufrag(unow, 16530, 17153) ]

unow = user1_2
walks_down_user1_2 = [  ufrag(unow, 13129, 13379), ufrag(unow, 13495, 13927), ufrag(unow, 15037, 15684), ufrag(unow, 16847, 17471)]

unow = user2_1
walks_down_user2_1 = [ ufrag(unow, 11704, 12390), ufrag(unow, 13649, 14244), ufrag(unow, 15477, 16002) ]

unow = user2_2
walks_down_user2_2 = [ ufrag(unow, 10438, 11056), ufrag(unow, 12171, 12732), ufrag(unow, 13862, 14444) ]

unow = user3_1
walks_down_user3_1 = [ ufrag(unow, 15098, 15693), ufrag(unow, 16964, 17559), ufrag(unow, 18810, 19325) ]

unow = user3_2
walks_down_user3_2 = [ ufrag(unow, 11720, 12410), ufrag(unow, 13660, 14332), ufrag(unow, 15375, 15972) ]

unow = user4_1
walks_down_user4_1 = [ ufrag(unow, 11340, 11596), ufrag(unow, 11824, 12047), ufrag(unow, 13625, 14341), ufrag(unow, 15391, 16025) ]

unow = user4_2
walks_down_user4_2 = [ ufrag(unow, 10629, 11231), ufrag(unow, 12205, 12762), ufrag(unow, 13729, 14283) ]

walks_down_user1_1_detrended = []
for i in walks_down_user1_1:
    walks_down_user1_1_detrended.append( detrend_user_walk(i) )
    
walks_down_user1_2_detrended = []
for i in walks_down_user1_2:
    walks_down_user1_2_detrended.append( detrend_user_walk(i) )
    
walks_down_user2_1_detrended = []
for i in walks_down_user2_1:
    walks_down_user2_1_detrended.append( detrend_user_walk(i) )
    
walks_down_user2_2_detrended = []
for i in walks_down_user2_2:
    walks_down_user2_2_detrended.append( detrend_user_walk(i) )
    
walks_down_user3_1_detrended = []
for i in walks_down_user3_1:
    walks_down_user3_1_detrended.append( detrend_user_walk(i) )

walks_down_user3_2_detrended = []
for i in walks_down_user3_2:
    walks_down_user3_2_detrended.append( detrend_user_walk(i) )

walks_down_user4_1_detrended = []
for i in walks_down_user4_1:
    walks_down_user4_1_detrended.append( detrend_user_walk(i) )
    
walks_down_user4_2_detrended = []
for i in walks_down_user4_2:
    walks_down_user4_2_detrended.append( detrend_user_walk(i) )


#--------Stand--------#
unow = user1_1
stand_user1_1 = [ ufrag(unow, 250, 1232), ufrag(unow, 2360, 3374) ]

unow = user1_2
stand_user1_2 = [ ufrag(unow, 251, 1226), ufrag(unow, 2378, 3304) ]

unow = user2_1
stand_user2_1 = [ ufrag(unow, 298, 1398), ufrag(unow, 2770, 3904) ]

unow = user2_2
stand_user2_2 = [ ufrag(unow, 524, 1351), ufrag(unow, 2449, 3337)]

unow = user3_1
stand_user3_1 = [ ufrag(unow, 243, 1364), ufrag(unow, 2471, 3618)]

unow = user3_2
stand_user3_2 = [ ufrag(unow, 482, 1493), ufrag(unow, 2699, 3873)]

unow = user4_1
stand_user4_1 = [ ufrag(unow, 198, 1291), ufrag(unow, 2512, 3416) ]

unow = user4_2
stand_user4_2 = [ ufrag(unow, 230, 1292), ufrag(unow, 2574, 3438) ]

stand_user1_1_detrended = []
for i in stand_user1_1:
    stand_user1_1_detrended.append( detrend_user_walk(i) )
    
stand_user1_2_detrended = []
for i in stand_user1_2:
    stand_user1_2_detrended.append( detrend_user_walk(i) )
    
stand_user2_1_detrended = []
for i in stand_user2_1:
    stand_user2_1_detrended.append( detrend_user_walk(i) )
    
stand_user2_2_detrended = []
for i in stand_user2_2:
    stand_user2_2_detrended.append( detrend_user_walk(i) )
    
stand_user3_1_detrended = []
for i in stand_user3_1:
    stand_user3_1_detrended.append( detrend_user_walk(i) )

stand_user3_2_detrended = []
for i in stand_user3_2:
    stand_user3_2_detrended.append( detrend_user_walk(i) )

stand_user4_1_detrended = []
for i in stand_user4_1:
    stand_user4_1_detrended.append( detrend_user_walk(i) )
    
stand_user4_2_detrended = []
for i in stand_user4_2:
    stand_user4_2_detrended.append( detrend_user_walk(i) )

#--------SIT--------#
unow = user1_1
sit_user1_1 = [ ufrag(unow, 1393, 2194), ufrag(unow, 4736, 5667) ]

unow = user1_2
sit_user1_2 = [ ufrag(unow, 1433, 2221), ufrag(unow, 4620, 5452) ]

unow = user2_1
sit_user2_1 = [ ufrag(unow, 1686, 2627), ufrag(unow, 5418, 6190) ]

unow = user2_2
sit_user2_2 = [ ufrag(unow, 1512, 2309), ufrag(unow, 4491, 5301)]

unow = user3_1
sit_user3_1 = [ ufrag(unow, 1507, 2360), ufrag(unow, 5075, 6059)]

unow = user3_2
sit_user3_2 = [ ufrag(unow, 1684, 2515), ufrag(unow, 5409, 6410)]

unow = user4_1
sit_user4_1 = [ ufrag(unow, 1528, 2381), ufrag(unow, 5032, 5896) ]

unow = user4_2
sit_user4_2 = [ ufrag(unow, 1471, 2430), ufrag(unow, 4785, 5600) ]

sit_user1_1_detrended = []
for i in sit_user1_1:
    sit_user1_1_detrended.append( detrend_user_walk(i) )
    
sit_user1_2_detrended = []
for i in sit_user1_2:
    sit_user1_2_detrended.append( detrend_user_walk(i) )
    
sit_user2_1_detrended = []
for i in sit_user2_1:
    sit_user2_1_detrended.append( detrend_user_walk(i) )
    
sit_user2_2_detrended = []
for i in sit_user2_2:
    sit_user2_2_detrended.append( detrend_user_walk(i) )
    
sit_user3_1_detrended = []
for i in sit_user3_1:
    sit_user3_1_detrended.append( detrend_user_walk(i) )

sit_user3_2_detrended = []
for i in sit_user3_2:
    sit_user3_2_detrended.append( detrend_user_walk(i) )

sit_user4_1_detrended = []
for i in sit_user4_1:
    sit_user4_1_detrended.append( detrend_user_walk(i) )
    
sit_user4_2_detrended = []
for i in sit_user4_2:
    sit_user4_2_detrended.append( detrend_user_walk(i) )


#--------Lay--------#
unow = user1_1
lay_user1_1 = [ ufrag(unow, 3663, 4538), ufrag(unow, 5860, 6786) ]

unow = user1_2
lay_user1_2 = [ ufrag(unow, 3573, 4435), ufrag(unow, 5690, 6467) ]

unow = user2_1
lay_user2_1 = [ ufrag(unow, 4323, 5139), ufrag(unow, 6416, 7367) ]

unow = user2_2
lay_user2_2 = [ ufrag(unow, 3543, 4348), ufrag(unow,5456, 6417)]

unow = user3_1
lay_user3_1 = [ ufrag(unow, 3835,4850), ufrag(unow, 6211, 7378)]

unow = user3_2
lay_user3_2 = [ ufrag(unow, 4109, 5211), ufrag(unow, 6581, 7720)]

unow = user4_1
lay_user4_1 = [ ufrag(unow, 3749, 4816), ufrag(unow, 6121, 7109) ]

unow = user4_2
lay_user4_2 = [ ufrag(unow, 3758, 4574), ufrag(unow, 5836, 6681) ]

lay_user1_1_detrended = []
for i in lay_user1_1:
    lay_user1_1_detrended.append( detrend_user_walk(i) )
    
lay_user1_2_detrended = []
for i in lay_user1_2:
    lay_user1_2_detrended.append( detrend_user_walk(i) )
    
lay_user2_1_detrended = []
for i in lay_user2_1:
    lay_user2_1_detrended.append( detrend_user_walk(i) )
    
lay_user2_2_detrended = []
for i in lay_user2_2:
    lay_user2_2_detrended.append( detrend_user_walk(i) )
    
lay_user3_1_detrended = []
for i in lay_user3_1:
    lay_user3_1_detrended.append( detrend_user_walk(i) )

lay_user3_2_detrended = []
for i in lay_user3_2:
    lay_user3_2_detrended.append( detrend_user_walk(i) )

lay_user4_1_detrended = []
for i in lay_user4_1:
    lay_user4_1_detrended.append( detrend_user_walk(i) )
    
lay_user4_2_detrended = []
for i in lay_user4_2:
    lay_user4_2_detrended.append( detrend_user_walk(i) )


#--------Stand--------#
unow = user1_1
stand_sit_user1_1 = ufrag(unow, 1233, 1392)
stand_sit_user1_1_detrended = detrend_user_walk(stand_sit_user1_1)

unow = user1_2
stand_sit_user1_2 = ufrag(unow, 1227, 1432)
stand_sit_user1_2_detrended = detrend_user_walk(stand_sit_user1_2)

unow = user2_1
stand_sit_user2_1 = ufrag(unow, 1399, 1555)
stand_sit_user2_1_detrended = detrend_user_walk(stand_sit_user2_1)

unow = user2_2
stand_sit_user2_2 = ufrag(unow, 1352, 1511)
stand_sit_user2_2_detrended = detrend_user_walk(stand_sit_user2_2)

unow = user3_1
stand_sit_user3_1 = ufrag(unow, 1365, 1506)
stand_sit_user3_1_detrended = detrend_user_walk(stand_sit_user3_1)

unow = user3_2
stand_sit_user3_2 = ufrag(unow, 1494, 1683)
stand_sit_user3_2_detrended = detrend_user_walk(stand_sit_user3_2)

unow = user4_1
stand_sit_user4_1 = ufrag(unow, 1292, 1527)
stand_sit_user4_1_detrended = detrend_user_walk(stand_sit_user4_1)

unow = user4_2
stand_sit_user4_2 = ufrag(unow, 1293, 1470)
stand_sit_user4_2_detrended = detrend_user_walk(stand_sit_user4_2)


#--------Sit to stand--------#
unow = user1_1
sit_stand_user1_1 = ufrag(unow, 2195, 2359)
sit_stand_user1_1_detrended = detrend_user_walk(sit_stand_user1_1)

unow = user1_2
sit_stand_user1_2 = ufrag(unow,2222, 2377)
sit_stand_user1_2_detrended = detrend_user_walk(sit_stand_user1_2)

unow = user2_1
sit_stand_user2_1 = ufrag(unow, 2628, 2769)
sit_stand_user2_1_detrended = detrend_user_walk(sit_stand_user2_1)

unow = user2_2
sit_stand_user2_2 = ufrag(unow, 2310, 2448)
sit_stand_user2_2_detrended = detrend_user_walk(sit_stand_user2_2)

unow = user3_1
sit_stand_user3_1 = ufrag(unow, 2361, 2470)
sit_stand_user3_1_detrended = detrend_user_walk(sit_stand_user3_1)

unow = user3_2
sit_stand_user3_2 = ufrag(unow, 2516, 2698)
sit_stand_user3_2_detrended = detrend_user_walk(sit_stand_user3_2)

unow = user4_1
sit_stand_user4_1 = ufrag(unow, 2382, 2511)
sit_stand_user4_1_detrended = detrend_user_walk(sit_stand_user4_1)

unow = user4_2
sit_stand_user4_2 = ufrag(unow,2431, 2573)
sit_stand_user4_2_detrended = detrend_user_walk(sit_stand_user4_2)


#--------Stand to lie--------#
unow = user1_1
stand_lie_user1_1 = ufrag(unow, 3375, 3662)
stand_lie_user1_1_detrended = detrend_user_walk(stand_lie_user1_1)

unow = user1_2
stand_lie_user1_2 = ufrag(unow,3305, 3572)
stand_lie_user1_2_detrended = detrend_user_walk(stand_lie_user1_2)

unow = user2_1
stand_lie_user2_1 = ufrag(unow, 3905,4322)
stand_lie_user2_1_detrended = detrend_user_walk(stand_lie_user2_1)

unow = user2_2
stand_lie_user2_2 = ufrag(unow, 3338, 3542)
stand_lie_user2_2_detrended = detrend_user_walk(stand_lie_user2_2)

unow = user3_1
stand_lie_user3_1 = ufrag(unow, 3619, 3834)
stand_lie_user3_1_detrended = detrend_user_walk(stand_lie_user3_1)

unow = user3_2
stand_lie_user3_2 = ufrag(unow, 3874, 4108)
stand_lie_user3_2_detrended = detrend_user_walk(stand_lie_user3_2)

unow = user4_1
stand_lie_user4_1 = ufrag(unow, 3417,3748)
stand_lie_user4_1_detrended = detrend_user_walk(stand_lie_user4_1)

unow = user4_2
stand_lie_user4_2 = ufrag(unow,3439, 3757)
stand_lie_user4_2_detrended = detrend_user_walk(stand_lie_user4_2)


#--------Lie to sit--------#
unow = user1_1
lie_sit_user1_1 = ufrag(unow, 4539, 4735)
lie_sit_user1_1_detrended = detrend_user_walk(lie_sit_user1_1)

unow = user1_2
lie_sit_user1_2 = ufrag(unow,4436, 4619)
lie_sit_user1_2_detrended = detrend_user_walk(lie_sit_user1_2)

unow = user2_1
lie_sit_user2_1 = ufrag(unow, 5140,5417)
lie_sit_user2_1_detrended = detrend_user_walk(lie_sit_user2_1)

unow = user2_2
lie_sit_user2_2 = ufrag(unow, 4349,4490)
lie_sit_user2_2_detrended = detrend_user_walk(lie_sit_user2_2)

unow = user3_1
lie_sit_user3_1 = ufrag(unow, 4851, 5074)
lie_sit_user3_1_detrended = detrend_user_walk(lie_sit_user3_1)

unow = user3_2
lie_sit_user3_2 = ufrag(unow,5212, 5408)
lie_sit_user3_2_detrended = detrend_user_walk(lie_sit_user3_2)

unow = user4_1
lie_sit_user4_1 = ufrag(unow,4817,5031)
lie_sit_user4_1_detrended = detrend_user_walk(lie_sit_user4_1)

unow = user4_2
lie_sit_user4_2 = ufrag(unow, 4575,4784)
lie_sit_user4_2_detrended = detrend_user_walk(lie_sit_user4_2)


#--------Sit to lie--------#
unow = user1_1
sit_lie_user1_1 = ufrag(unow, 5668, 5859)
sit_lie_user1_1_detrended = detrend_user_walk(sit_lie_user1_1)

unow = user1_2
sit_lie_user1_2 = ufrag(unow, 5453, 5689)
sit_lie_user1_2_detrended = detrend_user_walk(sit_lie_user1_2)

unow = user2_1
sit_lie_user2_1 = ufrag(unow, 6191, 6415)
sit_lie_user2_1_detrended = detrend_user_walk(sit_lie_user2_1)

unow = user2_2
sit_lie_user2_2 = ufrag(unow, 5302, 5455)
sit_lie_user2_2_detrended = detrend_user_walk(sit_lie_user2_2)

unow = user3_1
sit_lie_user3_1 = ufrag(unow, 6060, 6210)
sit_lie_user3_1_detrended = detrend_user_walk(sit_lie_user3_1)

unow = user3_2
sit_lie_user3_2 = ufrag(unow, 6411, 6580)
sit_lie_user3_2_detrended = detrend_user_walk(sit_lie_user3_2)

unow = user4_1
sit_lie_user4_1 = ufrag(unow, 5897, 6120)
sit_lie_user4_1_detrended = detrend_user_walk(sit_lie_user4_1)

unow = user4_2
sit_lie_user4_2 = ufrag(unow, 5601, 5835)
sit_lie_user4_2_detrended = detrend_user_walk(sit_lie_user4_2)


#--------Lie to stand--------#
unow = user1_1
lie_stand_user1_1 = ufrag(unow, 6787, 6977)
lie_stand_user1_1_detrended = detrend_user_walk(lie_stand_user1_1)

unow = user1_2
lie_stand_user1_2 = ufrag(unow, 6468, 6709)
lie_stand_user1_2_detrended = detrend_user_walk(lie_stand_user1_2)

unow = user2_1
lie_stand_user2_1 = ufrag(unow, 7368, 7548)
lie_stand_user2_1_detrended = detrend_user_walk(lie_stand_user2_1)

unow = user2_2
lie_stand_user2_2 = ufrag(unow, 6418, 6598)
lie_stand_user2_2_detrended = detrend_user_walk(lie_stand_user2_2)

unow = user3_1
lie_stand_user3_1 = ufrag(unow, 7379, 7562)
lie_stand_user3_1_detrended = detrend_user_walk(lie_stand_user3_1)

unow = user3_2
lie_stand_user3_2 = ufrag(unow, 7721, 7887)
lie_stand_user3_2_detrended = detrend_user_walk(lie_stand_user3_2)

unow = user4_1
lie_stand_user4_1 = ufrag(unow, 7110, 7275)
lie_stand_user4_1_detrended = detrend_user_walk(lie_stand_user4_1)

unow = user4_2
lie_stand_user4_2 = ufrag(unow, 6682,6852)
lie_stand_user4_2_detrended = detrend_user_walk(lie_stand_user4_2)



#---------------------------------------------------------------#
#------Make arrays with a specific activity from all users------#
#---------------------------------------------------------------#

#------Walking------#
walks_user = []
walks_user.append(walks_user1_1_detrended)
walks_user.append(walks_user1_2_detrended)
walks_user.append(walks_user2_1_detrended)
walks_user.append(walks_user2_2_detrended)
walks_user.append(walks_user3_1_detrended)
walks_user.append(walks_user3_2_detrended)
walks_user.append(walks_user4_1_detrended)
walks_user.append(walks_user4_2_detrended)

walks_user_nd = []
walks_user_nd.append(walks_user1_1)
walks_user_nd.append(walks_user1_2)
walks_user_nd.append(walks_user2_1)
walks_user_nd.append(walks_user2_2)
walks_user_nd.append(walks_user3_1)
walks_user_nd.append(walks_user3_2)
walks_user_nd.append(walks_user4_1)
walks_user_nd.append(walks_user4_2)

#------Walking down------#
walks_down_user = []
walks_down_user.append(walks_down_user1_1_detrended)
walks_down_user.append(walks_down_user1_2_detrended)
walks_down_user.append(walks_down_user2_1_detrended)
walks_down_user.append(walks_down_user2_2_detrended)
walks_down_user.append(walks_down_user3_1_detrended)
walks_down_user.append(walks_down_user3_2_detrended)
walks_down_user.append(walks_down_user4_1_detrended)
walks_down_user.append(walks_down_user4_2_detrended)

#------Walking up------#
walks_up_user = []
walks_up_user.append(walks_up_user1_1_detrended)
walks_up_user.append(walks_up_user1_2_detrended)
walks_up_user.append(walks_up_user2_1_detrended)
walks_up_user.append(walks_up_user2_2_detrended)
walks_up_user.append(walks_up_user3_1_detrended)
walks_up_user.append(walks_up_user3_2_detrended)
walks_up_user.append(walks_up_user4_1_detrended)
walks_up_user.append(walks_up_user4_2_detrended)

#------Sit------#
sit_user = []
sit_user.append(sit_user1_1_detrended)
sit_user.append(sit_user1_2_detrended)
sit_user.append(sit_user2_1_detrended)
sit_user.append(sit_user2_2_detrended)
sit_user.append(sit_user3_1_detrended)
sit_user.append(sit_user3_2_detrended)
sit_user.append(sit_user4_1_detrended)
sit_user.append(sit_user4_2_detrended)

sit_user_nd = []
sit_user_nd.append(sit_user1_1)
sit_user_nd.append(sit_user1_2)
sit_user_nd.append(sit_user2_1)
sit_user_nd.append(sit_user2_2)
sit_user_nd.append(sit_user3_1)
sit_user_nd.append(sit_user3_2)
sit_user_nd.append(sit_user4_1)
sit_user_nd.append(sit_user4_2)

#------Lay------#
lay_user = []
lay_user.append(lay_user1_1_detrended)
lay_user.append(lay_user1_2_detrended)
lay_user.append(lay_user2_1_detrended)
lay_user.append(lay_user2_2_detrended)
lay_user.append(lay_user3_1_detrended)
lay_user.append(lay_user3_2_detrended)
lay_user.append(lay_user4_1_detrended)
lay_user.append(lay_user4_2_detrended)

lay_user_nd = []
lay_user_nd.append(lay_user1_1)
lay_user_nd.append(lay_user1_2)
lay_user_nd.append(lay_user2_1)
lay_user_nd.append(lay_user2_2)
lay_user_nd.append(lay_user3_1)
lay_user_nd.append(lay_user3_2)
lay_user_nd.append(lay_user4_1)
lay_user_nd.append(lay_user4_2)

#------Stand------#
stand_user = []
stand_user.append(stand_user1_1_detrended)
stand_user.append(stand_user1_2_detrended)
stand_user.append(stand_user2_1_detrended)
stand_user.append(stand_user2_2_detrended)
stand_user.append(stand_user3_1_detrended)
stand_user.append(stand_user3_2_detrended)
stand_user.append(stand_user4_1_detrended)
stand_user.append(stand_user4_2_detrended)

stand_user_nd = []
stand_user_nd.append(stand_user1_1)
stand_user_nd.append(stand_user1_2)
stand_user_nd.append(stand_user2_1)
stand_user_nd.append(stand_user2_2)
stand_user_nd.append(stand_user3_1)
stand_user_nd.append(stand_user3_2)
stand_user_nd.append(stand_user4_1)
stand_user_nd.append(stand_user4_2)

#------Stand to sit------#
stand_sit_user = []
stand_sit_user.append(stand_sit_user1_1_detrended)
stand_sit_user.append(stand_sit_user1_2_detrended)
stand_sit_user.append(stand_sit_user2_1_detrended)
stand_sit_user.append(stand_sit_user2_2_detrended)
stand_sit_user.append(stand_sit_user3_1_detrended)
stand_sit_user.append(stand_sit_user3_2_detrended)
stand_sit_user.append(stand_sit_user4_1_detrended)
stand_sit_user.append(stand_sit_user4_2_detrended)

#------Sit to stand------#
sit_stand_user = []
sit_stand_user.append(sit_stand_user1_1_detrended)
sit_stand_user.append(sit_stand_user1_2_detrended)
sit_stand_user.append(sit_stand_user2_1_detrended)
sit_stand_user.append(sit_stand_user2_2_detrended)
sit_stand_user.append(sit_stand_user3_1_detrended)
sit_stand_user.append(sit_stand_user3_2_detrended)
sit_stand_user.append(sit_stand_user4_1_detrended)
sit_stand_user.append(sit_stand_user4_2_detrended)


#------Stand to lie------#
stand_lie_user = []
stand_lie_user.append(stand_lie_user1_1_detrended)
stand_lie_user.append(stand_lie_user1_2_detrended)
stand_lie_user.append(stand_lie_user2_1_detrended)
stand_lie_user.append(stand_lie_user2_2_detrended)
stand_lie_user.append(stand_lie_user3_1_detrended)
stand_lie_user.append(stand_lie_user3_2_detrended)
stand_lie_user.append(stand_lie_user4_1_detrended)
stand_lie_user.append(stand_lie_user4_2_detrended)

#------Lie to stand------#
lie_sit_user = []
lie_sit_user.append(lie_sit_user1_1_detrended)
lie_sit_user.append(lie_sit_user1_2_detrended)
lie_sit_user.append(lie_sit_user2_1_detrended)
lie_sit_user.append(lie_sit_user2_2_detrended)
lie_sit_user.append(lie_sit_user3_1_detrended)
lie_sit_user.append(lie_sit_user3_2_detrended)
lie_sit_user.append(lie_sit_user4_1_detrended)
lie_sit_user.append(lie_sit_user4_2_detrended)

#------Sit to lie------#
sit_lie_user = []
sit_lie_user.append(sit_lie_user1_1_detrended)
sit_lie_user.append(sit_lie_user1_2_detrended)
sit_lie_user.append(sit_lie_user2_1_detrended)
sit_lie_user.append(sit_lie_user2_2_detrended)
sit_lie_user.append(sit_lie_user3_1_detrended)
sit_lie_user.append(sit_lie_user3_2_detrended)
sit_lie_user.append(sit_lie_user4_1_detrended)
sit_lie_user.append(sit_lie_user4_2_detrended)

#------Lie to Stand------#
lie_stand_user = []
lie_stand_user.append(lie_stand_user1_1_detrended)
lie_stand_user.append(lie_stand_user1_2_detrended)
lie_stand_user.append(lie_stand_user2_1_detrended)
lie_stand_user.append(lie_stand_user2_2_detrended)
lie_stand_user.append(lie_stand_user3_1_detrended)
lie_stand_user.append(lie_stand_user3_2_detrended)
lie_stand_user.append(lie_stand_user4_1_detrended)
lie_stand_user.append(lie_stand_user4_2_detrended)
