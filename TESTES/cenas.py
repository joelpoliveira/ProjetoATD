#plot_activity_dft(walks_up_user4_2_detrended[0], len(walks_up_user4_2_detrended[0]), 'WALK_UP - Experiência 1 do User 4')

activity = 'walks_down'
title = 'WALK DOWN'

for i in range(4):
	for j in range(2):
		command = ''
		command += f'plot_activity_dft({activity}_user{i+1}_{j+1}[0], len({activity}_user{i+1}_{j+1}_detrended[0]), '
		command += f'{i+1}, {j+1}, \'{title}\')'
		print(command)
