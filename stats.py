import numpy as np
import pandas as pd

def main():

	dataset = 'TUM'

	if (dataset == 'TUM'):

		path_generated = "KeyFrameTrajectory.txt"
		path_groundtruth = "../rgbd_dataset_freiburg1_xyz/groundtruth.txt"
		
		f = open(path_generated, "r")		
		data_generated = []
		for x in f:
			data_generated.append([float(xi) for xi in x.split()])

		g = open(path_groundtruth, "r")		
		data_groundtruth = []
		g.readline()
		g.readline()
		g.readline()
		for x in g:
			data_groundtruth.append([float(xi) for xi in x.split()])

		ind = 0
		data_groundtruth_downsampled = []

		for j in range(1,len(data_groundtruth)):
			if(data_generated[ind][0] > data_groundtruth[j-1][0] and data_generated[ind][0] < data_groundtruth[j][0]):
				t_j1 = abs(data_generated[ind][0] - data_groundtruth[j-1][0])
				t_j = abs(data_generated[ind][0] - data_groundtruth[j][0])
				if(t_j1 > t_j):
					data_groundtruth_downsampled.append(data_groundtruth[j][0:])
				else:
					data_groundtruth_downsampled.append(data_groundtruth[j-1][0:])
				ind += 1
				j -= 1
			if ind == len(data_generated):
				break

		traj_gen = np.array(data_generated)
		traj_tru = np.array(data_groundtruth_downsampled)

		traj_gen = traj_gen[:,1:]
		traj_tru = traj_tru[:,1:]

		avg_traj_error = np.linalg.norm(traj_gen-traj_tru)

		print("The average trajectory error is: {}".format(avg_traj_error))


if __name__ == "__main__":
    main()