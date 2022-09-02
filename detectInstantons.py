import sys
import glob
import singPartDist as sp
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
import pylab as pl
import pandas as pd
from tqdm import tqdm


def mytanh(x,c1,t0,a,c2):
	"""
	hyberbolic tangens with for parameters
	
	Args:
		x (list or numpy array or float): input parameter for tanh
		c1 (float): multiplicative parameter  
		t0 (float): parameter to shift the centre
		a (float):  stretche or squeeze tanh in x-direction
		c2 (float): additive parameter, shift tanh up and down	
	Return:
		value of c1* tanh((x-t0)/a)+c2
	"""
	return c1*np.tanh((x-t0)/a)+c2

def detect(fileName,fileID, ta, a, N,numFrames, L,s=1,c1_0=0.3,a_0=10,c2_0 = 0.5):
	"""
	Detects the number of excitations in the given file
	Args:
		fileName (string): name of xyz file
		ta (int): timescale over which to search for excitations
		a (float): lenght scale as threshold for excitations
		N (int): number of particles
		numFrames (int): number of frames
		L (float):  length of box
	Optional inputs (default values worked well for my trajectories):
		s (float): smoothing parameter spline
		c1_0: initial guess for parameter c1 in tanh fit
		a_0: initial guess for parameter a in tanh fit
		c2_0: initial guess for parameter c2 in tanh fit
		
	Return:
		instantonIDs: DataFrame for each detected excitation with the following columns: 
		Id of the file it was detected for, particle ID, centre of the excitation t_0, duration of excitation deltat,amplitude of excitation
		failCount: list with 1: count of failed fits, 2: count of times the fit was too wide
	"""
	failCount = [0,0]
	instantonIDs = pd.DataFrame(columns = [ 'fileID', 'particleID', 't_0', 'deltat','amplitude']) # result DataFrame
	block = sp.readCoords(fileName,numFrames,N) # read coordinates
	for particle in tqdm(range(N)):
		p_coord = block[:,particle,:] # coordinates of particle
		p_deltat = []
		t0_temp = []
		cond1 =sp.averageDistPos(p_coord,0,ta,-1-ta,-1,0,L)  # caluclate  distance of average position in the first and the last ta of the trajectory
		if cond1 > a**2:	# check first condition: average distance from above larger than a
			for frame in range(ta,numFrames-ta):
				cond2 = sp.averageDistPos(p_coord,frame-ta,frame,frame,frame+ta,frame,L) # second test: for each frame check if the ta frames before and after have average positions larger than a (sudden jump)
				if cond2 > a**2:
					p = sp.singlepath(p_coord,frame,L)
					t = np.arange(p.diffs.shape[0])
					w = np.append(0,np.where(p.diffs[frame-ta:frame+ta]-a/2<0)[0])
					last = max(w)+1 # last time the distance drops below a/2	
					first = min(w[w>0]) # last time in other direction the  distance drops below a/2
					p_deltat.append(last-first) # append deltat to particle array
					t0_temp.append(frame)	# save frame as possible centre of excitation
			if len(p_deltat)>0:  # check if at least one frame fulfilled the condition (test3)
				p_t0  = t0_temp[np.argmin(p_deltat)] # frame with lowest delta is most likely to be the centre of the excitation
				refCoord = sp.averagePos(p_coord, int(p_t0-ta),int(p_t0-ta/2),L) # the reference coordinate for displacement here is the average position between ta and 1/2 ta before the centre of the excitation (to avoid large displacements in case of fluctuations)
				diffs = np.sqrt([sp.spaceDists(p_coord[t],refCoord,L) for t in range(numFrames)]) # displacement over the trajectory with respect to reference position
				p = sp.singlepath(p_coord,p_t0-ta,L)
				t_fit = t[p_t0-ta:p_t0+ta] # limit fit region to ta frames before and after centre
				p_fit = diffs[p_t0-ta:p_t0+ta]
				spl =UnivariateSpline(t_fit,p_fit,s=s) # smooth trajectory with spline
				try:
					fit_param, covar = curve_fit(mytanh,t_fit,spl(t_fit),[0.3,p_t0,ta/20,0.5], bounds = ([-np.inf,0,0,-np.inf], [np.inf,np.inf,np.inf,np.inf])) # tanh fit
				except:
					failCount[0] +=1		# count how often the fit fails
					continue
				if fit_param[2]*2<ta/4*3: 		# maximum allowed delta t is 3/4 of ta to avoid fitting  a straight line (fast diffusion)
				
					instantonIDs.loc[len(instantonIDs.index)] = {'fileID':fileID, 'particleID':particle, 't_0':fit_param[1], 'deltat':fit_param[2]*2,'amplitude':fit_param[0]*2} # save parameter of the detected excitation)
				else:
					failCount[1]+=1  # count how often t delta t is too large
	return instantonIDs, failCount

