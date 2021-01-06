import os
import numpy as np
# from array import *

import logging
log = logging.getLogger("CMAP")

def skim_convol(
		report_dir=None,
		emmemat_in_dir=None,
		emmemat_out_dir=None,
		peak=True,
		omx_out=False,
		extra_omx_out=False,
		cutoff=0.4,
		zone_types=None,
):
	"""
	Perform matrix convolution of transit skims.

	This function is based on the original TRANSIT_SKIM_FINAL_MATRICES1.PY

	Parameters
	----------
	report_dir : Path-like, optional
		The directory to which log reports are written, defaults
		to "report" in the current working directory.
	emmemat_in_dir : Path-like, optional
		The directory from which the input transit skims are read,
		defaults to "emmemat" in the current working directory.
	emmemat_out_dir : Path-like or False, optional
		The directory to which the output transit skims are written,
		defaults to the same directory as the inputs.  Set to `False`
		to skip writing the EMX format files (in which case set `omx_out`
		to True or you get nothing).
	peak : bool, default True
		Whether to process the peak or offpeak skims.
	omx_out : False or Path-like, default False
		Write the output skims to this OMX file.  Reading and writing to
		OMX is substantially faster for highly compressible data (like
		transit skims) although it prevents using the result as
		memory-mapped arrays.
	extra_omx_out : bool, default False
		Also write some input and intermediate skim arrays to the OMX file,
		which may be useful for debugging. Has no effect if `omx_out` is not
		set.
	cutoff : float, default 0.4
		Park-and-ride skim routing is ignored if the transit generalized
		cost exceeds this fraction of the congested drive-to-destination
		cost.  This prevents, for example, transit paths where the traveler
		drives a long distance to a park-and-ride lot, then boards transit
		for a short hop to the final destination.
		New for the 2020 model update: When the final destination is in the
		Chicago CBD, this cutoff is ignored when there is no valid non-PnR
		path, so the traveler is permitted to drive unlimited distances to
		access outlying transit (generally outlying Metra stations).
	zone_types : array-like, optional
		The zone type codes for zones in the model, used to filter for
		Chicago CBD zones to apply the new cutoff rules.  If not given, the
		zone types are loaded from the package-default M01 HW file (see
		m01_handler for details).

	"""

	if report_dir is None:
		report_dir = os.path.join(os.getcwd(), "report")
	os.makedirs(report_dir, exist_ok=True)

	if emmemat_in_dir is None:
		emmemat_in_dir = os.path.join(os.getcwd(), "emmemat")

	if emmemat_out_dir is None:
		emmemat_out_dir = emmemat_in_dir
	if emmemat_out_dir:
		os.makedirs(emmemat_out_dir, exist_ok=True)

	if peak:
		#   -- Input Matrix Numbers --
		inputmtx = (
			44,    ###0 AM peak hwy time matrix (mf44)
			803,   ###1 skimmed first mode (mf803)
			804,   ###2 skimmed priority mode (mf804)
			805,   ###3 skimmed last mode (mf805)
			808,   ###4 skimmed in-vehicle minutes (mf808)
			809,   ###5 skimmed transfer link minutes (mf809)
			810,   ###6 skimmed total wait minutes (mf810)
			811,   ###7 skimmed first wait minutes (mf811)
			818,   ###8 skimmed final average fare (mf818)
			819,   ###9 congested hwy generalized cost matrix (mf819)
			820,   ##10 indexed transit generalized cost (mf820)
			821,   ##11 intermediate zone matrix (mf821)
		)
		#   -- Output Matrix Numbers --
		outputmtx = (
			822,    ###0 indexed in-vehicle minutes (mf822)
			823,    ###1 indexed walk transfer minutes (mf823)
			824,    ###2 indexed total wait minutes (mf824)
			825,    ###3 indexed first wait minutes (mf825)
			828,    ###4 indexed final average fare (mf828)
			829,    ###5 indexed first mode (mf829)
			830,    ###6 indexed priority mode (mf830)
			831,    ###7 indexed last mode (mf831)
			832,    ###8 indexed auto generalized cost (mf832)
			833,    ###9 indexed auto min. to transit (mf833)
			834,    ##10 indexed transit/auto only (mf834)
		)
	else:
		#   -- Input Matrix Numbers --
		inputmtx = (
			46,    ### midday hwy time matrix (mf46)
			903,   ### skimmed first mode (mf903)
			904,   ### skimmed priority mode (mf904)
			905,   ### skimmed last mode (mf905)
			908,   ### skimmed in-vehicle minutes (mf908)
			909,   ### skimmed transfer link minutes (mf909)
			910,   ### skimmed total wait minutes (mf910)
			911,   ### skimmed first wait minutes (mf911)
			918,   ### skimmed final average fare (mf918)
			919,   ### congested hwy generalized cost matrix (mf919)
			920,   ### indexed transit generalized cost (mf920)
			921,   ### intermediate zone matrix (mf921)
		)
		#   -- Output Matrix Numbers --
		outputmtx = (
			922,    ### indexed in-vehicle minutes (mf922)
			923,    ### indexed walk transfer minutes (mf923)
			924,    ### indexed total wait minutes (mf924)
			925,    ### indexed first wait minutes (mf925)
			928,    ### indexed final average fare (mf928)
			929,    ### indexed first mode (mf929)
			930,    ### indexed priority mode (mf930)
			931,    ### indexed last mode (mf931)
			932,    ### indexed auto generalized cost (mf932)
			933,    ### indexed auto min. to transit (mf933)
			934,    ### indexed transit/auto only (mf934)
		)

	#   -- Input Matrices --
	mfauto  = os.path.join(emmemat_in_dir, "mf" + str(inputmtx[0]) + ".emx"  )
	mffmode = os.path.join(emmemat_in_dir, "mf" + str(inputmtx[1]) + ".emx"  )
	mfpmode = os.path.join(emmemat_in_dir, "mf" + str(inputmtx[2]) + ".emx"  )
	mflmode = os.path.join(emmemat_in_dir, "mf" + str(inputmtx[3]) + ".emx"  )
	mfinveh = os.path.join(emmemat_in_dir, "mf" + str(inputmtx[4]) + ".emx"  )
	mftrnfr = os.path.join(emmemat_in_dir, "mf" + str(inputmtx[5]) + ".emx"  )
	mftwait = os.path.join(emmemat_in_dir, "mf" + str(inputmtx[6]) + ".emx"  )
	mffwait = os.path.join(emmemat_in_dir, "mf" + str(inputmtx[7]) + ".emx"  )
	mfafare = os.path.join(emmemat_in_dir, "mf" + str(inputmtx[8]) + ".emx"  )
	mfcghwy = os.path.join(emmemat_in_dir, "mf" + str(inputmtx[9]) + ".emx"  )
	mftcost = os.path.join(emmemat_in_dir, "mf" + str(inputmtx[10]) + ".emx" )
	mfkzone = os.path.join(emmemat_in_dir, "mf" + str(inputmtx[11]) + ".emx" )
	#   -- Output Matrices --
	if emmemat_out_dir:
		mfinvehi = os.path.join(emmemat_out_dir, "mf" + str(outputmtx[0]) + ".emx"  )
		mftrnfri = os.path.join(emmemat_out_dir, "mf" + str(outputmtx[1]) + ".emx"  )
		mftwaiti = os.path.join(emmemat_out_dir, "mf" + str(outputmtx[2]) + ".emx"  )
		mffwaiti = os.path.join(emmemat_out_dir, "mf" + str(outputmtx[3]) + ".emx"  )
		mfafarei = os.path.join(emmemat_out_dir, "mf" + str(outputmtx[4]) + ".emx"  )
		mffmodei = os.path.join(emmemat_out_dir, "mf" + str(outputmtx[5]) + ".emx"  )
		mfpmodei = os.path.join(emmemat_out_dir, "mf" + str(outputmtx[6]) + ".emx"  )
		mflmodei = os.path.join(emmemat_out_dir, "mf" + str(outputmtx[7]) + ".emx"  )
		mfacosti = os.path.join(emmemat_out_dir, "mf" + str(outputmtx[8]) + ".emx"  )
		mfautrni = os.path.join(emmemat_out_dir, "mf" + str(outputmtx[9]) + ".emx"  )
		mfratioi = os.path.join(emmemat_out_dir, "mf" + str(outputmtx[10]) + ".emx" )

	#   -- Others --
	stats = os.path.join(report_dir, f"transit_skim_stats{8 if peak else 9}.txt")

	if os.path.exists(stats):
		os.remove(stats)

	# ---------------------------------------------------------------
	# Store matrix values in arrays.
	# ---------------------------------------------------------------
	#   -- Input Matrices --
	auto = np.fromfile(mfauto, dtype='f4') ## -- float, 4 bytes
	kzone = np.fromfile(mfkzone, dtype='f4')
	tcost = np.fromfile(mftcost, dtype='f4')
	inveh = np.fromfile(mfinveh, dtype='f4')
	trnfr = np.fromfile(mftrnfr, dtype='f4')
	twait = np.fromfile(mftwait, dtype='f4')
	fwait = np.fromfile(mffwait, dtype='f4')
	afare = np.fromfile(mfafare, dtype='f4')
	fmode = np.fromfile(mffmode, dtype='f4')
	pmode = np.fromfile(mfpmode, dtype='f4')
	lmode = np.fromfile(mflmode, dtype='f4')
	cghwy = np.fromfile(mfcghwy, dtype='f4')

	mcent = int(np.sqrt(auto.shape[0]))

	if zone_types is None:
		from .m01_handler import m01
		zone_types = m01.HW.zone_type.values
	assert zone_types.size <= mcent
	cbd_zone_indexes = np.where(zone_types == 1)[0]
	dzone_inside_cbd = np.zeros([mcent,mcent], dtype=bool)
	dzone_inside_cbd[:,cbd_zone_indexes] = True
	dzone_inside_cbd = dzone_inside_cbd.reshape(-1)

	## -- create leg1 (p-k) indices
	indxloc = np.arange(mcent*mcent)							## -- array of consecutive numbers representing element index values
	leg1pt1 = np.floor_divide(indxloc,mcent) * mcent					## -- portion of element index defining origin zone (division results in integer value)
	leg1indx = np.add(leg1pt1,kzone.astype('i4')-1,dtype='i4')				## -- add portion of element index defining destination zone
	log.info(
		f"Kzone 1-1: {kzone[0]}, Index 1-1: {leg1indx[0]}, "
		f"Kzone 121-2: {kzone[437882]}, Index 121-2: {leg1indx[437882]} \n"
	)

	## -- create leg2 (k-q) indices
	leg2pt1 = np.multiply(kzone.astype('i4')-1,mcent)
	leg2pt2 = np.mod(indxloc,mcent)
	leg2indx = np.add(leg2pt1,leg2pt2,dtype='i4')
	log.info(
		f"Kzone 1-1: {kzone[0]}, Index 1-1: {leg2indx[0]}, "
		f"Kzone 121-2: {kzone[437882]}, Index 121-2: {leg2indx[437882]} \n"
	)

	# ---------------------------------------------------------------
	# Create indexed matrices.
	# ---------------------------------------------------------------
	log.debug(f"Create indexed matrices")
	autoval = np.where(kzone>0, auto[leg1indx], kzone)					## -- hwy time matrix
	tcostval = np.where(kzone>0, tcost[leg1indx], kzone)				## -- indexed transit generalized cost
	invehval = np.where(kzone>0, inveh[leg2indx], kzone)				## -- skimmed in-vehicle minutes
	trnfrval = np.where(kzone>0, trnfr[leg2indx], kzone)				## -- skimmed transfer link minutes
	twaitval = np.where(kzone>0, twait[leg2indx], kzone)				## -- skimmed total wait minutes
	fwaitval = np.where(kzone>0, fwait[leg2indx], kzone)				## -- skimmed first wait minutes
	afareval = np.where(kzone>0, afare[leg2indx], kzone)				## -- skimmed final average fare
	fmodeval = np.where(kzone>0, fmode[leg2indx], kzone)				## -- skimmed first mode
	pmodeval = np.where(kzone>0, pmode[leg2indx], kzone)				## -- skimmed priority mode
	lmodeval = np.where(kzone>0, lmode[leg2indx], kzone)				## -- skimmed last mode
	threshold = np.where(cghwy>0, np.divide(tcostval,cghwy), cghwy)		## -- ratio of indexed transit cost to auto only cost

	log.debug(f"Swap original matrix value back in if the threshold exceeds the cutoff value")
	# -- Swap original matrix value back in if the threshold exceeds the cutoff value
	swap = (threshold>cutoff)
	# -- NEW 12/2020: but keep the park-and-ride indexed values if the destination zone
	#                 is inside the CBD and the original values are invalid
	prevent_swap = (dzone_inside_cbd) & (inveh > 999)
	swap &= ~prevent_swap
	autoval = np.where(swap, 0, autoval)
	tcostval = np.where(swap, 0, tcostval)
	invehval = np.where(swap, inveh, invehval)
	trnfrval = np.where(swap, trnfr, trnfrval)
	twaitval = np.where(swap, twait, twaitval)
	fwaitval = np.where(swap, fwait, fwaitval)
	afareval = np.where(swap, afare, afareval)
	fmodeval = np.where(swap, fmode, fmodeval)
	pmodeval = np.where(swap, pmode, pmodeval)
	lmodeval = np.where(swap, lmode, lmodeval)


	# ---------------------------------------------------------------
	# Write final matrix values into files.
	# ---------------------------------------------------------------
	# -- Arrays to write out
	mtxlist = (
		invehval,   ###0
		trnfrval,   ###1
		twaitval,   ###2
		fwaitval,   ###3
		afareval,   ###4
		fmodeval,   ###5
		pmodeval,   ###6
		lmodeval,   ###7
		tcostval,   ###8
		autoval,    ###9
		threshold,  ##10
	)
	if emmemat_out_dir:
		# -- Files to write to
		outmtx = (
			mfinvehi, ###0
			mftrnfri, ###1
			mftwaiti, ###2
			mffwaiti, ###3
			mfafarei, ###4
			mffmodei, ###5
			mfpmodei, ###6
			mflmodei, ###7
			mfacosti, ###8
			mfautrni, ###9
			mfratioi, ##10
		)
		x = 0
		with open(stats, 'a') as outFl:
			outFl.write("\n\n {0}\n\n".format('='*100,))

			for m in outmtx:
				mtxlist[x].tofile(outmtx[x])
				(fpath, fname) = os.path.split(outmtx[x])
				log.info(f"written to {fname}")
				outFl.write("{0} Written Successfully.\n".format(fname, ))
				outFl.write(
					f"\t-- Minimum = {min(mtxlist[x]):.4f}\n"
					f"\t-- Maximum = {max(mtxlist[x]):0.4f}\n"
					f"\t-- Mean = {sum(mtxlist[x])/len(mtxlist[x]):0.4f}\n"
					f"\t-- Sum = {sum(mtxlist[x]):0.4f}\n\n"
				)
				x += 1

	if omx_out:
		from larch import OMX
		with OMX(omx_out, mode='w') as omx:
			for n,m in enumerate(outputmtx):
				log.info(f"write to OMX mf{outputmtx[n]}")
				omx.add_matrix(f"mf{outputmtx[n]}", mtxlist[n].reshape([mcent,mcent]))
			# recalculate headway tables
			omx.add_matrix(f"mf{8 if peak else 9}38", mtxlist[2].reshape([mcent,mcent]) * 2)
			omx.add_matrix(f"mf{8 if peak else 9}39", mtxlist[2].reshape([mcent,mcent]) * 1.5)

			if extra_omx_out:
				omx.add_matrix(f"threshold", threshold.reshape([mcent, mcent]))
				omx.add_matrix(f"cghwy", cghwy.reshape([mcent, mcent]))
				omx.add_matrix(f"inveh", inveh.reshape([mcent, mcent]))
				omx.add_matrix(f"tcost", tcost.reshape([mcent, mcent]))
				omx.add_matrix(f"swap", swap.reshape([mcent, mcent]))

	log.info("-- TRANSIT SKIM MATRICES CREATED --")
