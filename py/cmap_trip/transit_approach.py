import numpy as np
import logging
log = logging.getLogger('CMAP')

from .m023_handler import m023
from .m01_handler import m01
from .distr_handler import distr
from .ae_distance_sim import simulate_ae_dist
from .skims_handler import first_mode_peak, first_mode_offpeak, last_mode_peak, last_mode_offpeak
from .random_states import check_random_state

# trip types
HW = 'HW'  # HOME PRODUCTIONS TO WORK ATTRACTIONS
HO = 'HO'  # HOME PRODUCTIONS TO NON-WORK/NON-HOME ATTRACTIONS
NH = 'NH'  # NON-HOME PRODUCTIONS TO NON-HOME ATTRACTIONS

ITER = 50  # NUMBER OF TRIPS USED TO COMPUTE AVERAGE IMPEDANCES

from .modecodes import (
	APPROACH_WALK,
	APPROACH_BUS,
	APPROACH_PARK_N_RIDE,
	APPROACH_KISS_N_RIDE,
	APPROACH_FEEDER_BUS,
	N_APPROACH_MODES,
	APPROACH_MODE_NAMES,
	DIST_TO_BUS,
	DIST_TO_CTA_RAIL,
	DIST_TO_METRA,
	DIST_TO_FEEDER_BUS,
	DIST_TO_PARK_N_RIDE_STATION,
	N_DIST_TO_TYPES,
	TransitModeCode_CTA_RAIL,
	TransitModeCode_METRA_RAIL,
	TransitModeCode_CTA_EXPRESS_BUS,
	TransitModeCode_CTA_REGULAR_BUS,
	TransitModeCode_PACE_BUS,
)



FRONT_END = 0
BACK_END = 1
N_TRIP_ENDS = 2

SPDWLK = 30
# SPDWLK = SYSTEM-WIDE SPEED OF WALKING,
#          DEFAULT IS 30 TENTHS OF A MILE PER HOUR

SPEEDS = [7, 15, 20, 30, 5, 10, 12, 17]
# SPEEDS = SPEEDS OF APPROACH AUTO AND BUS BY ZONE AREA TYPE
#          AUTO APPROACH SPEEDS:
#            ZONE TYPE 1 (CHICAGO CBD) DEFAULT IS 7 MPH
#            ZONE TYPE 2 (CHICAGO REMAINDER) DEFAULT IS 15 MPH
#            ZONE TYPE 3 (DENSE SUBURB) DEFAULT IS 20 MPH
#            ZONE TYPE 4 (SPARSE SUBURB) DEFAULT IS 30 MPH
#          BUS APPROACH SPEEDS:
#            ZONE TYPE 1 (CHICAGO CBD) DEFAULT IS 5 MPH
#            ZONE TYPE 2 (CHICAGO REMAINDER) DEFAULT IS 10 MPH
#            ZONE TYPE 3 (DENSE SUBURB) DEFAULT IS 12 MPH
#            ZONE TYPE 4 (SPARSE SUBURB) DEFAULT IS 17 MPH

DRVOT = 14
# DRVOT  = DRIVER'S VALUE OF TIME, DEFAULT IS 14 CENTS/MIN

PACE_BUS_BOARDING_FARE = m023.PACE_BUS_BOARDING_FARE
PACE_BUS_FIRST_XFER_FARE = m023.PACE_BUS_FIRST_XFER_FARE
FEEDER_BUS_BOARDING_FARE = m023.FEEDER_BUS_BOARDING_FARE
FEEDER_BUS_CBD_FARE = m023.FEEDER_BUS_CBD_FARE

CTA_FIRST_XFER_FARE = m023.CTA_FIRST_XFER_FARE
CTA_CBD_LINK_UP_FARE = m023.CTA_CBD_LINK_UP_FARE

AUTO_OPERATING_COST_BY_ZONETYPE = m023.AUTO_OPERATING_COST_BY_ZONETYPE  # AVERAGE OPERATING COST PER MILE FOR AUTO, BY ZONE TYPE

AFC1 = 35  # AUTO FIXED COSTS FOR AUTO DRIVER IN CENTS
AFC2 = 20  # AUTO FIXED COSTS FOR AUTO PASSENGER IN CENTS

#  W2PNR  = WALK TIME TO STATION FROM PARK AND RIDE LOT,
#           DEFAULT IS 2 MINUTES
W2PNR = 2


def _simulate_approach_distances(
		zone,
		attached_mode,
		trip_purpose,
		trip_end,
		out,
		random_state=None,
):
	"""

	Parameters
	----------
	zone : int
		Zone id (1-based)
	attached_mode : int
		Number for first or last mode (as matches this approach)
	trip_purpose : {'HW','HO','NH'}
		Trip purpose, used to select DISTR table and possibly filter
		approach modes
	trip_end : {0,1}
		Zero if approach to first mode, one if approach from last mode
	out : array-like
		Output array must already exist, as a float dtype,
		with shape [replications, N_APPROACH_MODES, ]

	"""
	random_state = check_random_state(random_state)
	replication = out.shape[0]
	for J in range(N_DIST_TO_TYPES):
		# OBTAIN APPROACH DISTANCES TO FIVE MODES
		if (J == DIST_TO_BUS):
			distr_params = distr[trip_purpose].loc[(zone, 'bus')]
		elif (J == DIST_TO_CTA_RAIL):
			# DISTANCE OBTAINED ONLY IF FIRST/LAST MODE IS CTA RAIL
			if attached_mode == TransitModeCode_CTA_RAIL:
				distr_params = distr[trip_purpose].loc[(zone, 'ctarail')]
			else:
				distr_params = (999,999,999)
		elif (J == DIST_TO_METRA):
			# DISTANCE OBTAINED ONLY IF FIRST/LAST MODE IS METRA
			if attached_mode == TransitModeCode_METRA_RAIL:
				distr_params = distr[trip_purpose].loc[(zone, 'metra')]
			else:
				distr_params = (999, 999, 999)
		elif (J == DIST_TO_FEEDER_BUS):
			# DISTANCE OBTAINED ONLY IF FIRST/LAST MODE IS METRA
			if attached_mode == TransitModeCode_METRA_RAIL:
				distr_params = distr[trip_purpose].loc[(zone, 'feederbus')]
			else:
				distr_params = (999, 999, 999)
		elif (J == DIST_TO_PARK_N_RIDE_STATION):
			# PARK AND RIDE STATION DISTANCE OBTAINED WHEN TRIP END IS FRONT
			if trip_end == FRONT_END:
				distr_params = distr[trip_purpose].loc[(zone, 'pnr')]
			else:
				distr_params = (999, 999, 999)
		else:
			raise ValueError(J)
		if distr_params[2] != 999:
			out[:,J] = simulate_ae_dist(*distr_params, replication=replication, random_state=random_state)
		else:
			out[:,J] = 255.0



# def simulate_approach_distances_222(ozone, dzone, firstmode, lastmode, trip_purpose):
# 	DOND = np.full([ITER, N_APPROACH_MODES, N_TRIP_ENDS], 255.)
# 	for I in range(N_TRIP_ENDS):
#
# 		# C     I=1 OBTAIN DISTANCES FOR ORIGIN
# 		# C     I=2 OBTAIN DISTANCES FOR DESTINATION
# 		if I == FRONT_END:
# 			Z=ozone
# 			M=firstmode
# 		else:
# 			Z=dzone
# 			M=lastmode
#
#
# 		for J in range(N_APPROACH_MODES):
# 			# C
# 			# C     OBTAIN APPROACH DISTANCES TO FIVE MODES
# 			DOND[:,J,I]=255.
#
# 			if (J == APPROACH_WALK):
# 				distr_params = distr[trip_purpose].loc[(ozone, 3)]
# 			elif (J == APPROACH_BUS):
# 				distr_params = distr[trip_purpose].loc[(ozone, 2)]
# 			elif (J == APPROACH_PARK_N_RIDE):
# 				distr_params = distr[trip_purpose].loc[(ozone, 1)]
# 			elif (J == APPROACH_KISS_N_RIDE):
# 				distr_params = distr[trip_purpose].loc[(ozone, 4)]
# 			else: # (J == FEEDER_BUS):
# 				distr_params = distr[trip_purpose].loc[(ozone, 5)]
#
# 			DOND[:,J,I] = simulate_ae_dist(*distr_params, replication=ITER)
#
# 	return DOND


def _IS_CTA(m):
	return (
		m == TransitModeCode_CTA_RAIL
		or m == TransitModeCode_CTA_REGULAR_BUS
		or m == TransitModeCode_CTA_EXPRESS_BUS
	)

def transit_approach(
		ozone,
		dzone,
		TPTYPE,
		replication=None,
		approach_distances=None,
		trace=False,
		random_state=None,
):
	"""
	Replaces TRAPP fortran.

	Parameters
	----------
	ozone, dzone : int
		Zone ID numbers
	TPTYPE : {'HW', 'HO', 'NH'}
		Trip type
	replication : int
		Number of simulation replications

	Returns
	-------
	ae_drivetime : array of int32, shape [replication]
		simulated in vehicle (drive) approach times, in minutes
	ae_walktime : array of int32, shape [replication]
		simulated out of vehicle (walk) approach times, in minutes
	ae_cost : array of int32, shape [replication]
		simulated approach costs, in cents
	ae_waittime : array of int32, shape [replication]
		simulated approach waiting times
	best_approach_mode : array of int8, shape [replication, 2]
		simulated best approach modes

	"""
	random_state = check_random_state(random_state or ozone+dzone)

	if replication is None:
		replication = ITER

	if trace:
		log.log(trace, f"transit_approach({ozone},{dzone},{TPTYPE},{replication})")

	ozone_idx = ozone-1
	dzone_idx = dzone-1

	m01_df = m01[TPTYPE]
	ZTYPE = m01_df['zone_type'] # integers 1-4
	if TPTYPE == HW:
		fwbus = m01_df['first_wait_bus_peak'] # FIRST WAIT FOR BUS IN APPROACH SUBMODEL
		fwfdr = m01_df['first_wait_feeder_peak'] # FIRST WAIT FOR FEEDER BUS IN APPROACH SUBMODEL
	else:
		fwbus = m01_df['first_wait_bus_offpeak'] # FIRST WAIT FOR BUS IN APPROACH SUBMODEL
		fwfdr = m01_df['first_wait_feeder_offpeak'] # FIRST WAIT FOR FEEDER BUS IN APPROACH SUBMODEL
	PNRAVL = m01_df['pnr_flag'] # park-n-ride available, by zone
	PRCOST = m01_df['pnr_parking_cost'] # park-n-ride cost, by zone

	# -- INITIALIZE VALUES --
	approach_cost = np.zeros([replication, N_APPROACH_MODES], dtype=np.float32)
	approach_waittime = np.zeros([N_APPROACH_MODES], dtype=np.float32)
	approach_drivetime = np.zeros([replication, N_APPROACH_MODES], dtype=np.float32)
	approach_walktime = np.zeros([replication, N_APPROACH_MODES], dtype=np.float32)
	TVAR4 = np.zeros([replication, 5], dtype=np.float32)

	best_approach_mode = np.zeros([replication, N_TRIP_ENDS], dtype=np.int8)
	best_cost = np.zeros([replication, N_TRIP_ENDS], dtype=np.int32)
	best_waittime = np.zeros([replication, N_TRIP_ENDS], dtype=np.int32)
	best_walktime = np.zeros([replication, N_TRIP_ENDS], dtype=np.int32)
	best_drivetime = np.zeros([replication, N_TRIP_ENDS], dtype=np.int32)
	#
	#     GET ZONE TYPES
	#
	ozone_type = ZTYPE.iloc[ozone_idx]
	dzone_type = ZTYPE.iloc[dzone_idx]
	#
	#     GET INTERCHANGE ATTRIBUTES
	#     FM=FIRST MODE,LM=LAST MODE,PM=PRIORITY MODE
	#
	if TPTYPE == 'HW':
		FM = first_mode_peak[ozone_idx, dzone_idx]
		LM = last_mode_peak[ozone_idx, dzone_idx]
	else:
		FM = first_mode_offpeak[ozone_idx, dzone_idx]
		LM = last_mode_offpeak[ozone_idx, dzone_idx]
	#
	#     INET TRANSIT NETWORK STORES SOME SUBURBAN BUS LINES (MODE=6)
	#     AS MODE=5 DUE TO ARRAY SIZE LIMITS.  IF MODE=5 AND
	#     ZONE TYPE NO. 1 IS OUTSIDE OF CHICAGO, THEN CHANGE MODE TO 6.
	#
	if (FM == 5 and ozone_type > 2):
		FM = 6
	if (LM == 5 and dzone_type > 2):
		LM = 6
	# DEBUGGING:      WRITE (*,'(A)') ' CHECK POINT 1 REACHED  '
	#
	#     GET APPROACH DISTANCES FOR FIRST AND LAST MODES
	#
	# DEBUGGING:        WRITE (31, '(A,10F8.4)')  '  RAN6 FOR ADIST IN TRAPP',
	# DEBUGGING:  ARAN6(1),RAN6(2),RAN6(3),RAN6(4),RAN6(5),RAN6(6),RAN6(7),RAN6(8),
	# DEBUGGING:       BRAN6(9),RAN6(10)

	####      CALL ADIST(ozone,dzone,FM,LM)
	if approach_distances is not None:
		assert approach_distances.shape == [replication, N_DIST_TO_TYPES, N_TRIP_ENDS]
	else:
		approach_distances = np.empty([replication, N_DIST_TO_TYPES, N_TRIP_ENDS])
		_simulate_approach_distances(
			ozone,
			attached_mode=FM,
			trip_purpose=TPTYPE,
			trip_end=0,
			out=approach_distances[:,:,0],
			random_state=random_state,
		)
		_simulate_approach_distances(
			dzone,
			attached_mode=LM,
			trip_purpose=TPTYPE,
			trip_end=1,
			out=approach_distances[:,:,1],
			random_state=random_state,
		)
	if trace:
		log.log(trace, f" PRODUCTION APPROACH DISTANCES")
		log.log(trace, f"  to Bus    {approach_distances[:5,DIST_TO_BUS,0]}")
		log.log(trace, f"  to El     {approach_distances[:5,DIST_TO_CTA_RAIL,0]}")
		log.log(trace, f"  to Metra  {approach_distances[:5,DIST_TO_METRA,0]}")
		log.log(trace, f"  to feeder {approach_distances[:5,DIST_TO_FEEDER_BUS,0]}")
		log.log(trace, f"  to PnR    {approach_distances[:5,DIST_TO_PARK_N_RIDE_STATION,0]}")
		log.log(trace, f" ATTRACTION APPROACH DISTANCES")
		log.log(trace, f"  to Bus    {approach_distances[:5,DIST_TO_BUS,1]}")
		log.log(trace, f"  to El     {approach_distances[:5,DIST_TO_CTA_RAIL,1]}")
		log.log(trace, f"  to Metra  {approach_distances[:5,DIST_TO_METRA,1]}")
		log.log(trace, f"  to feeder {approach_distances[:5,DIST_TO_FEEDER_BUS,1]}")
		log.log(trace, f"  to PnR    {approach_distances[:5,DIST_TO_PARK_N_RIDE_STATION,1]}")


	#     CHECK FIRST/LAST MODES AND COMPUTE APPROACH TIME AND COST
	#
	#     ARRAYS approach_walktime,APCOST,approach_drivetime CONTAIN TIME TO WALK,APPROACH COST,
	#     AND IN-VEHICLE APPROACH TIME RESPECTIVELY. THESE ARRAYS HAVE FIVE
	#     ELEMENTS FOR FIVE POSSIBLE APPROACH MODES.( 1-WALK,2-BUS,
	#     3-PARK & RIDE,4-KISS & RIDE,AND 5-FEEDER BUS)
	#

	for I in range(N_TRIP_ENDS):
		#     I=1 GET VALUES FOR ORIGIN
		#     I=2 GET VALUES FOR DESTINATION
		if (I == FRONT_END):
			Z = ozone
			M = FM
		else: # I == BACK_END
			Z = dzone
			M = LM

		ZTYPE_Z = ZTYPE.iloc[Z]

		#
		#  IN THIS CASE WE ARE MAKING THE STATION PARKING COST FOR HOME BASED OTHER AND
		#  NON-HOME BASED TRIPS EQUAL TO 60 PERCENT OF HOME BASED WORK
		#  CHANGE MADE 12/8/93 BY GWS NEXT LINE
		#      IF(TPTYPE.NE.1) PRCOST(Z) = PRCOST(Z) * 0.6
		# *******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************
		#     NONWORK PARK AND RIDE PARKING COST IS NOW READ FROM M01.  IN MANY
		#     CASES THE NONWORK PNR COSTS ARE HIGHER THAN WORK DUE TO
		#     DISCOUNTING OF MONTHLY PARKING FEES.
		#
		# *******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************
		#     SET HIGH STARTING VALUE OF TVAR5
		TVAR5 = np.full([replication], 1.E10)
		# *******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************
		#     IN CALCULATING TVAR4 AND TVAR5
		#       IN-VEHICLE TIME = DRVOT = 20 CENTS/MIN
		#       OUT-OF-VEHICLE TIME = 40 CENTS/MIN
		#       PASSENGER TIME = 0.5 DRIVER TIME
		# *******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************
		#
		#     FOLLOWING SECTION ADDED BY EASH TO SIMPLIFY LOGIC
		#     IF M IS BUS (MODE<7) THEN ONLY POSSIBLE APPROACH COST IS TIME TO
		#     WALK TO BUS.  OTHER APPROACH COSTS AND TIMES ARE LINE=HAUL.
		if (M < 7):
			J = APPROACH_WALK
			approach_walktime[:, 0] = approach_distances[:, DIST_TO_BUS, I] / SPDWLK * 600.
			# INCREASE WALK TIME IN CHICAGO CBD FOR WORK TRIPS
			if (ZTYPE_Z == 1 and TPTYPE == HW):
				approach_walktime[:, 0] = approach_walktime[:, 0] * 1.20
			TVAR4[:, J] = approach_walktime[:, 0] * DRVOT * 2.0
			TVAR5[:] = TVAR4[:, J]
			best_approach_mode[:, I] = 0
			best_drivetime[:, I] = 0
			best_walktime[:, I] = approach_walktime[:, 0] + .5
			best_cost[:, I] = 0
			best_waittime[:, I] = 0
			for J in [APPROACH_BUS, APPROACH_PARK_N_RIDE, APPROACH_KISS_N_RIDE, APPROACH_FEEDER_BUS]:
				TVAR4[:, J] = 0.
				approach_cost[:, J] = 0.
				approach_waittime[J] = 0.
				approach_drivetime[:, J] = 0.
				approach_walktime[:, J] = 0.

		else:
			#     REMAINDER OF SUBROUTINE FOR RAIL TRANSIT/COMMUTER RAIL ONLY
			#     GET VALUES FOR FIVE ALTERNATIVES
			for J in range(5):
				TVAR4[:, J] = 0.
				approach_cost[:, J] = 0.
				approach_waittime[J] = 0.
				approach_drivetime[:, J] = 0.
				approach_walktime[:, J] = 0.

				K = int(max(0, M - 6)) # 0 for BUS, 1 for CTA RAIL(7-1), 2 for METRA(8-1)

				# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
				#     J=0(WALK).COMPUTE WALKING TIME TO FIRST MODE.NO COST OR IN-VEHICLE TIME
				# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
				if J == APPROACH_WALK:
					approach_walktime[:, J] = approach_distances[:, K, I] / SPDWLK * 600.
					# INCREASE WALK TIME IN CHICAGO CBD FOR WORK TRIPS
					if (ZTYPE_Z == 1 and TPTYPE == HW):
						approach_walktime[:, J] = approach_walktime[:, J] * 1.20

					TVAR4[:, J] = approach_walktime[:, J] * DRVOT * 2.0
					# ADD APPROACH TIMES AND COSTS - EVERYTHING SHOULD NOW BE IN CENTS
					TVAR4[:, J] = TVAR4[:, J] + approach_cost[:, J]

				# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
				#    J=1(BUS) FIRST MODE. COMPUTE WALKING TIME, COST, AND IN-VEHICLE TIME
				# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
				elif J == APPROACH_BUS:
					approach_walktime[:, J] = approach_distances[:, DIST_TO_BUS, I] / SPDWLK * 600.
					# INCREASE WALK TIME IN CHICAGO CBD
					if (ZTYPE_Z == 1):
						approach_walktime[:, J] = approach_walktime[:, J] * 1.20
					approach_waittime[J] = fwbus[Z]
					approach_drivetime[:, J] = approach_distances[:, K, I] / SPEEDS[ZTYPE_Z-1 + 4] * 60.
					TVAR4[:, J] = approach_walktime[:, J] * DRVOT * 2.0 + approach_drivetime[:, J] * DRVOT + approach_waittime[J] * DRVOT * 2.0

					# *******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************
					#
					#     COST COMPUTATIONS FOR APPROACH BUS
					#     REVISED BY EASH 12/4/93 TO REFLECT CURRENT FARES
					#   ====  ORIGIN  ====
					if (I == FRONT_END):
						# FIRST MODE SUBURBAN RAIL - CHECK ZONE TYPE AT ORIGIN						
						if (M == TransitModeCode_METRA_RAIL):
							if (ozone_type > 2):
								# --- SUBURBAN ORIGIN, PACE BUS ---
								# PACE BUS - METRA RAIL, ADDED FARE IS PACE FEEDER BUS FARE
								if (LM == TransitModeCode_METRA_RAIL):
									approach_cost[:, J] = FEEDER_BUS_BOARDING_FARE
								#     PACE BUS - METRA RAIL - CTA, NO ADDED FARE, LINKUP > FEEDER BUS
								if _IS_CTA(LM):
									approach_cost[:, J] = 0
								# PACE BUS - METRA RAIL - PACE, ADDED FARE IS LINKUP LESS FEEDER BUS
								if (LM == TransitModeCode_PACE_BUS):
									approach_cost[:, J] = FEEDER_BUS_CBD_FARE
								# ADD APPROACH TIMES AND COSTS - EVERYTHING SHOULD NOW BE IN CENTS
								TVAR4[:, J] = TVAR4[:, J] + approach_cost[:, J]
							else:
								#   --- CHICAGO ORIGIN, CTA BUS ---
								# CTA BUS - METRA RAIL, ADDED FARE IS LINKUP FARE (SINGLE RIDE)
								if (LM == TransitModeCode_METRA_RAIL):
									approach_cost[:, J] = CTA_CBD_LINK_UP_FARE
								# CTA BUS - METRA RAIL - CTA, ADDED FARE IS CTA TRANSFER
								if _IS_CTA(LM):
									approach_cost[:, J] = CTA_FIRST_XFER_FARE
								# CTA BUS - METRA RAIL - PACE, ADDED FARE IS LINKUP LESS FEEDER BUS
								if (LM == TransitModeCode_PACE_BUS):
									approach_cost[:, J] = FEEDER_BUS_CBD_FARE
								# ADD APPROACH TIMES AND COSTS - EVERYTHING SHOULD NOW BE IN CENTS
								TVAR4[:, J] = TVAR4[:, J] + approach_cost[:, J]

						elif (LM < 7):
							#     FIRST MODE CTA RAIL
							#     WHEN THIS IS TRUE A FULL FARE AND TRANSFER HAVE
							#     BEEN PAID, SO NO ADDED FARE IS NEEDED FOR BUS
							TVAR4[:, J] = TVAR4[:, J] + approach_cost[:, J]

						# ORIGIN OTHER THAN CHICAGO, ADDED FARE IS NOW AN RTA TRANSFER
						elif (ozone_type > 2):
							approach_cost[:, J] = PACE_BUS_FIRST_XFER_FARE
							TVAR4[:, J] = TVAR4[:, J] + approach_cost[:, J]

						# CHICAGO ORIGIN, ADDED FARE IS CTA TRANSFER
						else:
							approach_cost[:, J] = CTA_FIRST_XFER_FARE
							TVAR4[:, J] = TVAR4[:, J] + approach_cost[:, J]


					#   ====  DESTINATION  ====
					else:
						# LAST MODE SUBURBAN RAIL
						if (M == TransitModeCode_METRA_RAIL):
							if (dzone_type > 2):
								#     SUBURBAN DESTINATION, PACE BUS
								#     METRA RAIL - PACE BUS, ADDED FARE IS PACE FEEDER BUS FARE
								if (FM == TransitModeCode_METRA_RAIL):
									approach_cost[:, J] = FEEDER_BUS_BOARDING_FARE
								#     CTA - METRA RAIL - PACE BUS,  NO ADDED FARE, LINKUP > FEEDER BUS
								if _IS_CTA(FM):
									approach_cost[:, J] = 0
								#     PACE - METRA RAIL - PACE BUS, ADDED FARE IS LINKUP LESS FEEDER BUS
								if (FM == TransitModeCode_PACE_BUS):
									approach_cost[:, J] = FEEDER_BUS_CBD_FARE
								TVAR4[:, J] = TVAR4[:, J] + approach_cost[:, J]
							else:
								#     CHICAGO DESTINATION, CTA BUS
								#     METRA - CTA BUS, ADDED COST IS LINKUP FARE (SINGLE RIDE)
								if (FM == TransitModeCode_METRA_RAIL):
									approach_cost[:, J] = CTA_CBD_LINK_UP_FARE
								#     CTA - METRA - CTA BUS, ADDED COST IS CTA TRANSFER
								if _IS_CTA(FM):
									approach_cost[:, J] = CTA_FIRST_XFER_FARE
								#     PACE - METRA - CTA BUS, ADDED COST IS LINKUP MINUS FEEDER BUS
								if (FM == TransitModeCode_PACE_BUS):
									approach_cost[:, J] = FEEDER_BUS_CBD_FARE
								TVAR4[:, J] = TVAR4[:, J] + approach_cost[:, J]


						# LAST MODE CTA RAPID TRANSIT
						# NO ADDED FARE FOR EGRESS BUS, TRANSFER OR LINKUP ALREADY PURCHASED
						elif (FM < 7):
							TVAR4[:, J] = TVAR4[:, J] + approach_cost[:, J]

						# ADD CTA TRANSFER IF NOT ALREADY PAID
						else:
							approach_cost[:, J] = CTA_FIRST_XFER_FARE * ((best_approach_mode[:, FRONT_END]==APPROACH_BUS)|(best_approach_mode[:, FRONT_END]==APPROACH_FEEDER_BUS))
							TVAR4[:, J] = TVAR4[:, J] + approach_cost[:, J]
						# # DESTINATION OTHER THAN CHICAGO, ADDED COST IS STILL CTA TRANSFER
						# elif (dzone_type > 2):
						# 	approach_cost[:, J] = CTA_FIRST_XFER_FARE
						# 	TVAR4[:, J] = TVAR4[:, J] + approach_cost[:, J]
						#
						# # CHICAGO DESTINATION, ADDED COST IS CTA TRANSFER
						# else:
						# 	approach_cost[:, J] = CTA_FIRST_XFER_FARE
						# 	TVAR4[:, J] = TVAR4[:, J] + approach_cost[:, J]


				# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
				#    J=2(PARK & RIDE) FIRST MODE. PARK & RIDE FOR APPROACH TO RAPID TRANSIT AND SUBURBAN RAIL ROAD
				# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
				elif (J == APPROACH_PARK_N_RIDE and I == FRONT_END):
					approach_drivetime[:, J] = approach_distances[:, DIST_TO_PARK_N_RIDE_STATION, I] / SPEEDS[ZTYPE_Z - 1] * 60.
					approach_walktime[:, J] = W2PNR
					#     APPROACH COST=PER MILE COST + FIXED COST
					approach_cost[:, J] = approach_distances[:, DIST_TO_PARK_N_RIDE_STATION, I] * AUTO_OPERATING_COST_BY_ZONETYPE[ZTYPE_Z - 1]
					#     OPERATING COST MAY NOT BE LESS THAN 5 CENTS
					approach_cost[:, J] = np.fmin(approach_cost[:, J], 5.0)

					approach_cost[:, J] = approach_cost[:, J] + AFC1
					#     ADD HALF OF THE PARKING COST IF PARK-&-RIDE AVAILABLE
					if (PNRAVL[Z]):
						approach_cost[:, J] = approach_cost[:, J] + PRCOST[Z] / 2
					#     IF NO PARK-&-RIDE FACILITY AVAILABLE INCREASE WALK TIME
					if (not PNRAVL[Z]):
						approach_walktime[:, J] = 3 * W2PNR

					TVAR4[:, J] = approach_walktime[:, J] * DRVOT * 2.0 + approach_drivetime[:, J] * DRVOT
					TVAR4[:, J] = TVAR4[:, J] + approach_cost[:, J]

				# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
				#    J=3(KISS & RIDE) FIRST MODE. KISS & RIDE FOR APPROACH TO RAPID TRANSIT AND SUBURBAN RAIL ROAD
				# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
				elif (J == APPROACH_KISS_N_RIDE and I == FRONT_END):
					approach_drivetime[:, J] = approach_distances[:, DIST_TO_PARK_N_RIDE_STATION, I] / SPEEDS[ZTYPE_Z - 1] * 60.
					approach_walktime[:, J] = W2PNR
					approach_cost[:, J] = approach_distances[:, DIST_TO_PARK_N_RIDE_STATION, I] * AUTO_OPERATING_COST_BY_ZONETYPE[ZTYPE_Z - 1]
					approach_cost[:, J] = np.fmin(approach_cost[:, J], 5.0)
					# *******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************
					#     ASSUMPTION IS THAT KISS AND RIDE REQUIRES A SPECIAL
					#     TRIP FROM HOME.  DRIVER AND PASSENGER TIME VALUES NOW EQUAL.
					#      APCOST[J]=APCOST[J]*2.+AFC2+(DRVOT*approach_drivetime[J]*2.)/10
					approach_cost[:, J] = approach_cost[:, J] * 2. + AFC2
					TVAR4[:, J] = approach_walktime[:, J] * DRVOT * 2 + approach_drivetime[:, J] * 2.0 * DRVOT + approach_drivetime[:, J] * DRVOT
					# *******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************
					TVAR4[:, J] = TVAR4[:, J] + approach_cost[:, J]
				# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
				#    J=4(FEEDER BUS) FIRST MODE. FEEDER BUS FOR RAIL ONLY
				# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
				elif (J == APPROACH_FEEDER_BUS and I == FRONT_END):
					approach_walktime[:, J] = approach_distances[:, DIST_TO_FEEDER_BUS, I] / SPDWLK * 600.
					approach_waittime[J] = fwfdr[Z]
					approach_drivetime[:, J] = approach_distances[:, K, I] / SPEEDS[ZTYPE_Z-1 + 4] * 60.
					# *******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************
					TVAR4[:, J] = approach_walktime[:, J] * DRVOT * 2.0 + approach_drivetime[:, J] * DRVOT + approach_waittime[J] * DRVOT * 2.0
					# *******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************
					# ## SAME CODE BLOCK AS FOR BUS
					#     FIRST MODE SUBURBAN RAIL - CHECK ZONE TYPE AT ORIGIN
					if (M == 8):
						if (ozone_type > 2):
							#   --- SUBURBAN ORIGIN, PACE BUS ---
							#     PACE BUS - METRA RAIL, ADDED FARE IS PACE FEEDER BUS FARE
							if (LM == 8):
								approach_cost[:, J] = FEEDER_BUS_BOARDING_FARE
							#     PACE BUS - METRA RAIL - CTA, NO ADDED FARE, LINKUP > FEEDER BUS
							if (LM == 7 or LM == 5 or LM == 4):
								approach_cost[:, J] = 0
							#     PACE BUS - METRA RAIL - PACE, ADDED FARE IS LINKUP LESS FEEDER BUS
							if (LM == 6):
								approach_cost[:, J] = FEEDER_BUS_CBD_FARE
							#     ADD APPROACH TIMES AND COSTS - EVERYTHING SHOULD NOW BE IN CENTS
							TVAR4[:, J] = TVAR4[:, J] + approach_cost[:, J]
						else:
							#   --- CHICAGO ORIGIN, CTA BUS ---
							#     CTA BUS - METRA RAIL, ADDED FARE IS LINKUP FARE (SINGLE RIDE)
							if (LM == 8):
								approach_cost[:, J] = CTA_CBD_LINK_UP_FARE
							#     CTA BUS - METRA RAIL - CTA, ADDED FARE IS CTA TRANSFER
							if (LM == 7 or LM == 5 or LM == 4):
								approach_cost[:, J] = CTA_FIRST_XFER_FARE
							#     CTA BUS - METRA RAIL - PACE, ADDED FARE IS LINKUP LESS FEEDER BUS
							if (LM == 6):
								approach_cost[:, J] = FEEDER_BUS_CBD_FARE
							#     ADD APPROACH TIMES AND COSTS - EVERYTHING SHOULD NOW BE IN CENTS
							TVAR4[:, J] = TVAR4[:, J] + approach_cost[:, J]

					#     FIRST MODE CTA RAPID TRANSIT
					#     WHEN FOLLOWING STATEMENT IS TRUE A FULL FARE AND TRANSFER HAVE
					#     BEEN PAID, SO NO ADDED FARE IS NEEDED FOR BUS
					elif (LM < 7):
						TVAR4[:, J] = TVAR4[:, J] + approach_cost[:, J]

					#     ORIGIN OTHER THAN CHICAGO, ADDED FARE IS NOW AN RTA TRANSFER
					elif (ozone_type > 2):
						approach_cost[:, J] = PACE_BUS_FIRST_XFER_FARE
						TVAR4[:, J] = TVAR4[:, J] + approach_cost[:, J]

					#     CHICAGO ORIGIN, ADDED FARE IS CTA TRANSFER
					else:
						approach_cost[:, J] = CTA_FIRST_XFER_FARE
						TVAR4[:, J] = TVAR4[:, J] + approach_cost[:, J]

			#     END OF LOOP FOR FIVE APPROACH ALTERNATIVES

		# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
		#     EVALUATE APPROACH MODES AND SELECT THE BEST
		# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
		#     FOLLOWING CODE CHANGED BY EASH 12/6/93 TO REFLECT
		#     NEW TRIP TYPES
		#     TPTYPE = 1 HOME PRODUCTIONS TO WORK ATTRACTIONS
		#     TPTYPE = 2 HOME PRODUCTIONS TO NON-WORK/NON-HOME ATTRACTIONS
		#     TPTYPE = 3 NON-HOME PRODUCTIONS TO NON-HOME ATTRACTIONS
		for J in range(N_APPROACH_MODES):
			#     NO KISS-&-RIDE FOR NON-WORK TRIPS
			if (TPTYPE != HW and J == APPROACH_KISS_N_RIDE):
				continue
			#     NO PARK-AND RIDE OR KISS-&-RIDE AT THE WORK/OTHER
			#     ATTRACTION END FOR HOME BASED TRIPS
			if (TPTYPE != NH and I == BACK_END and (J == APPROACH_PARK_N_RIDE or J == APPROACH_KISS_N_RIDE)):
				continue
			#     NO PARK-&-RIDE OR KISS AND RIDE FOR NON-HOME TO NON-HOME TRIPS
			if (TPTYPE == NH and (J == APPROACH_PARK_N_RIDE or J == APPROACH_KISS_N_RIDE)):
				continue
			# --  FIND LOWEST COST APPROACH
			low_cost = (TVAR4[:, J] < TVAR5) & (TVAR4[:, J] > 0)
			TVAR5[low_cost] = TVAR4[low_cost, J]
			best_approach_mode[low_cost, I] = J
			best_drivetime[low_cost, I] = approach_drivetime[low_cost, J] + .5
			best_walktime[low_cost, I] = approach_walktime[low_cost, J] + .5
			best_cost[low_cost, I] = approach_cost[low_cost, J] + .5
			best_waittime[low_cost, I] = approach_waittime[J] + .5
			if trace:
				log.log(trace, f" DIRECTION {I} APPROACH TYPE {J} {APPROACH_MODE_NAMES.get(J)}")
				log.log(trace, f"  drivetime {approach_drivetime[:5, J]}")
				log.log(trace, f"  walktime  {approach_walktime[:5, J]}")
				log.log(trace, f"  cost      {approach_cost[:5, J]}")
				log.log(trace, f"  waittime  {approach_waittime[J]}")
				log.log(trace, f"  gen cost  {TVAR4[:5, J]}")
		if trace:
			log.log(trace, f" DIRECTION {I} BEST APPROACH TYPE {best_approach_mode[:5,I]}")


	#     ADD ORIGIN AND DESTINATION QUANTITIES AND PASS BACK TO TRIPS

	ae_drivetime = best_drivetime[:, 0] + best_drivetime[:, 1]
	ae_walktime = best_walktime[:, 0] + best_walktime[:, 1]
	ae_cost = best_cost[:, 0] + best_cost[:, 1]
	ae_waittime = best_waittime[:, 0] + best_waittime[:, 1]

	return ae_drivetime, ae_walktime, ae_cost, ae_waittime, best_approach_mode
