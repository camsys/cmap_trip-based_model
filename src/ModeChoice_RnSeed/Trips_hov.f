      SUBROUTINE TRIPS_HOV(ORIG)
      IMPLICIT INTEGER (A-Z)
C*******************
C  THIS SUBROUTINE PERFORMS THE OPERATION OF SIMULATING THE TRIPS
C       FROM THIS ORIGIN
C  THIS SUBROUTINE IS CALLED ONCE PER ZONE SELECTED FOR ANALYSIS.
C  NOTE THAT SUBROUTING IS NOT CALLED WHEN ZONE HAS NO TRIPS.
C  THIS IS THE VERSION USED WHEN HOV IS TURNED ON AND TOLL IS OFF.
C  THE MODEL FOR THE HOV CALUCLATION IS BORROWED FROM THE COMSIS-
C  MARYLAND NATIONAL CAPITAL PARKS AND PLANNING COMMISSION MODEL
C*******************
C
C     THIS IS THE INTERCHANGE VERSION OF THE MODEL AND THERE ARE
C     SEVERAL CODING CHANGES IN THIS SUBROUTINE FROM THE TRIP
C     END VERSION
C     THERE IS AN ADDITIONAL LOOP OUTSIDE THE DO LOOP FOR ITERATIONS
C     FOR ALL DESTINATIONS WITH INTERCHANGES FROM THE ORIGIN ZONE
C     AS THE SIMULATION PROGRESSES, TWO FILES ARE WRITTEN, ONE FOR
C     SIMULATED TRANSIT TRIPS AND ONE FOR AUTO TRIPS
C     MODE CHOICE IS DONE USING BINARY MONTE CARLO SIMULATION
C
C     THIS VERSION INCLUDES AN OPTION TO SUBDIVIDE THE AUTO MODE
C     INTO ONE, TWO AND THREE PLUS PERSON AUTO OCCUPANCY.  THE MODEL
C     APPLIED IS THE COMSIS-MNCPPC OCCUPANCY MODEL.
C
C     LOGIC IS TO FIRST ESTIMATE AUTO-TRANSIT MODE CHOICE USING THE
C     CATS BINAARY MODEL MODE CHOICE COEFFICIENTS.  THIS MODE SHARE IS
C     THEN USED TO CALCULATE THE TRANSIT UTILITY IN THE COMSIS-MNCPPC
C     MODEL.
C
C ##  Heither 09-06-2016: Add ITER to program       
C ##  Heither 11-24-2017: Vectorized version of the code.   
C*******************
      INCLUDE 'Common_params.fi'
	INCLUDE 'Common_data.fI'
      INCLUDE 'Common_emme4bank.fi'
	INCLUDE 'Common_approach_model.fi'
	INCLUDE 'Common_cbdparking.fi'

      INCLUDE 'Common_auto_params.fi'
      INCLUDE 'Common_auto_emme4bank.fi'

	REAL*4 RAN_NUM1, RAN_NUM2, TTP1, TTP2, TTPAVG
	REAL*4 D, DD, ACOST, ACOST0, ACOST1, ACOST2
      REAL*4 SC, SCA, SC1, SC2, SC3, SCT, HEADER 
	REAL, DIMENSION(ITER) :: INCOME, AUTOCC, ACOSTVC, ARNUMA, ARNUMT, TTP
	REAL, DIMENSION(ITER) :: RAN_NUM, CAPK, WALK, ACOST1VC, ACOST2VC, USR
      REAL, DIMENSION(ITER) :: UAUTO, UTRAN, U2, U3, U4PLUS, UDA, TEMP1
      REAL, DIMENSION(ITER) :: TEMP2
	INTEGER,DIMENSION (ITER) :: INTOCC, C0, C1, C2PLUS
      INTEGER,DIMENSION (ITER) :: EC, EIVT, EOVT
      INTEGER, ALLOCATABLE :: SEED(:)
      INTEGER SIZE
	  
C*******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************
      REAL*4 HOV_BIAS1, HOV_BIAS2
C*******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************

	LOGICAL*4 CBDIND, NOTTRO, NOTTRD
C
C     REOPEN EMMEBANK
C
C*******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************
C      OPEN (UNIT=32, FILE='EMMEBANK',
C     A  ACCESS='DIRECT',RECL=1,STATUS='OLD')
C
C      OPEN (UNIT=42, FILE='AUTOBANK',
C     A  ACCESS='DIRECT',RECL=1,STATUS='OLD')
C*******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************
      OPEN (UNIT=951, FILE='emmemat/'//TEMX_AUTO,
     A  ACCESS='DIRECT',RECL=1,STATUS='OLD')
      OPEN (UNIT=952, FILE='emmemat/'//TEMX_TRANSIT,
     A  ACCESS='DIRECT',RECL=1,STATUS='OLD')
      OPEN (UNIT=953, FILE='emmemat/'//TEMX_SOV1,
     A  ACCESS='DIRECT',RECL=1,STATUS='OLD')
      OPEN (UNIT=954, FILE='emmemat/'//TEMX_HOV2,
     A  ACCESS='DIRECT',RECL=1,STATUS='OLD')
      OPEN (UNIT=955, FILE='emmemat/'//TEMX_HOV3,
     A  ACCESS='DIRECT',RECL=1,STATUS='OLD')
C
C     ALL TRIPS ARE SIMULATED FOR EACH DESTINATION
C## Heither, 11-24-2017: use ZONES r/t 3000
      DO II=1,ZONES
          ATRIP(II) = 0.0		!! Auto trips
          ATRIP1(II) = 0.0	!! SOV Auto trips
          ATRIP2(II) = 0.0	!! HOV2 Auto trips
          ATRIP3(II) = 0.0	!! HOV3+ Auto trips
          TTRIP(II) = 0.		!! Transit trips
      ENDDO
C
C  REVISED COUNTERS
      SC = 0.0				!! Person trip counter
      SCA = 0.0				!! Auto Person trip counter
      SC1 = 0.0 			!! SOV Auto Person trip counter
	SC2 = 0.0 			!! HOV2 Auto Person trip counter
	SC3 = 0.0				!! HOV3+ Auto Person trip counter
	SCT = 0.0 			!! Transit Person trip counter
C
C  THE NEXT FEW STATEMENTS DETERMINE WHETHER OR NOT THIS ZONE
C               IS SERVED BY A TRANSIT MODE
C                      IF NOT, DON'T BOTHER SIMULATING
      NOTTRO = .TRUE.
C
C    NEED ONE DESTINATION ZONE ACCESSIBLE BY TRANSIT
C## Heither, 11-24-2017: use ZONES r/t 3000
      DO I=1,ZONES
		IF (DZOI(I)) THEN
          IF ((DISTR(I,1,3).NE.999).OR.(DISTR(I,2,3).NE.999).OR.
     A      (DISTR(I,3,3).NE.999).OR.(DISTR(I,4,3).NE.999)) THEN
            IF(PMD(I).GE.3) THEN
              NOTTRO = .FALSE.
            ENDIF
          ENDIF
        ENDIF
      ENDDO
C
C     CHECK ORIGIN FOR DISTR PARAMETERS
C
      IF ((DISTR(ORIG,1,3).EQ.999).AND.(DISTR(ORIG,2,3).EQ.999).AND.
     A  (DISTR(ORIG,3,3).EQ.999).AND.(DISTR(ORIG,4,3).EQ.999).AND.
     B  (DISTR(ORIG,5,3).EQ.999)) THEN
        NOTTRO = .TRUE.
      ENDIF 

      IF (NOTTRO) THEN
          WRITE(31,505) ORIG
      ENDIF

  505 FORMAT(' ZONE ',I5,' NOT SERVED BY A TRANSIT MODE(4-8)')
C****************************************
C  THIS BEGINS THE SIMULATION OF TRIPS  *
C****************************************
C
C     START OF LOOP BY DESTINATION ZONE
      DO DEST=1,ZONES
C
C##     PROCESS DESTINATION ZONE ONLY IF:
C##       1.  THERE ARE TRIPS BETWEEN ORIGIN AND DESTINATION ZONES, AND
C##       2.  THE DESTINATION ZONE IS ACCESSIBLE
C##       OTHERWISE: TTRIP=0, ATRIP1=0, ATRIP2=0, ATRIP3=0
		IF (DZOI(DEST) .AND. PTRIP(DEST).GT.0) THEN
              
C##### Heither, 01-12-2019: GET ZONAL INTERCHANGE SEED VALUE 
              SEED1 = ((ORIG-1)*mcent) + DEST
              READ(40, REC=SEED1) ZNINTSD
              CALL RANDOM_SEED(SIZE=SIZE)
              ALLOCATE(SEED(SIZE))
              SEED = ZNINTSD
              CALL RANDOM_SEED(PUT=SEED)   !!! SET SEED VALUE
              IF (ORIG.EQ.1 .AND. DEST.LE.10) THEN
                  WRITE (31, '(A,3I8)')  '  ORIG,DEST,RANDOM SEED',
     AORIG,DEST,ZNINTSD
              ENDIF
              DEALLOCATE(SEED)            
                         
C===========================================================================
C## PART 1: AUTO	  
C===========================================================================	  
C  SET THE MINIMUM LINE HAUL DRIVING TIME TO ONE MINUTE
			IF(ZLHT(DEST).LT.1) ZLHT(DEST)=1
			IF(ZLHT1(DEST).LT.1) ZLHT1(DEST)=1
			IF(ZLHT2(DEST).LT.1) ZLHT2(DEST)=1
C
C  RUN A CHECK FOR A CBD DESTINATION
			CBDIND=.FALSE.
			IF(ZCBD(DEST)) CBDIND=.TRUE.
C*******************
C  NEXT COMPUTE THE AUTO OPERATING COSTS IN CENTS
C*******************
			D=ZLHD(DEST)
			DD=ZLHT(DEST)
			IF (D .GT. 0 .AND. DD .GT. 0) THEN 
				CALL AUTCST(ORIG,DEST,D,DD,ACOST)
			ELSE
				ACOST = 0.0
			ENDIF
C     SOV COSTS
			D=ZLHD1(DEST)
			DD=ZLHT1(DEST)
			IF (D .GT. 0 .AND. DD .GT. 0) THEN 
				CALL AUTCST(ORIG,DEST,D,DD,ACOST1)
			ELSE
				ACOST1 = 0.0
			ENDIF
C     HOV COSTS
			D=ZLHD2(DEST)
			DD=ZLHT2(DEST)
			IF (D .GT. 0 .AND. DD .GT. 0) THEN 
				CALL AUTCST(ORIG,DEST,D,DD,ACOST2)
			ELSE
				ACOST2 = 0.0
              ENDIF
C##    VECTORIZE AUTO COSTS, Heither 11-29-2017
              ACOSTVC=ACOST
              ACOST1VC=ACOST1
              ACOST2VC=ACOST2              
C              
C*******************
C  THE NEXT SECTION OBTAINS THE INCOME OF THE TRIPMAKER
C      IF THIS A HOME BASED TRIP START WITH THE ZONAL MEDIAN INCOME
C         AT THE DESTINATION RATHER THAN AT THE ORIGIN
C*******************

C## Heither, 01-11-2018: the following INCDIS change was implemented in TRIPS but not TRIPS_HOV until now. 
C*******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************
			IF (TPTYPE .NE. 3) THEN
				MEDINC=INC(ORIG)*100
				CALL INCDIS(ORIG,DEST,MEDINC,INCOME)
C
C  IF WE HAVE A NON-HOME BASED TRIP USE THE AVERAGE REGIONAL
C     2007 ACS HH INCOME FOR CHICAGO METRO AREA
			ELSE
				INCOME = 59300.
			ENDIF				

C PARKING NEXT
C  THE CALL TO PRKCST OBTAINS THE COST OF PARKING FOR A HIGHWAY TRIP
C    CAPK CONTAINS COST OF PARKING
C    WALK IS THE AVERAGE WALK TIME FROM THE AUTO TO THE DESTINATION
			INTOCC = 0
			CALL PRKCST(ORIG,DEST,INCOME,CAPK,WALK,INTOCC)
C
			ACOSTVC=0.5*CAPK+ACOST+AFC1	!! TOTAL COST OF THE HIGHWAY TRIP ON PATH WITHOUT HOV LANE IN PLACE
			ACOST1VC=0.5*CAPK+ACOST1+AFC1	!! COST OF THE HIGHWAY TRIP ON THE SOV PATH WITH HOV LANE IN PLACE
			ACOST2VC=0.5*CAPK+ACOST2+AFC1	!! COST OF THE HIGHWAY TRIP ON THE HOV PATH WITH HOV LANE IN PLACE
C
C     AUTO OCCUPANCY DETERMINED BY THE FOLLOWING OPTIONS/PRIORITIES:
C       1.  FROM PARKING MODEL WHEN DEST IS A CBD PARKING ZONE
C       2.  OCCUPANCIES INPUT VIA TABLE FROM EMMEBANK
C       3.  OBTAINED FROM M023 INPUT
			AUTOCC = ZOCC(DEST)
C
C     FOLLOWING CODE ALLOWS USER TO INPUT AUTO OCCUPANCIES
			IF (AOCC) AUTOCC = COCC(DEST)
C
C     OCCUPANCY RETURNED FROM PRKCBD ALWAYS IS USED
			WHERE(INTOCC .GT. 0) AUTOCC=INTOCC 
C
C    COMPUTE FINAL AUTO COST
			ACOST0 = ACOST	!! TOTAL COST OF THE HIGHWAY TRIP ON PATH WITHOUT HOV LANE IN PLACE FOR THE HOV CALCULATIONS
			ACOSTVC=ACOSTVC/AUTOCC			
C
C     COSTS ARE DISCOUNTED FROM 1990 TO 1970 (CALIBRATION YEAR) IN CATS 
C     MODEL
C*******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************
C     COSTS ARE ESTIMATED IN CURRENT DOLLARS FOR I290 RUNS AND MODEL 
C     COST COEFFICIENT HAS BEEN ADJUSTED ACCORDINGLY, SO DISCNT IS 
C     INPUT AS 1.00
C*******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************
			ACOSTVC = ACOSTVC * DISCNT
			ACOST1VC = ACOST1VC * DISCNT
			ACOST2VC = ACOST2VC * DISCNT
			ACOST0 = ACOST0 * DISCNT              
C
C    COMPUTE AUTO UTILITY
C
			IF(.NOT.CBDIND) THEN
				ARNUMA= COEFF1(1)*ZLHT(DEST)+
     A          COEFF1(2)*ACOSTVC+
     B          COEFF1(3)*WALK
			ENDIF
			IF(CBDIND) THEN
				ARNUMA= COEFF2(1)*ZLHT(DEST)+
     A          COEFF2(2)*ACOSTVC+
     B          COEFF2(3)*WALK
			ENDIF
C
C     WRITE UTILITIES IF TRACE IS TURNED ON
C
			IF (TRACE) THEN
				WRITE (31, '(/A)') ' HIGHWAY INPUTS IN SUBROUTINE TRIPS'
				WRITE (31, '(A,I6)') '   ORIGIN ZONE=', ORIG
				WRITE (31, '(A,I6)') '   DESTINATION ZONE=', DEST
				WRITE (31, '(A,I8)') '   HOUSEHOLD MEDIAN INCOME=', MEDINC
				WRITE (31, '(A,F8.0)') '   HOUSEHOLD INCOME=', INCOME(1)
				WRITE (31, '(A,F8.3)') '   HIGHWAY TRAVEL TIME (ZLHT)=',
     A	ZLHT(DEST)
				WRITE (31, '(A,F8.3)') '   HIGHWAY DISTANCE (ZLHD)=',ZLHD(DEST)
				WRITE (31, '(A,F8.3)') '   AUTO OCCUPANCY (AUTOCC)= ',AUTOCC(1)
				WRITE (31, '(A,F8.3)') '   CBD PARKING COST (CAPK)= ',CAPK(1)
				WRITE (31, '(A,I8)')   '   AUTO FIXED COST (AFC1)= ',AFC1
				WRITE (31, '(A,F8.3)') '   FINAL AUTO COST (ACOST)= ',ACOSTVC(1)
				WRITE (31, '(A,F8.3)') '   WALK TIME (WALK)= ',WALK(1)
				WRITE (31, '(A,F8.3)') '   HIGHWAY UTILITY (ARNUMA)=',ARNUMA(1)
			ENDIF

C===========================================================================
C## PART 2: TRANSIT	  
C===========================================================================				
C
C  VERIFY THAT TRANSIT WILL GO TO THE DESTINATION ZONE
			NOTTRD = .TRUE.
			IF(PMD(DEST).GE.3) THEN
				IF ((DISTR(DEST,1,3).NE.999).OR.(DISTR(DEST,2,3).NE.999).OR.
     A    (DISTR(DEST,3,3).NE.999).OR.(DISTR(DEST,4,3).NE.999)) THEN
					NOTTRD = .FALSE.
				ENDIF
			ENDIF
C
			IF ((.NOT. NOTTRO) .AND. (.NOT. NOTTRD)) THEN 
				CALL TRAPP(ORIG,DEST,EIVT,EOVT,EC)	!! RETURN THE TRANSIT APPROACH TIMES
C
				HEADER=0.5*HWAY(DEST)
C
C     COSTS ARE DISCOUNTED FROM 1990 TO 1970 (CALIBRATION YEAR)
C*******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************
C     COSTS ARE ESTIMATED IN CURRENT DOLLARS AND MODEL COST COEFFICIENT 
C     HAS BEEN ADJUSTED ACCORDINGLY, SO DISCNT = 1.00
C*******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************
				FARE(DEST) = FARE(DEST) * DISCNT
				EC = EC * DISCNT
C
C   COMPUTE TRANSIT UTILITY:  NOTE THAT SIGN ON TRANSIT BIAS COEFFICIENT
C   IS NEGATIVE BECAUSE IT IS ENTERED AS MINUS NUMBER.  TRANSIT BIAS IS
C   AN ADDITIONAL TRANSIT COST, I.E., TRANSIT IS LESS THAN 50 PERCENT
C   EVERYTHING ELSE BEING EQUAL.
				IF(.NOT.CBDIND) THEN
					ARNUMT= COEFF1(1)*IVT(DEST)+
     A            COEFF1(2)*(FARE(DEST)+EC)+
     B            COEFF1(3)*EIVT-
     C            COEFF1(4)+
     D            COEFF1(5)*(OVT(DEST)+EOVT)+
     E            COEFF1(6)*HEADER
				ENDIF
C
				IF(CBDIND) THEN
					ARNUMT= COEFF2(1)*IVT(DEST)+
     A            COEFF2(2)*(FARE(DEST)+EC)+
     B            COEFF2(3)*EIVT-
     C            COEFF2(4)+
     D            COEFF2(5)*(OVT(DEST)+EOVT)+
     E            COEFF2(6)*HEADER
				ENDIF
C
C     WRITE UTILITIES IF TRACE IS TURNED ON
				IF (TRACE) THEN
					WRITE (31, '(/A)') ' TRANSIT INPUTS IN SUBROUTINE TRIPS'
					WRITE (31, '(A,I6)') '   ORIGIN ZONE=', ORIG
					WRITE (31, '(A,I6)') '   DESTINATION ZONE=', DEST
					WRITE (31, '(A,F8.3)') '   FIRST MODE (FMD)=',FMD(DEST)
					WRITE (31, '(A,F8.3)') '   LAST MODE (LMD)=',LMD(DEST)
					WRITE (31, '(A,F8.3)') '   PRIORITY MODE (PMD)=',PMD(DEST)
					WRITE (31, '(A,F8.3)') '   IN-VEHICLE TIME (IVT)=',IVT(DEST)
					WRITE (31, '(A,F8.3)') '   TRANSIT FARE (FARE)=',FARE(DEST)
					WRITE (31, '(A,F8.3)') '   OUT-OF-VEHICLE TIME (OVT)=',
     A      OVT(DEST)
					WRITE (31, '(A,F8.3)') '   INITIAL WAIT TIME (1/2 HEADWAY)=',
     A      HEADER
					WRITE (31, '(A,I6)') '   APPROACH IVT (EIVT)=',EIVT(1)
					WRITE (31, '(A,I6)') '   APPROACH TRANSIT FARE (EC)=',EC(1)
					WRITE (31, '(A,I6)') '   APPROACH OVT (EOVT)=',EOVT(1)
					WRITE (31, '(A,F8.3)') '   TRANSIT UTILITY (ARNUMT)=',ARNUMT(1)
				ENDIF
				TTP = EXP(-ARNUMT)/(EXP(-ARNUMT)+EXP(-ARNUMA))	!! PROBABILITY THIS TRIP IS BY TRANSIT IN THE CATS MODEL
			ELSE
				TTP = 0.0
			ENDIF

C===========================================================================
C## PART 3: COMSIS-MNCPPC MODEL	  
C===========================================================================	
C
C   1.  DETERMINE AUTO OWNERSHIP LEVEL TO USE IN DRIVE ALONE-SHARED RIDE MODEL
			CALL RANDOM_NUMBER(RAN_NUM)
			C0 = 0
			C1 = 0
			C2PLUS = 1
			WHERE(RAN_NUM <= COWN1(ORIG))
				C2PLUS = 0
				C1 =1
			END WHERE
			WHERE(RAN_NUM <= COWN0(ORIG))
				C1 =0
				C0 =1
			END WHERE
C   2.  U2, U3, U4PLUS ARE SHARED RIDE UTILITIES COMPUTED WITHOUT HOV LANE
C
C*******************  RWE CHANGE FOR I290 OCTOBER 2009  ****************
C     ORIGINAL COMSIS COST COEFFICIENTS ADJUSTED TO 2009 (*0.55)
			IF (CBDIND) THEN
				HOV_BIAS1 = HOV_CBDBIAS(1)
				HOV_BIAS2 = HOV_CBDBIAS(2)
			ELSE
				HOV_BIAS1 = HOV_BIAS(1)
				HOV_BIAS2 = HOV_BIAS(2)
			ENDIF

			U2 = -0.138*ZLHT(DEST) - 0.00275*ACOST0/2
			U3 = -0.138*ZLHT(DEST) - 0.00275*ACOST0/3 - HOV_BIAS1
			U4PLUS = -0.138*ZLHT(DEST) - 0.00275*ACOST0/5.56 - HOV_BIAS1
C ## Heither, 02-27-2017: Coded added so numeric precision not exceeded
              WHERE(U2<-85.0) U2 = -85.0
              WHERE(U3<-85.0) U3 = -85.0
              WHERE(U4PLUS<-85.0) U4PLUS = -85.0
C ##			U2 = MAX(U2,-85.0)    
C ##			U3 = MAX(U3,-85.0)  
C ##			U4PLUS = MAX(U4PLUS,-85.0)       
C
C   3.  UDA = DRIVE ALONE UTILITY (NO HOV LANE)
C       USR = SHARED RIDE UTILITY (NO HOV LANE)
			UDA = -0.078*ZLHT(DEST) - 0.001155*ACOST0
			USR = LOG(EXP(U2)+EXP(U3)+EXP(U4PLUS))
			USR = 0.536*USR + 3.204*C0 - 1.941*C1 - 2.373*C2PLUS - HOV_BIAS2
C			
C*******************  RWE CHANGE FOR I290 OCTOBER 2009  ****************
C   4.  UAUTO = AUTO UTILITY (NO HOV LANE)
			UAUTO = LOG(EXP(UDA)+EXP(USR))
			UAUTO = 0.571*UAUTO - 0.141*WALK
C			
C   5.  NOW ESTIMATE EQUIVALENT TRANSIT UTILITY IN COMSIS MODEL (UTRAN)
C       WITHOUT HOV LANE IN PLACE
C
C*******************  RWE CHANGE FOR I290 OCTOBER 2009  ****************
C##			IF ((.NOT. NOTTRO) .AND. (.NOT. NOTTRD) .AND. (TTP.GT.0.0)) THEN
C*******************  RWE CHANGE FOR I290 OCTOBER 2009  ****************
C##				UTRAN = TTP*EXP(UAUTO)/(1-TTP)
C      WRITE (31,*) UTRAN, TTP, UAUTO, NOTTRO, NOTTRD
C##				UTRAN = LOG(UTRAN)
C##			ELSE
C##				UTRAN = 0.0
C##              ENDIF           
              IF ((.NOT. NOTTRO) .AND. (.NOT. NOTTRD)) THEN
                  WHERE(TTP>0)
                      UTRAN = TTP*EXP(UAUTO)/(1-TTP)
                      UTRAN = LOG(UTRAN)
                  ELSEWHERE
                      UTRAN = 0.0
                  END WHERE
              ELSE
                  UTRAN = 0.0
              ENDIF
          
          
C
C     WRITE COMSIS MODEL PARAMETERS IF TRACE IS TURNED ON
			IF (TRACE) THEN
				WRITE (31, '(/A)') ' COMSIS MODEL PARAMETERS WITHOUT HOV LANE'
				WRITE (31, '(A,I6)') '   ORIGIN ZONE=', ORIG
				WRITE (31, '(A,I6)') '   DESTINATION ZONE=', DEST
				WRITE (31, '(A,3F8.4)') '   TRANSIT PROBABILITY: 3 (TTP)=', TTP(1),
     A TTP(2), TTP(3)             
				WRITE (31, '(A,3F8.4)') '   RANDOM NUMBER (RAN_NUM):3 =', RAN_NUM(1)
     A,RAN_NUM(2), RAN_NUM(3)
				WRITE (31, '(A,3F8.4)')'   CAR OWNERSHIP PROBABILITIES (COWN0, C
     AOWN1, COWN2)=', COWN0(DEST), COWN1(DEST), COWN2(DEST)
				WRITE (31, '(A,6I2)')  '   CAR OWNERSHIP VARIABLES (C0, C1, C2PL
     AUS)x2=', C0(1), C1(1), C2PLUS(1), C0(2), C1(2), C2PLUS(2)           
				WRITE (31, '(A,F6.2)')  '   AUTO COST WITHOUT HOV LANE (ACOST0)=
     A', ACOST0
				WRITE (31, '(A,3F8.3)')  '   SHARED RIDE UTILITIES (U2, U3, U4PL
     AUS)=', U2(1), U3(1), U4PLUS(1)
				WRITE (31, '(A,4F8.3)')  '   DRIVE ALONE UTILITY (UDA)=', UDA(1),
     AUDA(2), UDA(3), UDA(4)            
				WRITE (31, '(A,4F8.3)') '   COMPOSITE SHARED RIDE UTILITY (USR)='
     A, USR(1), USR(2), USR(3), USR(4)
				WRITE (31, '(A,2F8.3)') '   WALK TIME (WALK)= ',WALK(1), WALK(2)
				WRITE (31, '(A,5F8.3)')  '   AUTO UTILITY (UAUTO)=', UAUTO(1),
     AUAUTO(2), UAUTO(3), UAUTO(4), UAUTO(5)           
				WRITE (31, '(A,5F8.3)')  '   DERIVED TRANSIT UTILITY (UTRAN)=',
     AUTRAN(1), UTRAN(2), UTRAN(3), UTRAN(4), UTRAN(5)
      ENDIF
C   
C   6.  NEXT STEP IS TO RECALCULATE AUTO SIDE UTILITIES WITH HOV LANE IN PLACE      
			U2 = -0.138*ZLHT1(DEST) - 0.00275*ACOST1/2.0
			IF (HOV2)  U2 = -0.138*ZLHT2(DEST) - 0.00275*ACOST2/2.0
			U3 = -0.138*ZLHT2(DEST) - 0.00275*ACOST2/3 - HOV_BIAS1
			U4PLUS = -0.138*ZLHT2(DEST) - 0.00275*ACOST2/5.56 - HOV_BIAS1
C ## Heither, 02-27-2017: Coded added so numeric precision not exceeded
              WHERE(U2<-85.0) U2 = -85.0
              WHERE(U3<-85.0) U3 = -85.0
              WHERE(U4PLUS<-85.0) U4PLUS = -85.0
C ##			U2 = MAX(U2,-85.0)    
C ##			U3 = MAX(U3,-85.0)  
C ##			U4PLUS = MAX(U4PLUS,-85.0)       
C
C   7.  UDA = DRIVE ALONE UTILITY (WITH HOV LANE)
C       USR = SHARED RIDE UTILITY (WITH HOV LANE)
			UDA = -0.078*ZLHT1(DEST) - 0.001155*ACOST1
			USR = LOG(EXP(U2)+EXP(U3)+EXP(U4PLUS))
			USR = 0.536*USR + 3.204*C0 - 1.941*C1 - 2.373*C2PLUS - HOV_BIAS2
C
C   8.  UAUTO = AUTO UTILITY (WITH HOV LANE)
			UAUTO = LOG(EXP(UDA)+EXP(USR))
			UAUTO = 0.571*UAUTO - 0.141*WALK			
C
C   9.  TTP IS NOW THE REVISED PROBABILITY THIS TRIP IS BY TRANSIT IN
C       COMSIS-MNCPPC MODEL
C    
C*******************  RWE CHANGE FOR I290 OCTOBER 2009  ****************
C##			IF ((.NOT. NOTTRO) .AND. (.NOT. NOTTRD) .AND. (TTP.GT.0.0)) THEN
C*******************  RWE CHANGE FOR I290 OCTOBER 2009  ****************
C##				TTP=EXP(UTRAN)/(EXP(UAUTO)+EXP(UTRAN))
C##			ELSE
C##				TTP=0.0
C##              ENDIF
              IF ((.NOT. NOTTRO) .AND. (.NOT. NOTTRD)) THEN
                  WHERE(TTP>0)
                      TTP=EXP(UTRAN)/(EXP(UAUTO)+EXP(UTRAN))
                  ELSEWHERE
                      TTP=0.0
                  END WHERE
              ELSE
                  TTP=0.0
              ENDIF

C     WRITE COMSIS MODEL PARAMETERS IF TRACE IS TURNED ON
			IF (TRACE) THEN
				WRITE (31, '(/A)') ' COMSIS MODEL PARAMETERS WITH HOV LANE'
				WRITE (31, '(A,I6)') '   ORIGIN ZONE=', ORIG
				WRITE (31, '(A,I6)') '   DESTINATION ZONE=', DEST
				WRITE (31, '(A,5F8.4)') '   TRANSIT PROBABILITY (TTP)=', TTP(1), 
     ATTP(2), TTP(3), TTP(4), TTP(5)
				WRITE (31, '(A,5F8.2)')  '   SOV AUTO COST WITH HOV LANE (ACOST1VC)
     A=', ACOST1VC(1), ACOST1VC(2), ACOST1VC(3), ACOST1VC(4), 
     BACOST1VC(5)
				WRITE (31, '(A,5F8.2)')  '   HOV AUTO COST WITH HOV LANE (ACOST2VC)
     A=', ACOST2VC(1), ACOST2VC(2), ACOST2VC(3), ACOST2VC(4), 
     BACOST2VC(5)
				WRITE (31, '(A,12F8.3)')  '   SHARED RIDE UTILITIES (U2, U3, U4PL
     AUS)=', U2(1), U3(1), U4PLUS(1), U2(2), U3(2), U4PLUS(2), U2(3), 
     B U3(3), U4PLUS(3), U2(4), U3(4), U4PLUS(4)  
				WRITE (31, '(A,5F8.3)')  '   DRIVE ALONE UTILITY (UDA)=', UDA(1),
     AUDA(2), UDA(3), UDA(4), UDA(5)              
				WRITE (31, '(A,5F8.3)') '   COMPOSITE SHARED RIDE UTILITY (USR)='
     A, USR(1), USR(2), USR(3), USR(4), USR(5) 
				WRITE (31, '(A,5F8.3)') '   WALK TIME (WALK)= ',WALK(1), WALK(2),
     A WALK(3), WALK(4), WALK(5)
				WRITE (31, '(A,5F8.3)')  '   AUTO UTILITY (UAUTO)=', UAUTO(1),
     A UAUTO(2), UAUTO(3), UAUTO(4), UAUTO(5)
				WRITE (31, '(A,5F8.3)')  '   DERIVED TRANSIT UTILITY (UTRAN)=',
     A UTRAN(1), UTRAN(2), UTRAN(3), UTRAN(4), UTRAN(5)
      ENDIF
C
C     SEE WHETHER THIS TRIP IS AN AUTO OR TRANSIT TRIP
C
C************ RWE CHANGE FOR I290 ROUND 3 MODELS MARCH 2013 ************
C       SIMULATE 1000 TRIPS TO OBTAIN MODE SHARES
C***********************************************************************

C## CALCULATE VARIOUS PROBABILITIES
              TTPAVG = SUM(TTP)/ITER  !! AVERAGE TRANSIT PROBABILITY
C##			TTRIP(DEST) = TTRIP(DEST) + TTP
C##			ATRIP(DEST) = ATRIP(DEST) + (1.0-TTP)              
              TTRIP(DEST) = TTRIP(DEST) + TTPAVG
              ATRIP(DEST) = ATRIP(DEST) + (1.0-TTPAVG)
			TEMP1 = EXP(UDA)/(EXP(UDA)+EXP(USR))
              TTP1 = SUM(TEMP1)/ITER           
C##			TTP1 = TTP1*(1.0-TTP)	!! THE PROBABILITY THIS TRIP IS BY SOV
              TTP1 = TTP1*(1.0-TTPAVG)	!! THE PROBABILITY THIS TRIP IS BY SOV
			ATRIP1(DEST) = ATRIP1(DEST) + TTP1
C             
C##			TTP2 = EXP(U2)/(EXP(U2)+EXP(U3)+EXP(U4PLUS))
              TEMP2 = EXP(U2)/(EXP(U2)+EXP(U3)+EXP(U4PLUS))
              TTP2 = SUM(TEMP2)/ITER         
C##			TTP2 = TTP2*(1.0-TTP-TTP1)	!! THE PROBABILITY THIS TRIP IS BY TWO PERSON HOV
              TTP2 = TTP2*(1.0-TTPAVG-TTP1)	!! THE PROBABILITY THIS TRIP IS BY TWO PERSON HOV
			ATRIP2(DEST) = ATRIP2(DEST) + TTP2
C##			ATRIP3(DEST) = ATRIP3(DEST) + (1.0-TTP-TTP1-TTP2)
              ATRIP3(DEST) = ATRIP3(DEST) + (1.0-TTPAVG-TTP1-TTP2)
C
C     WRITE COMSIS MODEL AUTO SUBMODE CHOICE PARAMETERS IF TRACE IS TURNED ON
			IF (TRACE) THEN
				WRITE (31, '(/A)') ' COMSIS SUBMODE CHOICES WITH HOV LANE'
				WRITE (31, '(A,I6)') '   ORIGIN ZONE=', ORIG
				WRITE (31, '(A,I6)') '   DESTINATION ZONE=', DEST
				WRITE (31, '(A,F9.6)') '   TRANSIT PROBABILITY (TTPAVG)=', TTPAVG 
				WRITE (31, '(A,F9.6)') '   SOV PROBABILITY (TTP1)=', TTP1
				WRITE (31, '(A,F9.6)') '   TWO PERSON PROBABILITY (TTP2)=', TTP2
				WRITE (31, '(A,F8.3)') '   CURRENT TRANSIT TRIPS (TTRIP)=',
     A    TTRIP(DEST)
				WRITE (31, '(A,F8.3)') '   CURRENT SOV TRIPS (ATRIP1)=',
     A    ATRIP1(DEST)
				WRITE (31, '(A,F8.3)') '   CURRENT TWO PERSON HOV TRIPS (ATRIP2)
     A=', ATRIP2(DEST)
				WRITE (31, '(A,F8.3)') '   CURRENT THREE OR MORE PERSON HOV TRIP
     AS (ATRIP3)=', ATRIP3(DEST)
                  WRITE (31, '(A,5F9.6)') '   TRANSIT PROBABILITY SIMULA
     ATIONS (TTP)=', TTP(1), TTP(2), TTP(3), TTP(4), TTP(5) 
			ENDIF
	  
		ENDIF	!! END OF DESTINATION ZONE PROCESSING

C     WRITE TRIPS INTO EMMEBANK
C       MF23 = AUTO TRIPS
C       MF24 = TRANSIT TRIPS
C       MF05 = SOV PERSON TRIPS
C       MF06 = TWO PERSON HOV PERSON TRIPS
C       MF07 = THREE OR MORE PERSON PERSON TRIPS
C
		P = ORIG
		Q = DEST
		REC1 = ((P-1)*mcent) + Q 
		SC = SC + PTRIP(Q)      !! Total Person trips
		PTRIP_TOT = PTRIP_TOT + PTRIP(Q)
          
C##		ATRIP(Q) = ATRIP(Q)*PTRIP(Q)/ITER
          ATRIP(Q) = ATRIP(Q)*PTRIP(Q)
		WRITE(951, REC=REC1) ATRIP(Q)
		SCA = SCA + ATRIP(Q)    !! Auto Person trips
		ATRIP_TOT = ATRIP_TOT + ATRIP(Q)
C
C##		TTRIP(Q) = TTRIP(Q)*PTRIP(Q)/ITER
          TTRIP(Q) = TTRIP(Q)*PTRIP(Q)
		WRITE(952, REC=REC1) TTRIP(Q)
		SCT = SCT + TTRIP(Q)    !! Transit Person trips
		TTRIP_TOT = TTRIP_TOT + TTRIP(Q)
C
C##		ATRIP1(Q) = ATRIP1(Q)*PTRIP(Q)/ITER
          ATRIP1(Q) = ATRIP1(Q)*PTRIP(Q)
		WRITE(953, REC=REC1) ATRIP1(Q)
		SC1 = SC1 + ATRIP1(Q)   !! SOV Auto Person trips
		SOVTRIP_TOT = SOVTRIP_TOT + ATRIP1(Q)
      
C##		ATRIP2(Q) = ATRIP2(Q)*PTRIP(Q)/ITER
          ATRIP2(Q) = ATRIP2(Q)*PTRIP(Q)
		WRITE(954, REC=REC1) ATRIP2(Q)
		SC2 = SC2 + ATRIP2(Q)   !! HOV2 Auto Person trips
		HOV2TRIP_TOT = HOV2TRIP_TOT + ATRIP2(Q)
      
C##		ATRIP3(Q) = ATRIP3(Q)*PTRIP(Q)/ITER
          ATRIP3(Q) = ATRIP3(Q)*PTRIP(Q)
		WRITE(955, REC=REC1) ATRIP3(Q)
		SC3 = SC3 + ATRIP3(Q)   !! HOV3+ Auto Person trips
		HOV3TRIP_TOT = HOV3TRIP_TOT + ATRIP3(Q) 
	  ENDDO		!! END OF DESTINATION ZONE LOOP
C
C  THIS ZONE IS DONE
C
	  WRITE(31,80803) ORIG,SC
	  WRITE(31,80804) ORIG,SCT
      WRITE(31,80805) ORIG,SCA
      WRITE(31,80806) ORIG,SC1
      WRITE(31,80807) ORIG,SC2
      WRITE(31,80808) ORIG,SC3

80803 FORMAT(' ZONE ',I5,' HAS SIMULATED ',F12.4,' PERSON TRIPS')
80804 FORMAT(' ZONE ',I5,' HAS ',F12.4,' TRANSIT PERSON TRIPS')
80805 FORMAT(' ZONE ',I5,' HAS ',F12.4,' AUTO PERSON TRIPS')

80806 FORMAT(' ZONE ',I5,' HAS ',F12.4,' SOV PERSON TRIPS')
80807 FORMAT(' ZONE ',I5,' HAS ',F12.4,' TWO PERSON HOV TRIPS')
80808 FORMAT(' ZONE ',I5,' HAS ',F12.4,' THREE OR MORE PERSON HOV TRIPS'
     A)
     
	  CLOSE (951)
	  CLOSE (952)
	  CLOSE (953)
	  CLOSE (954)
	  CLOSE (955)
      RETURN
      END
	  