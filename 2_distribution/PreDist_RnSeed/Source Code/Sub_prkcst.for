      SUBROUTINE PRKCST(ORIG,DEST,INCOME,COST,WALK,INTOCC)
      IMPLICIT INTEGER (A-Z)
C*******************
C## Revised for vectorized calculations - Heither, 11-02-2017
C  THIS SUBROUTINE WILL CALCULATE THE PARKING COST FOR THE AUTO TRIP
C       FOR THE CBD, THE ROUTINE WILL USE THE HOURLY RATE AND THE 
C           PROBABILITY OF PARKING AT THAT RATE IN THE CBD 
C       FOR OTHER ZONES THE ROUTINE USES FIXED HOURLY RATES BASED
C           ON ZONE TYPE #1
C*******************

      INCLUDE 'Common_params.fi'
	INCLUDE 'Common_data.fi'

      REAL*4 WALK1, WALK2
      REAL,INTENT(OUT),DIMENSION (ITER) :: WALK, COST
      REAL,INTENT(IN),DIMENSION (ITER) :: INCOME
      REAL,DIMENSION (ITER) :: WALK3
      INTEGER,INTENT(OUT),DIMENSION (ITER) :: INTOCC

C  SET ORIGIN, DESTINATION AND HOURS OF PARKING
      HOURS=10
      WALK1=WFA(ZTYPE(ORIG))
      WALK2=WFA(ZTYPE(DEST))
      COST=APC(ZTYPE(DEST))
C
C  HOME BASED OTHER TRIPS ARE SET TO SIX HOURS AND NON-HOME
C    BASED TRIPS ARE SET TO 3 HOURS.  tHIS WILL MAKE THE
C    PARKING COST FOR HOME BASED OTHER 60 PERCENT OF WORK
C    AND FOR NON-HOME BASED 30 PERCENT OF WORK
C
C    CHANGES MADE 12/8/93 BY GWS (NEXT TWO LINES)
C
C*******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************
C      IF (TPTYPE.EQ.2) HOURS=6
C      IF (TPTYPE.EQ.3) HOURS=3
C*******************  RWE CHANGE FOR I290 AUGUST-SEPT 2009  ************
C
	IF (TRACE) THEN
	  WRITE (31, '(/A)') ' PARKING COST INPUTS FROM SUBROUTINE PRKCST'
	  WRITE (31, '(A,I6)') '   ORIGIN ZONE=', ORIG
	  WRITE (31, '(A,I6)') '   DESTINATION ZONE=', DEST
  	  WRITE (31, '(A,L1)') '   DEST IS CBD PARKING ZONE? ',
     A    ZCBD_PARK(DEST)
      WRITE (31, '(A,F8.0)') '   HH INCOME (INCOME)=', INCOME(1)
	  WRITE (31, '(A,I6)') '   TRIP TYPE (TPTYPE)=',TPTYPE
	  WRITE (31, '(A,I6)') '   PARKING HOURS (HOURS)=',HOURS
      WRITE (31, '(A,F8.0)') '   WALK TIME TO PARK (WALK1)=', WALK1
      WRITE (31, '(A,F8.0)') '   WALK TIME FROM PARK (WALK2)=', WALK2
      WRITE (31, '(A,F8.0)') '   NON-CBD MODEL PARK COST (COST)=',
     A    COST(1)
      ENDIF
	IF (TRACE) THEN
	  WRITE (31, '(/A)') ' PARKING COST INPUTS FROM PRKCST 2ND ELEMENT'
        WRITE (31, '(A,I6)') '   ORIGIN ZONE=', ORIG
	  WRITE (31, '(A,I6)') '   DESTINATION ZONE=', DEST
  	  WRITE (31, '(A,L1)') '   DEST IS CBD PARKING ZONE? ',
     A    ZCBD_PARK(DEST)
      WRITE (31, '(A,F8.0)') '   HH INCOME (INCOME)=', INCOME(2)
	  WRITE (31, '(A,I6)') '   TRIP TYPE (TPTYPE)=',TPTYPE
	  WRITE (31, '(A,I6)') '   PARKING HOURS (HOURS)=',HOURS
      WRITE (31, '(A,F8.0)') '   WALK TIME TO PARK (WALK1)=', WALK1
      WRITE (31, '(A,F8.0)') '   WALK TIME FROM PARK (WALK2)=', WALK2
      WRITE (31, '(A,F8.0)') '   NON-CBD MODEL PARK COST (COST)=',
     A    COST(2)
      ENDIF      
      
      WALK3 = WALK2
C
C  DETERMINE IF THIS ZONE HAS A SPECIAL PARKING STRUCTURE
C
      IF (ZCBD_PARK(DEST)) 
     A  CALL PRKCBD (ORIG,DEST,INCOME,COST,WALK3,HOURS,INTOCC)
C 
      WALK=WALK1+WALK3
C DEBUGGING:           WRITE (31, '(/A)') '   RETURN FROM PRKCBD TO PRKCST OK'
C
      RETURN
      END