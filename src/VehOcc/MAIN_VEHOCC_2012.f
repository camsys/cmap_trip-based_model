      PROGRAM VEHOCC_2012
      USE IFPORT
      USE IFCORE
C***********************************************************************
C
C     THIS BRIEF PROGRAM ESTIMATES VEHICLE OCCUPANCIES FOR WORK AND 
C     NOW-WORK TRIPS.  IT IS COMPATIBLE WITH ALL OPTIONS IN THE CMAP
C     MODE CHOICE MODEL FOR HOME WORK TRIPS. 
C
C       1.  OLD VERSION:  NO INCOME SEGMENTATION, NO AUTO OCCUPANCY
C           SUBMODE.
C       2.  HOV VERSION:  NO WORKER SEGMENTATION, AUTO OCCUPANCY SUBMODE
C       3.  HOV PLUS LOW/HIGH INCOME WORKERS:
C
C     THIS VERSION OF THE SOFTWARE WAS PREPARED FOR CMAP ON-CALL PROJECT
C     IN JUNE 2012 AT PARSONS BRINCKERHOFF CHICAGO OFFICES.
C
C     THE PROGRAM READS TRIP INTERCHANGE FILES AFTER A COMPLETE
C     (THREE TRIP TYPE) MODE CHOICE MODEL RUN.  IT RANDOMLY ASSIGNS A 
C     HOUSEHOLD TO EACH TRIP.  THE HOUSEHOLD TYPE DETERMINES THE 
C     PROBABILITY OF DIFFERENT AUTO OCCUPANCY LEVELS FOR HOME-WORK AND
C     HOME-OTHER TRIPS.  
C
C     WHEN THE MODE CHOICE MODEL IS APPLIED WITH AUTO SUBMODE 
C     ESTIMATION, THE MODE CHOICE AUTO OCCUPANCY IS SUBSTITUTED FOR THE 
C     HOUSEHOLD TYPE DETERMINED AUTO OCCUPANCY.  
C
C     TRIPS BY AUTO OCCUPANCY LEVELS ARE ACCUMULATED AT THE ATTRACTION
C     END OF HOME-WORK AND HOME-OTHER TRIPS.  THESE THEN DETERMINE THE 
C     PROBABILITIES FOR NON-HOME TRIPS.
C
C     TRIPS ARE SPLIT BY AUTO OCCUPANCY AND THE RESULTING AUTO TRIP 
C     TABLES WRITTEN IN THE EMMEBANK.
C
C***********************************************************************
C
C     WRITTEN BY R. EASH AT CHICAGO PB, JUNE 2012.
C
C     Edits by C. Heither, CMAP 11-06-2013 & 10-29-2014 as noted in script
C***********************************************************************
      IMPLICIT INTEGER (A-Z)
      
      INCLUDE 'COMMON_PARAMS.FI'

C ##### Heither, CMAP  11-06-2013, Emme 4-based changes      
      CHARACTER*1 ASTERIX(80)/80*'*'/
C ##	CHARACTER*9 DAY
C ##	CHARACTER*8 HOUR
	CHARACTER*8 DATE8
      CHARACTER*10  CTIME10
      CHARACTER*2 DATE2(4), CTIME2(5)
      EQUIVALENCE (DATE8,DATE2(1)), (CTIME10,CTIME2(1))      
      REAL*4 RAN_NUM  
      CHARACTER(LEN=30) :: FL1, FL2
C ### --- GET NAMELIST FILE AND LOG FILE ARGUMENTS      
      CALL GETARG(1, FL1)
      CALL GETARG(2, FL2)
      FL1 = TRIM(ADJUSTL(FL1))
      FL2 = TRIM(ADJUSTL(FL2))	
	  
C
C     OPEN OUTPUT LOG (UNIT=16)
C
      OPEN (UNIT=16,FILE=FL2,ERR=9916)

C ##      CALL DATE(DAY)
C ##	CALL TIME(HOUR)
      CALL DATE_AND_TIME(DATE8,CTIME10)	
      WRITE (16,'(15A)') 'LOGFILE OPENED:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)	  
C
C     LOGON
C     
      WRITE (*,'(80A1)') ASTERIX
C ##      CALL DATE(DAY)
C ##	CALL TIME(HOUR)
      WRITE (*,'(/A)')          'CMAP VEHICLE OCCUPACY PROGRAM'
	WRITE (*,'(A)') 'VERSION 1.0 (2012 CMAP ON-CALL PROJECT:  JUNE 201
     A2'
      WRITE (*,'(80A1)') ASTERIX
C
      CALL DATE_AND_TIME(DATE8,CTIME10)	
      WRITE (*,'(/15A)') 'PROGRAM STARTED:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)      	  
      WRITE (*,'(80A1)') ASTERIX
C
	WRITE (16,'(80A1)') ASTERIX
C ##      CALL DATE(DAY)
C ##	CALL TIME(HOUR)
      WRITE (16,'(/A)')          'CMAP VEHICLE OCCUPACY PROGRAM'
	WRITE (16,'(A)') 'VERSION 1.0 (2012 CMAP ON-CALL PROJECT:  JUNE 20
     A12'
      WRITE (16,'(80A1)') ASTERIX
C
      CALL DATE_AND_TIME(DATE8,CTIME10)	
      WRITE (16,'(/15A)') 'PROGRAM STARTED:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3) 	  
      WRITE (16,'(80A1)') ASTERIX
C
C	OPEN VEHOCC NAMELIST FILE (UNIT=33)
C
      OPEN (UNIT=33,FILE=FL1,ERR=9933)
C
      CALL DATE_AND_TIME(DATE8,CTIME10)
      WRITE (*,'(/15A)') 'VEHOCC_NAMELIST.TXT OPENED:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)	  
      WRITE (16,'(/15A)') 'VEHOCC_NAMELIST.TXT OPENED:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)
C*******************
C  DATA1 SUBROUTINE READS THE NAMELIST PARAMETERS INPUT BY THE USER
C    DEFAULTS THOSE PARAMETERS NOT SPECIFIED
C    OBTAINS THE RANDOM NUMBER SEED
C*******************
      CALL DATE_AND_TIME(DATE8,CTIME10)
      WRITE (*,'(/15A)') 'DATA1 CALLED:  ', 
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)	  
      WRITE (16,'(/15A)') 'DATA1 CALLED:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)	  
C
      CALL DATA1
C
      CALL DATE_AND_TIME(DATE8,CTIME10)
      WRITE (*,'(/15A)') 'RETURN FROM DATA1:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)	  
      WRITE (16,'(/15A)') 'RETURN FROM DATA1:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)	  
      WRITE (16,'(80A1)') ASTERIX
C
C     SET RANDOM NUMBER SEED
C
      IF (RNSEED .EQ. 0) THEN
        CALL SEED(RND$TIMESEED)
        CALL RANDOM(RAN_NUM)
C        WRITE (16,*) RNSEED, RND$TIMESEED, RAN_NUM
	  WRITE (*,'(/A,I10)') ' RANDOM NUMBER SEEDS(1):  ',RND$TIMESEED
	  WRITE (*,'(A,F12.8)') ' FIRST RANDOM NUMBER:  ', RAN_NUM
	  WRITE (16,'(/A,I10)')' RANDOM NUMBER SEEDS(1):  ',RND$TIMESEED
        WRITE (16,'(A,F12.8)')' FIRST RANDOM NUMBER:  ', RAN_NUM

	ELSE 
      
	  CALL SEED(RNSEED)
	  CALL RANDOM(RAN_NUM) 

        WRITE (*,'(/A,2I10)') ' RANDOM NUMBER SEEDS(2):  ', RNSEED
        WRITE (*,'(A,F12.8)') ' FIRST RANDOM NUMBER:  ', RAN_NUM
	  WRITE (16,'(/A,2I10)')' RANDOM NUMBER SEEDS(2):  ', RNSEED
        WRITE (16,'(A,F12.8)')' FIRST RANDOM NUMBER:  ', RAN_NUM

	ENDIF

      WRITE (16,'(80A1)') ASTERIX
C*******************
C  SUBROUTINE DATA2 READS THE FILES OF ZONE LEVEL ENUMERATED HOUSEHOLDS
C  AND HOUSEHDLD TYPE VEHICLE OCCUPANCIES
C*******************
      CALL DATE_AND_TIME(DATE8,CTIME10)
      WRITE (*,'(/15A)') 'DATA2 CALLED:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)   	  
      WRITE (16,'(/15A)') 'DATA2 CALLED:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)   	  
C
      CALL DATA2
C
      CALL DATE_AND_TIME(DATE8,CTIME10)
      WRITE (*,'(/A,A,A,A)') 'RETURN FROM DATA2:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)   	  
      WRITE (16,'(/15A)') 'RETURN FROM DATA2:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)   	  
C
C     OPEN UP THE PRINCIPAL EMMEBANK AND GET THE EMME PARAMETERS
C
      CALL OPEN_EMME4
C
C     OPEN HOME WORK OUTPUT MATRICES  
      OPEN (UNIT=913, FILE='emmemat/'//TEMX_OUT_HW_1,
     A ACCESS='DIRECT',RECL=1, STATUS='OLD')
      OPEN (UNIT=914, FILE='emmemat/'//TEMX_OUT_HW_2,
     A ACCESS='DIRECT',RECL=1, STATUS='OLD')
      OPEN (UNIT=915, FILE='emmemat/'//TEMX_OUT_HW_3,
     A ACCESS='DIRECT',RECL=1, STATUS='OLD')	 
	 
	 
C     READ HOME WORK FILE HOV = FALSE
C
      IF (.NOT. HOV) THEN
        CALL DATE_AND_TIME(DATE8,CTIME10)
      WRITE (*,'(/15A)') 'HWORK1 CALLED:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)  
      WRITE (16,'(/15A)') 'HWORK1 CALLED:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)  

      OPEN (UNIT=901, FILE='emmemat/'//TEMX_IN_HW,
     A ACCESS='DIRECT',RECL=1,STATUS='OLD')
C	 
        CALL HWORK1
C
      CLOSE(901)		
      ENDIF 
C
C     READ HOME WORK FILE HOV = TRUE
C
      IF (HOV) THEN
        CALL DATE_AND_TIME(DATE8,CTIME10)
        WRITE (*,'(/15A)') 'HWORK2 CALLED:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)
        WRITE (16,'(/15A)') 'HWORK2 CALLED:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)
	 
        IF (.NOT. WORKER$) THEN
         OPEN (UNIT=902, FILE='emmemat/'//TEMX_IN_HW_1, ACCESS='DIRECT',
     A RECL=1,STATUS='OLD')
	     OPEN (UNIT=903, FILE='emmemat/'//TEMX_IN_HW_2, ACCESS='DIRECT',
     A RECL=1,STATUS='OLD')
         OPEN (UNIT=904, FILE='emmemat/'//TEMX_IN_HW_3, ACCESS='DIRECT',
     A RECL=1,STATUS='OLD')	
        ENDIF	

        IF (WORKER$) THEN		
         OPEN (UNIT=905, FILE='emmemat/'//TEMX_IN_HW_LOW_1,
     A ACCESS='DIRECT',RECL=1,STATUS='OLD')
	     OPEN (UNIT=906, FILE='emmemat/'//TEMX_IN_HW_HIGH_1,
     A ACCESS='DIRECT',RECL=1,STATUS='OLD')
	     OPEN (UNIT=907, FILE='emmemat/'//TEMX_IN_HW_LOW_2,
     A ACCESS='DIRECT',RECL=1,STATUS='OLD')
	     OPEN (UNIT=908, FILE='emmemat/'//TEMX_IN_HW_HIGH_2,
     A ACCESS='DIRECT',RECL=1,STATUS='OLD')
	     OPEN (UNIT=909, FILE='emmemat/'//TEMX_IN_HW_LOW_3,
     A ACCESS='DIRECT',RECL=1,STATUS='OLD')
	     OPEN (UNIT=910, FILE='emmemat/'//TEMX_IN_HW_HIGH_3,
     A ACCESS='DIRECT',RECL=1,STATUS='OLD')	
           
        ENDIF		
C	 
        CALL HWORK2
C
        IF (TTYPE .EQ. 10) THEN
         CLOSE(902)
	   CLOSE(903)
         CLOSE(904)
        ENDIF	
C		
        IF (TTYPE .EQ. 100) THEN
         CLOSE(905)
	   CLOSE(906)
         CLOSE(907)
         CLOSE(908)
	   CLOSE(909)
         CLOSE(910)		 
        ENDIF	
		
      ENDIF

      CLOSE(913)	  
      CLOSE(914)
      CLOSE(915)	  
C
C     READ HOME OTHER FILE
C      
      CALL DATE_AND_TIME(DATE8,CTIME10)
        WRITE (*,'(/15A)') 'HOTHER CALLED:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)	  
        WRITE (16,'(/15A)') 'HOTHER CALLED:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)	  
 
C# ##  Heither, CMAP 11-08-2013: modified for Emme4 
      OPEN (UNIT=911, FILE='emmemat/'//TEMX_IN_HO,
     A ACCESS='DIRECT',RECL=1,STATUS='OLD')	     
      OPEN (UNIT=916, FILE='emmemat/'//TEMX_OUT_HO_1,
     A  ACCESS='DIRECT',RECL=1,STATUS='OLD')
      OPEN (UNIT=917, FILE='emmemat/'//TEMX_OUT_HO_2,
     A  ACCESS='DIRECT',RECL=1,STATUS='OLD')
      OPEN (UNIT=918, FILE='emmemat/'//TEMX_OUT_HO_3,
     A  ACCESS='DIRECT',RECL=1,STATUS='OLD')   
C	 
      CALL HOTHER
C	  
      CLOSE(911)	  
      CLOSE(916)
      CLOSE(917)
      CLOSE(918)
C
C     READ NON-HOME OTHER FILE
C          
      CALL DATE_AND_TIME(DATE8,CTIME10)
      WRITE (*,'(/15A)') 'NONHOME CALLED:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)	 	  
      WRITE (16,'(/15A)') 'NONHOME CALLED:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)	 	  

      OPEN (UNIT=912, FILE='emmemat/'//TEMX_IN_NH,
     A ACCESS='DIRECT',RECL=1,STATUS='OLD')	 
      OPEN (UNIT=919, FILE='emmemat/'//TEMX_OUT_NH_1,
     A  ACCESS='DIRECT',RECL=1,STATUS='OLD')
      OPEN (UNIT=920, FILE='emmemat/'//TEMX_OUT_NH_2,
     A  ACCESS='DIRECT',RECL=1,STATUS='OLD')
      OPEN (UNIT=921, FILE='emmemat/'//TEMX_OUT_NH_3,
     A  ACCESS='DIRECT',RECL=1,STATUS='OLD')
C	 
      CALL NONHOME
C      
      CLOSE(912)
      CLOSE(919)	  
      CLOSE(920)  
      CLOSE(921)        
C***********************************************************************
C
C     I/O ERRORS
C
C***********************************************************************     
      GO TO 9999

 9916 CONTINUE
      WRITE (*,*) 'ERROR:  UNABLE TO OPEN LOG OUTPUT FILE'
      STOP

 9932 CONTINUE
      WRITE (*,*) 'ERROR:  UNABLE TO OPEN EMMEBANK INPUT/OUTPUT FILE'
      STOP

 9933 CONTINUE
      WRITE (*,*) 'ERROR:  UNABLE TO OPEN NAMELIST INPUT FILE'
      STOP

 9999 CONTINUE
      WRITE (*,'(/A)') 'CMAP VEHICLE OCCUPANCY PROGRAM COMPLETED'
C
      CLOSE (16)
C# ##      CLOSE (32)
	CLOSE (33)

      STOP
      END