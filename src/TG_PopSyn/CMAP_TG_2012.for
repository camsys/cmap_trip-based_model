      PROGRAM CMAP_TG_2012
      IMPLICIT INTEGER(A-Z)
C***********************************************************************
C        1         2         3         4         5         6         7
C23456789012345678901234567890123456789012345678901234567890123456789012
C***********************************************************************
C
C     THIS VERSION OF THE CMAP TRIP GENERATION MODEL WAS RECODED IN
C     MARCH-JUNE 2012.
C
C     CODED BY RONALD EASH AT PB CHICAGO OFFICES
C     
C     PROJECT 16886A, TASK NUMBER 01:  FULLY ENUMERATED TRIP GENERATION
C
C-- Heither, 03-13-2018: changes to clean up PopSyn version of the code:
C--    IPF routines replaced with synthetic household file.
C***********************************************************************
C
C     SUMMARY OF NEW FEATURES:
C
C     1.  INPUT FILES HAVE BEEN UPDATED WITH 2006-2010 ACS PUMS AND
C         SUMMARY FILES
C
C     2.  THE ITERATIVE PROPORTIONAL FITTING ROUTINE NOW INCLUDES:
C         A.  ONE TO FOUR PLUS PERSON HOUSEHOLD SIZE CATEGORIES
C         B.  A FIFTH DIMENSION FOR AGE OF THE HEAD OF HOUSEHOLD
C
C     3.  THE PROGRAM NOW HAS TWO IPF ROUTINES, ONE TO ESTIMATE THE 
C         NUMBER OF REGIONAL HOUSEHOLDS IN 624 TYPE CATEGORIES 
C         (4 ADULTS 4 WORKERS, 4 CHILDREN, 4 INCOME, 3 AGE OF 
C         HOUSEHOLDER) FROM REGIONAL CONTROL TOTALS AND A REGIONAL
C         SEED TABLE BUILT FROM ACS PUMS FILES (2006-2010) IS FIRST 
C         ONE USED.  THE SECOND IPF ADJUSTS THE INITIAL SUBZONE LEVEL
C         HOUSEHOLD TYPE ESTIMATES TO AGREE WITH THE REGIONAL TABLE.
C
C     4.  THE PROGRAM HAS A NEW OPTION TO ALLOW ENUMBERATION OF 
C         HOUSEHOLDS FROM EITHER THE 2006-2010 ACS PUMS OR THE CMAP
C         TRAVEL TRACKER SURVEY.
C
C     4.  GROUP QUARTERS INPUTS NOW DISTINGUISH BETWEEN GQ PERSONS IN
C         MILITARY QUARTERS, COLLEGE/UNIVERSITY DORMS AND OTHER 
C         CATEGORIES.
C
C     RON EASH
C     JUNE 2012
C
C***********************************************************************
C
C     FOLLOWING ARE HISTORICAL NOTES FROM PREVIOUS VERSIONS
C
C***********************************************************************
C
C     THIS VERSION OF THE CMAP TRIP GENERATION IS BASED ON THE VERSION 
C     DEVELOPED FOR THE SECOND ROUND OF I-290 HOV LANE FORECASTS IN 
C     FALL 2006.  THE FOLLOWING ENHANCEMENTS ARE INCORPORATED INTO THIS
C     CODE:
C
C       1. THE MATCH FILE BETWEEN TG SUBZONES AND OTHER GEOGRAPHIC UNITS
C          IS NOW INPUT INTO THE PROGRAM THORUGH THE FILE GEOFILE.TXT.
C       
C       2. TRIP GENERATION RATES HAVE BEEN REVISED BASED ON THE CMAP 
C          HOUSEHOLD TRAVEL SURVEY COMPLETED IN 2008.
C
C       3. AVERAGE HOUSEHOLD CHARACTERISTICS CAN NOW BE INPUT BY FIVE
C          PERCENT PUMAS AS WELL AS BY TRIP GENERATION SUBZONES.
C
C     THIS VERSION OF THE CODE IS DATED OCTOBER 1, 2008 AND WAS WRITTEN
C     BY RONALD EASH UNDER CONTRACT TO PARSONS BRINCKERHOFF.   
C
C***********************************************************************
C
C     NOTE FOR PREVIOUS VERSION:
C
C     THIS IS A SUBSTANTIAL OVERHAUL OF THE CATS TRIP GENERATION MODEL
C     IN USE AS OF 2006.  THE CODE HAS BEEN RVISED TO ACCEPT BASE DATA
C     FROM THE 2000 CENSUS AND LATER (ACS) PUMS.  
C
C     FOUR WAY MATRIX BALANCING IS NOW USED TO DEVELOP THE DISTRIBUTION 
C     OF HOUSEHOLDS IN A TG ZONE BY ADULTS, WORKERS, CHILDREN AND 
C     INCOME.  TRIP GENERATION RATES ARE BASED ON 2001 NHTS RATES.  THE
C     ROUTINE TO ADJUST THE INCOME INDEX TO BETTER ALLOCATE HOUSEHOLDS
C     IS ALSO INCLUDED.
C
C     OTHER CHANGES ARE DOCUMENTED IN THE CODE.
C
C     RON EASH
C     NOVEMBER 14, 2006
C
C***********************************************************************
C
C     CORRECTED COUNTY LEVEL;ATTRACTION OUTPUT TABLES FOR ORIGINAL 
C     TRIP CATEGORIES
C
C     RWE 8/24/2009
C
C***********************************************************************
C
C     ADJUSTED NONMOTORIZED MODEL BIAS CONSTANTS FOR 2010 CENSUS BLOCK 
C     FILE CODING.
C
C     RWE 3/6/2011
C
C***********************************************************************
	INCLUDE 'COMMON_PARAM.FI'
C
C	OUTPUT LOGFILE (PRINT OUTPUT) IS UNIT 16
C
      OPEN (UNIT=16,FILE='TG_OUTPUT.TXT',STATUS='NEW',ERR=199)

      WRITE (*,'(/A)') '********************************************'
	WRITE (*,'(A)')  '* CHICAGO METROPOLITAN AGENCY FOR PLANNING *'
	WRITE (*,'(A)')  '*    HOUSEHOLD TRIP GENERATION SOFTWARE    *'
	WRITE (*,'(A)')  '*                                          *'
	WRITE (*,'(A)')  '*   REVISED BY RONALD EASH IN FALL 2008    *' 
	WRITE (*,'(A)')  '*  CMAP 2007-2008 HOUSEHOLD TRAVEL SURVEY  *'
	WRITE (*,'(A)')  '*          TRIP GENERATION RATES           *'
	WRITE (*,'(A)')  '*                                          *'
	WRITE (*,'(A)')  '*   REVISED BY RONALD EASH IN MARCH 2011   *' 
	WRITE (*,'(A)')  '*       CENSUS 2010 BLOCK GEOGRAPHY        *'
     	WRITE (*,'(A)')  '*                                          *'
    	WRITE (*,'(A)')  '*  REVISED BY RONALD EASH IN SPRING 2012   *' 
	WRITE (*,'(A)')  '*          **ACS 2006-2010 DATA**          *'
      WRITE (*,'(A)')  '*           **NEW IPF PROCESS**            *'
      WRITE (*,'(A)')  '*          **ADDED HH CATEGORY**           *'
      WRITE (*,'(A)')  '*       **AGE OF HEAD OF HOUSEHOLD**       *'
      WRITE (*,'(A)')  '*                                          *'
      WRITE (*,'(A)')  '*  - - - - - - - - - - - - - - - - - - - - *' 
      WRITE (*,'(A)')  '*                                          *'
      WRITE (*,'(A)')  '*  LAST REVISED BY CMAP STAFF APRIL 2018:  *' 
      WRITE (*,'(A)')  '*   - USE SYNTHETIC POPULATION             *'
      WRITE (*,'(A)')  '*   - REDUCED PROBABILITY OF SELECTING     *'
      WRITE (*,'(A)')  '*     RARE SURVEY HOUSEHOLD                *'
	WRITE (*,'(A)')  '********************************************'
C	
      WRITE (16,'(/A)') '********************************************'
	WRITE (16,'(A)')  '* CHICAGO METROPOLITAN AGENCY FOR PLANNING *'
	WRITE (16,'(A)')  '*    HOUSEHOLD TRIP GENERATION SOFTWARE    *'
	WRITE (16,'(A)')  '*                                          *'
	WRITE (16,'(A)')  '*   REVISED BY RONALD EASH IN FALL 2008    *' 
	WRITE (16,'(A)')  '*  CMAP 2007-2008 HOUSEHOLD TRAVEL SURVEY  *'
	WRITE (16,'(A)')  '*          TRIP GENERATION RATES           *'
      WRITE (16,'(A)')  '*                                          *'
	WRITE (16,'(A)')  '*   REVISED BY RONALD EASH IN MARCH 2011   *' 
	WRITE (16,'(A)')  '*       CENSUS 2010 BLOCK GEOGRAPHY        *'
     	WRITE (16,'(A)')  '*                                          *'
    	WRITE (16,'(A)')  '*  REVISED BY RONALD EASH IN SPRING 2012   *' 
	WRITE (16,'(A)')  '*          **ACS 2006-2010 DATA**          *'
      WRITE (16,'(A)')  '*           **NEW IPF PROCESS**            *'
      WRITE (16,'(A)')  '*          **ADDED HH CATEGORY**           *'
      WRITE (16,'(A)')  '*       **AGE OF HEAD OF HOUSEHOLD**       *'  
      WRITE (16,'(A)')  '*                                          *'
      WRITE (16,'(A)')  '*  - - - - - - - - - - - - - - - - - - - - *' 
      WRITE (16,'(A)')  '*                                          *'
      WRITE (16,'(A)')  '*  LAST REVISED BY CMAP STAFF APRIL 2018:  *' 
      WRITE (16,'(A)')  '*   - USE SYNTHETIC POPULATION             *'
      WRITE (16,'(A)')  '*   - REDUCED PROBABILITY OF SELECTING     *'
      WRITE (16,'(A)')  '*     RARE SURVEY HOUSEHOLD                *' 
      WRITE (16,'(A)')  '********************************************'
C	  
	WRITE (*,'(//A)') 'CONTROL SUBROUTINE FOR PROGRAM OPTIONS'
	WRITE (16,'(//A)')'CONTROL SUBROUTINE FOR PROGRAM OPTIONS'

      CALL SUB_CONTROL
      CALL SEED(RNSEED)
C	  
	WRITE (*,'(//A)') 'GEOG SUBROUTINE FOR REGIONAL GEOGRAPHY'
	WRITE (16,'(//A)')'GEOG SUBROUTINE FOR REGIONAL GEOGRAPHY'

      CALL SUB_GEOG
C	
C--	WRITE (*,'(//A)') 'PUMS SUBROUTINE TO LOAD PUMS REGIONAL DATA'
C--	WRITE (16,'(//A)')'PUMS SUBROUTINE TO LOAD PUMS REGIONAL DATA'

      CALL SUB_PUMS
C
C-- Heither, 03-13-2018: do not call SUB_REG_HHTYPE or SUB_ROWCOL
C
      GO TO 3
	  
      WRITE (*, '(//A)') 'REG_HHTYPE SUBROUTINE TO ESTIMATE FIVE-WAY REG
     AIONAL HOUSEHOLD DISTRIBUTION'
	WRITE (*,'(A)')    '(ADULTS BY WORKERS BY CHILDREN BY INCOME QUART
     AILE BY AGE OF HOUSEHOLDER)'
      WRITE(16, '(//A)') 'REG_HHTYPE SUBROUTINE TO ESTIMATE FIVE-WAY REG
     AIONAL HOUSEHOLD DISTRIBUTION'
	WRITE (16,'(A)')   '(ADULTS BY WORKERS BY CHILDREN BY INCOME QUART
     AILE BY AGE OF HOUSEHOLDER)'
	
c--      CALL SUB_REG_HHTYPE
  
C	
	WRITE (*,'(//A)') 'ROWCOL SUBROUTINE TO LOAD HOUSEHOLD TYPE DISTRI
     ABUTIONS'
	WRITE (16,'(//A)')'ROWCOL SUBROUTINE TO LOAD HOUSEHOLD TYPE DISTRI
     ABUTIONS'

      CALL SUB_ROWCOL 
C
    3 CONTINUE 
     
      WRITE (*, '(//A)') 'TAZ_HHTYPE SUBROUTINE TO CREATE SUBZONE HOUSEH
     AOLD TYPE TABLES'
      WRITE(16, '(//A)') 'TAZ_HHTYPE SUBROUTINE TO CREATE SUBZONE HOUSEH
     AOLD TYPE TABLES'
      
      CALL SUB_TAZ_HHTYPE
C
	
      WRITE (*, '(//A)') 'PRINT_TAZHH SUBROUTINE TO TABULATE SUBZONE HOU
     ASEHOLD TYPE TABLES'
      WRITE(16, '(//A)') 'PRINT_TAZHH SUBROUTINE TO TABULATE SUBZONE HOU
     ASEHOLD TYPE TABLES'
      
      CALL SUB_PRINT_TAZHH   
      
      IF (HHENUM) THEN 
        WRITE (*, '(//A)')  'PUMS_HHENUM SUBROUTINE TO SET UP INTERNAL A
     ARRAYS FROM ACS PUMS HOUSEHOLDS'
        WRITE (*, '(A)')    'OUTPUTS FILE OF PUMS HHS BY TG SUBZONE'  
        WRITE (16, '(//A)') 'PUMS_HHENUM SUBROUTINE TO SET UP INTERNAL A
     ARRAYS FROM ACS PUMS HOUSEHOLDS'
        WRITE (16, '(A)')   'OUTPUTS FILE OF PUMS HHS BY TG SUBZONE'  
          
        CALL SUB_PUMS_HHENUM
        
        WRITE (*, '(//A)')  'END OF PUMS HOUSEHOLD ENUMERATION'
	  WRITE (16, '(//A)')  'END OF PUMS HOUSEHOLD ENUMERATION'
        STOP
      ENDIF  
C
      IF (TRIPGEN) THEN
		CALL SUB_HHVEH
       
        WRITE (*, '(//A)')  'HI_HHENUM SUBROUTINE TO SET UP INTERNAL ARR
     AAYS FROM TRAVEL SURVEY PUMS HOUSEHOLDS'
        WRITE (*, '(A)')    'OUTPUTS FILE OF SURVEY HHS WITH TRIP PRODUC
     ATIONS/ATTRACTIONS BY TG SUBZONE'  
        WRITE (16, '(//A)') 'HI_HHENUM SUBROUTINE TO SET UP INTERNAL ARR
     AAYS FROM TRAVEL SURVEY PUMS HOUSEHOLDS'
        WRITE (16, '(A)')   'OUTPUTS FILE OF SURVEY HHS WITH TRIP PRODUC
     ATIONS/ATTRACTIONS BY TG SUBZONE' 
      
        CALL SUB_HI_HHENUM
      
        WRITE (*, '(//A)')  'TRIPGEN4 SUBROUTINE TO PRINT'
	  WRITE (*,'(A)')   'HOUSEHOLD BASED TRIP PRODUCTIONS'
	 	
        WRITE (16, '(//A)')  'TRIPGEN4 SUBROUTINE TO PRINT'
	  WRITE (16,'(A)')     'HOUSEHOLD BASED TRIP PRODUCTIONS'

        CALL SUB_TRIPGEN4
C
        WRITE (*, '(//A)')  'TRIPGEN5 SUBROUTINE TO ESTIMATE'
	  WRITE (*,'(A)')     'GROUP QUARTERS TRIP PRODUCTIONS'
	 	
        WRITE (16, '(//A)')  'TRIPGEN5 SUBROUTINE TO ESTIMATE'
	  WRITE (16,'(A)')     'GROUP QUARTERS TRIP PRODUCTIONS'

        CALL SUB_TRIPGEN5
C
	  WRITE (*, '(//A)')  'TRIPGEN6 SUBROUTINE TO ALLOCATE'
	  WRITE (*,'(A)')     'NONHOME TRIP PRODUCTIONS AND ATTRACTIONS'
	 	
        WRITE (16, '(//A)')  'TRIPGEN6 SUBROUTINE TO ALLOCATE'
	  WRITE (16,'(A)')     'NONHOME TRIP PRODUCTIONS AND ATTRACTIONS'

        CALL SUB_TRIPGEN6
C
	  WRITE (*, '(//A)')'TRIPGEN7 SUBROUTINE TO ESTIMATE TRIPS TO AND' 
	  WRITE (*,'(A)')    'FROM EXTERNAL LOCATIONS'
	 	
        WRITE (16,'(//A)')'TRIPGEN7 SUBROUTINE TO ESTIMATE TRIPS TO AND'
	  WRITE (16,'(A)')    'FROM EXTERNAL LOCATIONS'
      
	  CALL SUB_TRIPGEN7
C
	  WRITE (*, '(//A)')'TRIPGEN8 TO ESTIMATE NONMOTORIZED TRIPS'
	 	
        WRITE (16,'(//A)')'TRIPGEN8 TO ESTIMATE NONMOTORIZED TRIPS'

        CALL SUB_TRIPGEN8
C
	  WRITE (*, '(//A)')'TRIPGEN9 TO WRITE FINAL REPORTS AND FILES'
        WRITE (16,'(//A)')'TRIPGEN9 TO WRITE FINAL REPORTS AND FILES'

        CALL SUB_TRIPGEN9
      ENDIF  

      GO TO 999

  199 WRITE (*,'(/A)') 'ERROR;  UNABLE TO OPEN TEXT OUTPUT FILE'
      STOP 199

  999 CONTINUE
	WRITE (*, '(//A)')  'END OF TRIP GENERATION'
	WRITE (16, '(//A)')  'END OF TRIP GENERATION'
       

	CLOSE (16,DISP='KEEP')
C
      STOP
	END