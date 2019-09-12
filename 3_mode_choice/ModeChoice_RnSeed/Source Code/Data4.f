      SUBROUTINE DATA4(ORIG)
      IMPLICIT INTEGER (A-Z)
C*******************
C  THIS SUBROUTINE READS THE SECONDARY EMME DATABANK FOR ADDITIONAL 
C     AUTO PATHS WHEN THE TOLL OR HOV OPTIONS ARE IN EFFECT.  IT IS 
C     CALLED ONCE FOR EACH ZONE SELECTED FOR ANALYSIS.
C
C  THE EMME DATABANK DEFAULT TABLES ARE AS FOLLOWS:
C
C  1.  TOLL=T,HOV2=F, AND HOV3=F
C
C       MF01 = UNTOLLED TRAVEL TIME
C       MF02 = UNTOLLED TRAVEL DISTANCE
C       MF03 = TOLL PATH TRAVEL TIME
C       MF04 = TOLL PATH TRAVEL DISTANCE
C       MF05 = TOLLS VIA TOLL PATH
C
C  2.  TOLL=F,HOV2=T OR HOV3=T
C
C       MF01 = SOV TRAVEL TIME
C       MF02 = SOV TRAVEL DISTANCE
C       MF03 = HOV TRAVEL TIME
C       MF04 = HOV TRAVEL DISTANCE
C
C  3.  TOLL=T,HOV2=T OR HOV3=T
C
C       MF01 = UNTOLLED TRAVEL TIME (NO HOV LANE)
C       MF02 = UNTOLLED TRAVEL DISTANCE (NO HOV LANE)
C       MF03 = TOLL PATH TRAVEL TIME (NO HOV LANE)
C       MF04 = TOLL PATH TRAVEL DISTANCE (NO HOV LANE)
C       MF05 = TOLLS VIA TOLL PATH (NO HOV LANE)
C       MF06 = UNTOLLED SOV TRAVEL TIME
C       MF07 = UNTOLLED SOV TRAVEL DISTANCE
C       MF08 = UNTOLLED HOV TRAVEL TIME
C       MF09 = UNTOLLED HOV TRAVEL DISTANCE
C       MF10 = TOLL PATH SOV TRAVEL TIME
C       MF11 = TOLL PATH SOV TRAVEL DISTANCE
C       MF12 = TOLL PATH SOV TOLLS
C       MF13 = TOLL PATH HOV TRAVEL TIME
C       MF14 = TOLL PATH HOV TRAVEL DISTANCE
C       MF15 = TOLL PATH HOV TOLLS
C
C*******************
	INCLUDE 'Common_params.fi'
	INCLUDE 'Common_auto_params.fi'
	INCLUDE 'Common_emme4bank.fi'
	INCLUDE 'Common_auto_emme4bank.fi'
C***********************************************************************
C
C     REOPEN AUTO EMMEBANK
C
C*******************  RWE CHANGE FOR I290 OCTOBER 2009  ****************
C      OPEN (UNIT=42, FILE='EMMEBANK',
C     A  ACCESS='DIRECT',RECL=1,STATUS='OLD')
C*******************  RWE CHANGE FOR I290 OCTOBER 2009  ****************
C***********************************************************************
C
C     READ TABLES BY ROW
C
C***********************************************************************
	P = ORIG
C
C     ONLY TOLL OPTION IN EFFECT
C
      IF (TOLL .AND. ((.NOT. HOV2) .AND. (.NOT. HOV3))) THEN
        
        OPEN (UNIT=921, FILE='emmemat/'//TEMX_FREE_TIME,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
        OPEN (UNIT=922, FILE='emmemat/'//TEMX_FREE_DIST,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
        OPEN (UNIT=923, FILE='emmemat/'//TEMX_TOLL_TIME,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
        OPEN (UNIT=924, FILE='emmemat/'//TEMX_TOLL_DIST,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
        OPEN (UNIT=925, FILE='emmemat/'//TEMX_TOLL,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
C
C     P TO Q DIRECTION
C 
        DO Q=1,ZONES
          REC1 = ((P-1)*mcent) + Q
          READ(921, REC=REC1) ZLHT(Q)
          READ(922, REC=REC1) ZLHD(Q)
          READ(923, REC=REC1) ZLHT$(Q)
          READ(924, REC=REC1) ZLHD$(Q)
          READ(925, REC=REC1) ZLHC(Q)
        ENDDO
        
        CLOSE (921)
        CLOSE (922)
        CLOSE (923)
        CLOSE (924)
        CLOSE (925)
      ENDIF  
C
C     NO TOLLS AND HOV OPTIONS TURNED ON
C
      IF ((.NOT. TOLL) .AND. (HOV2 .OR. HOV3)) THEN 
          
        OPEN (UNIT=921, FILE='emmemat/'//TEMX_SOV_TIME,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
        OPEN (UNIT=922, FILE='emmemat/'//TEMX_SOV_DIST,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
        OPEN (UNIT=923, FILE='emmemat/'//TEMX_HOV_TIME,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
        OPEN (UNIT=924, FILE='emmemat/'//TEMX_HOV_DIST,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
C
C     P TO Q DIRECTION
C
        DO Q=1,ZONES
          REC1 = ((P-1)*mcent) + Q
          READ(921, REC=REC1) ZLHT1(Q)
          READ(922, REC=REC1) ZLHD1(Q)
          READ(923, REC=REC1) ZLHT2(Q)
          READ(924, REC=REC1) ZLHD2(Q)
        ENDDO
        
        CLOSE (921)
        CLOSE (922)
        CLOSE (923)
        CLOSE (924)
      ENDIF  
C
C     BOTH TOLL AND HOV OPTIONS TURNED ON
C
      IF (TOLL .AND. (HOV2 .OR. HOV3)) THEN
                      
        OPEN (UNIT=921, FILE='emmemat/'//TEMX_FREE_TIME,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
        OPEN (UNIT=922, FILE='emmemat/'//TEMX_FREE_DIST,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
        OPEN (UNIT=923, FILE='emmemat/'//TEMX_TOLL_TIME,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
        OPEN (UNIT=924, FILE='emmemat/'//TEMX_TOLL_DIST,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
        OPEN (UNIT=925, FILE='emmemat/'//TEMX_TOLL,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')

        OPEN (UNIT=926, FILE='emmemat/'//TEMX_SOV_FREE_TIME,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
        OPEN (UNIT=927, FILE='emmemat/'//TEMX_SOV_FREE_DIST,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
        OPEN (UNIT=928, FILE='emmemat/'//TEMX_HOV_FREE_TIME,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
        OPEN (UNIT=929, FILE='emmemat/'//TEMX_HOV_FREE_DIST,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')

        OPEN (UNIT=930, FILE='emmemat/'//TEMX_SOV_TOLL_TIME,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
        OPEN (UNIT=931, FILE='emmemat/'//TEMX_SOV_TOLL_DIST,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
        OPEN (UNIT=932, FILE='emmemat/'//TEMX_SOV_TOLL,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')

        OPEN (UNIT=933, FILE='emmemat/'//TEMX_HOV_TOLL_TIME,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
        OPEN (UNIT=934, FILE='emmemat/'//TEMX_HOV_TOLL_DIST,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
        OPEN (UNIT=935, FILE='emmemat/'//TEMX_HOV_TOLL,
     A    ACCESS='DIRECT',RECL=1,STATUS='OLD')
C
C     P TO Q DIRECTION
C 
        DO Q=1,ZONES
          READ(921, REC=REC1) ZLHT(Q)
          READ(922, REC=REC1) ZLHD(Q)
          READ(923, REC=REC1) ZLHT$(Q)
          READ(924, REC=REC1) ZLHD$(Q)
          READ(925, REC=REC1) ZLHC(Q)
          READ(926, REC=REC1) ZLHT1(Q)
          READ(927, REC=REC1) ZLHD1(Q)
          READ(928, REC=REC1) ZLHT2(Q)
          READ(929, REC=REC1) ZLHD2(Q)
          READ(930, REC=REC1) ZLHT1$(Q)
          READ(931, REC=REC1) ZLHD1$(Q)
          READ(932, REC=REC1) ZLHC1(Q)
          READ(933, REC=REC1) ZLHT2$(Q)
          READ(934, REC=REC1) ZLHD2$(Q)
          READ(935, REC=REC1) ZLHC2(Q)
        ENDDO
        
        CLOSE (921)
        CLOSE (922)
        CLOSE (923)
        CLOSE (924)
        CLOSE (925)
        CLOSE (926)
        CLOSE (927)
        CLOSE (928)
        CLOSE (929)
        CLOSE (930)
        CLOSE (931)
        CLOSE (932)
        CLOSE (933)
        CLOSE (934)
        CLOSE (935)
      ENDIF
C
C     FOLLOWING CODE TO TRACE ERRORS
C
      IF (TRACE) THEN

        WRITE(31,9001) ORIG

        DO Z=1,ZONES
          WRITE(31,9007)  Z, ZLHT(Z), ZLHT1(Z), ZLHT2(Z), ZLHT$(Z),
     A      ZLHT1$(Z), ZLHT2$(Z), ZLHD(Z), ZLHD1(Z), ZLHD2(Z), ZLHD$(Z),
     A      ZLHD1$(Z), ZLHD2$(Z), ZLHC(Z), ZLHC1(Z), ZLHC2(Z)
        ENDDO
      ENDIF

 9001 FORMAT(/' FOR ORIGIN ZONE',I5,' THE AUTO EMME MATRIX DATA IS',/,
     A'         FREE TRAVEL TIME     TOLL TRAVEL TIME      FREE PATH DIS
     BT.      TOLL PATH DIST.     TOLL PAID (CENTS)',/,
     C' ZONE   BASE    SOV    HOV   BASE    SOV    HOV   BASE    SOV    
     DHOV   BASE    SOV    HOV   BASE    SOV    HOV ')

 9007 FORMAT(' ',I4, 5(3F7.2))

      RETURN
      END