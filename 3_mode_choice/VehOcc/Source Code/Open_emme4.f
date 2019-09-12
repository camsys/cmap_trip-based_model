C***********************************************************************
C        1         2         3         4         5         6         7 
C23456789012345678901234567890123456789012345678901234567890123456789012
C***********************************************************************
      SUBROUTINE OPEN_EMME4
      USE IFPORT
C***********************************************************************
C
C     THIS SUBROUTINE OPENS THE INPUT EMME DATABANK AND LOADS 
C     COMMON/EMMEBANK/ AND COMMON/BANKDIM/
C
C***********************************************************************
C
C     REVISED FROM EXISTING CODE BY EASH (AUGUST 1999)
C
C***********************************************************************
C
C     NOTE THAT EMME FILE NUMBERS ARE +1 FROM INRO MANUAL
C
C***********************************************************************
      IMPLICIT INTEGER*4 (A-Z) 
          
      INCLUDE 'COMMON_EMME4BANK.FI' 
      INCLUDE 'COMMON_PARAMS.FI'
      
	CHARACTER*1 ASTERIX(80)/80*'*'/
C ##### Heither, CMAP  10-24-2013
C #####  Coding changes for Emme4.	
	CHARACTER*8 DATE8
      CHARACTER*10  CTIME10
      CHARACTER*2 DATE2(4), CTIME2(5)
      EQUIVALENCE (DATE8,DATE2(1)), (CTIME10,CTIME2(1))	
C ##### ##### ##### ##### ##### #####

      INTEGER*4 cflag(999)
	  
      INTEGER*4 DB_SIZE	  	  
C
C     OPEN EMMEBANK FOR INPUT (UNIT=32)
C
      OPEN (UNIT=32, FILE='EMMEBANK',ACCESS='DIRECT',RECL=1,
     A  STATUS='OLD', ERR=992)
C
C #####  Coding changes for Emme4.	
      CALL DATE_AND_TIME(DATE8,CTIME10)
      WRITE (*,'(/A,A,A,A)') ' EMMEBANK OPENED:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3) 
      WRITE (16,'(/15A)') ' EMMEBANK OPENED:  ',
     A DATE2(3),'/',DATE2(4),'/',DATE2(1), DATE2(2),'  ', 
     B CTIME2(1),':', CTIME2(2),':', CTIME2(3)	
C ##### ##### ##### ##### ##### #####	 
C
C     INITIALIZE EMMEBANK COMMON AREAS
C
      DO NUM=1,200

        EMME_FILE_OFFSET(NUM) = -99
        EMME_FILE_TYPE(NUM) = -99
        EMME_FILE_NUM_RECORDS(NUM) = -99
        EMME_FILE_WORD_RECORD(NUM) = - 99   

      ENDDO
C
C     READ FIRST 512 WORDS IN EMMEBANK
C
      READ (32, REC=101) DB_SIZE
	  
      DO 28 WORD=1,100
C #####  Code block implements Eash's changes to read Emme4 full matrix files.	
      REC1 = WORD
      REC2 = (2*WORD-1)+100
      REC3 = WORD+300
      REC4 = WORD+400

      IF (WORD .GT. 1) THEN
        READ (32, REC=REC1) EMME_FILE_NUM_RECORDS(WORD-1)
        READ (32, REC=REC2) EMME_FILE_OFFSET(WORD-1)
        READ (32, REC=REC3) EMME_FILE_WORD_RECORD(WORD-1)
        READ (32, REC=REC4) EMME_FILE_TYPE(WORD-1)
      ENDIF
C ##### ##### ##### ##### ##### #####	
C
C     WRITE FOR DEBUGGING
C
C      DO 28 WORD=1,512
C	 REC1 = WORD
C      READ (32, REC=REC1) TEMP
C	 WRITE (16, '(2I10)') WORD, TEMP

C      WRITE (16,'(2I10)') WORD, EMME_FILE_OFFSET(WORD)
C      WRITE (16,*) WORD, EMME_FILE_TYPE(WORD)
C      WRITE (16,*) WORD, EMME_FILE_WORD_RECORD(WORD)
C      WRITE (16,*) WORD, EMME_FILE_NUM_RECORDS(WORD)

   28 CONTINUE
C
C     READ DATABANK DIMENSIONS FROM EMMEBANK
C
C #####  Code block implements Eash's changes to read Emme4 full matrix files.
      FILE = 1

      REC1 = EMME_FILE_OFFSET(FILE) + 51

      READ (32, REC=REC1) mscen
      REC1 = REC1 + 1
      READ (32, REC=REC1) mcent
      REC1 = REC1 + 1
      READ (32, REC=REC1) mnode
      REC1 = REC1 + 1
      READ (32, REC=REC1) mlink
      REC1 = REC1 + 1
      READ (32, REC=REC1) mturn
      REC1 = REC1 + 1
      READ (32, REC=REC1) mline
      REC1 = REC1 + 1
      READ (32, REC=REC1) mlseg
      REC1 = REC1 + 1
      READ (32, REC=REC1) mmat
      REC1 = REC1 + 1
      READ (32, REC=REC1) mfunc
      REC1 = REC1 + 1
      READ (32, REC=REC1) moper
C
C     READ PROJECT TITLE FROM EMMEBANK
C
C #####  Code block implements Eash's changes to read Emme4 full matrix files.
      FILE = 2
      DO 39 WORD = 1,40
      REC1 = EMME_FILE_OFFSET(FILE) + WORD
      READ (32, REC=REC1) iptit(WORD)
   39 CONTINUE  
C
C     WRITE OUT EMMEBANK DATA
C
      WRITE (16,'(/A)') ' EMMEBANK DATABANK'
	WRITE (16,'(80A1)') ASTERIX

      WRITE (16,1002) (iptit(WORD), WORD=1,40)
 1002 FORMAT (' DATABANK TITLE:  '40A2)
      WRITE(16,'(A,I12)')'   DATABANK SIZE IN WORDS= ', DB_SIZE  
      WRITE(16,'(A,I5)')'   MAX NUMBER OF SCENARIOS= ', mscen  
      WRITE(16,'(A,I5)')'   MAX NUMBER OF CENTROIDS= ', mcent
      WRITE(16,'(A,I5)')'   MAX NUMBER OF NODES= ', mnode
      WRITE(16,'(A,I5)')'   MAX NUMBER OF LINKS= ', mlink
      WRITE(16,'(A,I5)')'   MAX LENGTH OF TURN PENALTY TABLE= ', mturn
      WRITE(16,'(A,I5)')'   MAX NUMBER OF TRANSIT LINES= ', mline
      WRITE(16,'(A,I5)')'   MAX TOTAL LINE SEGMENTS= ', mlseg       
      WRITE(16,'(A,I5)')'   MAX NUMBER OF MATRICES= ', mmat
      WRITE(16,'(A,I5)')'   MAX NUMBER OF FUNCTIONS/CLASS= ', mfunc
      WRITE(16,'(A,I5)')'   MAX NUMBER OF OPERATORS/FUNCTION CLASS= ',
     A moper
C
      WRITE (16,'(/A)')  '   FILE  TYPE    OFFSET  WORD/REC   RECORDS'
C #####  Code block implements Eash's changes to read Emme4 full matrix files.
      DO I=1,99
        WRITE (16, '(I7, I6,3I10)') I, EMME_FILE_TYPE(I),
     A    EMME_FILE_OFFSET(I),
     B    EMME_FILE_WORD_RECORD(I),
     C    EMME_FILE_NUM_RECORDS(I)		
      ENDDO
C***********************************************************************
C
C     CHECK TO SEE WHETHER MATRICES ARE STORED CORRECTLY
C
C***********************************************************************
      DO I=1,mmat

        J = 3*mmat+I   

        REC1 = EMME_FILE_OFFSET(60) + J
        READ (32, REC=REC1 ) cflag(I)

        IF (cflag(I) .GT. 1) THEN
          WRITE (16,'(A)') ' ERROR:  UNSUITABLE (COLUMNWISE) MATRIX mf', 
     A      I
          PAUSE
        ENDIF 

      ENDDO
C***********************************************************************
C
C     CHECK SIZE OF EMME DATABANK
C
C***********************************************************************      
	IF (mcent .LT. ZONES) THEN
	  WRITE (16,'(A, 2I5)') ' ERROR:  EMMEBANK DIMENSIONED SMALLER TH
     AAN MAXIMUM ZONE NUMBER', mcent, ZONES
	  STOP
      ENDIF
     
      CLOSE (32)
C
      RETURN

  992 CONTINUE
      WRITE (16,'(A)') ' ERROR:  CANNOT OPEN UNIT 32 EMMEBANK'
      STOP 
C
      END