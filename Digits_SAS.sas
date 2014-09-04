ods html body='ttest.htm' style=HTMLBlue;

LIBNAME DGT_LIB "C:\games\Kaggle\Digits\";
RUN;

PROC IMPORT OUT=DGT_LIB.DIGITS_SET
	DATAFILE="C:\games\Kaggle\Digits\Trunc_stats.csv"
	DBMS=csv
	REPLACE;
    getnames=yes;
RUN;

PROC MEANS DATA = DGT_LIB.DIGITS_SET;
	class numbers;
	*ID index;
	run;

	/*
PROC LOGISTIC DATA = DGT_LIB.DIGITS_SET
		OUTMODEL=DGT_LIB.LOGITRESULT;
	MODEL numbers = holes horizontal_cuts vertical_cuts pixel_count mass_x mass_y longest_contour contours_count / NOINT;
	TITLE 'Digits Logit Model';
RUN;
	*/

ods html close;
