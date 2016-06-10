all: mcquad burger burger2

burger2: burger2.o trid.o
	gcc -o burger2 burger2.o trid.o -lm -lgfortran -llapack -lblas -std=c99
burger2.o: burger2.c trid.h
	gcc -c burger2.c -lm -lgfortran -lblas -llapacke -std=c99
trid.o: trid.c trid.h
	gcc -c trid.c -std=c99
burger: burger.o
	gcc -o burger burger.o -lm -std=c99
mcquad: mcquad.o func.o
	gcc -o mcquad mcquad.o func.o -lm
mcquad.o: mcquad.c func.h
	gcc -c mcquad.c
func.o: func.c func.h
	gcc -c func.c
burger.o: burger.c
	gcc -c burger.c -lm -std=c99
