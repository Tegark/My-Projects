#pragma once

#include <iostream>

class MPI_Jacobi
{
public:
	MPI_Jacobi(int *argc, char ***argv, int size);
	~MPI_Jacobi();

	double * solve();
	void generateRandom();
	int getSize();

	friend std::istream& operator>>(std::istream &stream, MPI_Jacobi &res); // matrixA and matrixB
	friend std::ostream& operator<<(std::ostream &stream, MPI_Jacobi &res); // solution

private:
	int *argc;
	char ***argv;
	double *matrixA;
	double *matrixB;
	//double *pX;
	double eps;
	int size;
};

