#include "MPI_Jacobi.h"
#include <ctime>
#include <mpi.h>
#include <iostream>


MPI_Jacobi::MPI_Jacobi(int *argc, char ***argv, int size)
	: argc(argc), argv(argv), matrixA(NULL)
	, matrixB(NULL), eps(0.0000000000001)
	, size(size)
{
}

MPI_Jacobi::~MPI_Jacobi()
{
	delete[] matrixA;
	delete[] matrixB;
}

void MPI_Jacobi::generateRandom() {
	matrixA = new double[size * size];
	matrixB = new double[size];
	srand(time(0));
	for (int i = 0; i < size; i++) {
		double sum = 0;
		for (int j = 0; j < size; j++) {
			if (i != j) {
				matrixA[i * size + j] = rand() % 18 - 9;
				sum += abs(matrixA[i * size + j]);
			}
		}
		int sign;
		if (rand() > 16000)
			sign = 1;
		else
			sign = -1;
		matrixA[i * size + i] = sign * sum * (rand() / 32767.0 + 1);
		matrixB[i] = rand() % 18 - 9;
	}
}

std::istream& operator>>(std::istream &stream, MPI_Jacobi &res) {
	delete[] res.matrixA;
	delete[] res.matrixB;
	int tempSize;
	stream >> res.size >> tempSize;
	res.matrixA = new double[res.size * res.size];
	res.matrixB = new double[res.size];
	for (int i = 0; i < res.size; ++i) {
		for (int j = 0; j < res.size; ++j) {
			stream >> res.matrixA[i * res.size + j];
		}
		stream >> res.matrixB[i];
	}
	return stream;
}

std::ostream& operator<<(std::ostream &stream, MPI_Jacobi &res) {
	stream << res.size << " " << (res.size + 1) << std::endl;
	for (int i = 0; i < res.size; ++i) {
		for (int j = 0; j < res.size; ++j) {
			stream << res.matrixA[i * res.size + j] << " ";
		}
		stream << res.matrixB[i] << std::endl;
	}
	return stream;
}

double * MPI_Jacobi::solve() {
	double* matA;
	double* matB;
	double* matX;
	double* pProcRowsA;
	double* pProcRowsB;
	double* pProcTempX;
	int matSize, rowNum, procNum, procRank;
	double eps, norm, maxNorm;
	double start, finish;

	MPI_Comm_size(MPI_COMM_WORLD, &procNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
	
	if (procRank == 0) 
	{
		matSize = this->size;
	}

	/*std::cout << "Proc " << procRank << " waiting for input" << std::endl;*/
	MPI_Bcast(&matSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	//std::cout << "Proc " << procRank << " Size broadcated" << std::endl;
	rowNum = matSize / procNum;

	//std::cout << "RowNum = " << rowNum << std::endl;
	eps = 0.00000000000001;

	matX = new double[matSize];
	pProcRowsA = new double[rowNum * matSize];
	pProcRowsB = new double[rowNum];
	pProcTempX = new double[rowNum];
	
	if (procRank == 0) 
	{
	
		matA = matrixA;
		matB = matrixB;

		for (int i = 0; i < matSize; i++) {
			matX[i] = 0;
		}
	}

	MPI_Bcast(matX, matSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	//std::cout << "Proc " << procRank << " pX broadcasted" << std::endl;
	
	MPI_Scatter(matA, rowNum * matSize, MPI_DOUBLE, pProcRowsA, rowNum * matSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(matB, rowNum, MPI_DOUBLE, pProcRowsB, rowNum, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	start = MPI_Wtime();

	do 
	{
		for (int i = 0; i < rowNum; i++) 
		{
			pProcTempX[i] = -pProcRowsB[i];
			for (int j = 0; j < matSize; j++) 
			{
				if (procRank * rowNum + i != j)
					pProcTempX[i] += pProcRowsA[i * matSize + j] * matX[j];
			}
			pProcTempX[i] /= -pProcRowsA[procRank * rowNum + i + i * matSize];
		}


		norm = fabs( matX[procRank * rowNum] - pProcTempX[0] );
		for (int i = 0; i < rowNum; i++) 
		{
			if ( fabs( matX[procRank * rowNum + i] - pProcTempX[i]) > norm )
				norm = fabs( matX[procRank * rowNum + i] - pProcTempX[i] );
		}
		MPI_Reduce(&norm, &maxNorm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Bcast(&maxNorm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Allgather(pProcTempX, rowNum, MPI_DOUBLE, matX, rowNum, MPI_DOUBLE, MPI_COMM_WORLD);
	} while (maxNorm > eps);
	
	finish = MPI_Wtime();
	
	if (procRank == 0) 
	{
		std::cout << std::endl << "Execution time =" << (finish - start) << " sec." << std::endl;
	}

	if (procRank != 0)
	{
		delete[] matX;
	}
	
	delete[] pProcRowsA;
	delete[] pProcRowsB;
	delete[] pProcTempX;
	
	return matX;

}

int MPI_Jacobi::getSize() {
	return size;
}
