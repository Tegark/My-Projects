#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include "MPI_Jacobi.h"

using namespace std;

int main(int argc, char* argv[])
{
	MPI_Jacobi jac(&argc, &argv, 800);
	int procRank;
	//jac.generateRandom();
	//ofstream out("jac4.txt");
	//out << jac;
	//out.close();
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
	if (procRank == 0)
	{
		ifstream in("jac800.txt");
		in >> jac;

		in.close();
	}

	double *pX = jac.solve();


	/*if (procRank == 0) {
		int size = jac.getSize();
		cout << "size = " << size << endl;
		for (int i = 0; i < size; ++i) {
			cout << pX[i] << " ";
		}
	}*/
	
	MPI_Finalize();
	return 0;

}