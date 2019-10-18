#pragma once
#include "neuron.h"
#include "synapse.h"

class nn {

private:
	int layerCount;
	int* topology;
	double RMSE, RMSRE;
	neuron** n;
	synapse*** s;
	neuron* bn; //bias neurons
	synapse** bs; //bias neuron synapses

public:
	nn();
	nn(int,int*);
	~nn();
	void print();
	void printRMSE();
	void printRMSRE();
	void initialize();
	void ff(double*);
	void bp(double*);
	void getOutput(double*);
	void reset();
};