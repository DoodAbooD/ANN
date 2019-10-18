#pragma once
#include "neuron.h"

class synapse {
private:
	double w, dw;
	neuron *F;
	neuron *B;

public:
	synapse();
	void setNeurons(neuron*, neuron*);
	void setInitialWeight(double);
	void ff();
	void bp();
	void print(); //temp
};