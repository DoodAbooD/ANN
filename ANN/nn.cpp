//Code by: Abdelqader Okasha
//for direct contact: abd.okasha@gmail.com

#include "neuron.h"
#include "synapse.h"
#include "nn.h"
#include "activations.h"
#include <math.h>
#include <iostream>

#define ETA 0.2 //Rate of learning 0-1
#define ALPHA 0.2 //Momentum of learning 0-1
#define INITIAL_WEIGHT (rand()%10)/10.0
#define INITIAL_BIAS 0.5

int neuron::count = 0; //temp


// **********************************************************
//					NEURON CLASS
// **********************************************************
void neuron::fire() { //TODO: add other activation functions
	//TODO : move this whole switching to the activation class, neuron just calls method with its type and activation class does its job
	switch (type) {
	case 's': //sigmoid neuron
		a = activations::sigmoid(SI);
		DSI = activations::d_sigmoid(SI);
		break;
	case 'r': //ReLU neuron
		a = activations::ReLU(SI);
		DSI = activations::d_ReLU(SI);
	case 'i': //input neuron
		a = SI;
		break;
	case 'd': //linear neuron
		a = SI;
		DSI = 1;
		break;
	case 'b': //bias neuron
		a = SI;
		break;

	}
	
}
neuron::neuron() {
	type = 's';
	SI = 0;
	DSI = 0;
	a = 0;
	SDW = 0;
	id = count++;
}
void neuron::receive(double input) {
	SI += input;
}
void neuron::receiveSDW(double input) {
	SDW += input;
}

void neuron::reset() {
	SI = 0;
	DSI = 0;
	SDW = 0;
	if (type != 'b')
		a = 0;
}
double neuron::getDSI() {
	return DSI;
}
double neuron::geta() {
	return a;
}
double neuron::getSDW() {
	return SDW;
}
void neuron::normalizeSDW(int n) {
	SDW = SDW / n;
}
void neuron::changeType(char c) {
	type = c;
}
int neuron::getid() { //temp
	return id;
}
void  neuron::print() { //temp
	std::cout << ((type == 'b') ? "Bias " : "") << "Neuron ID: " << id << " Type: " << type << " SI = " << SI << " a = " << a << std::endl;
}

char neuron::getType() {
	return type;
}

// **********************************************************
// **********************************************************



// **********************************************************
//					SYNAPSE CLASS
// **********************************************************

synapse::synapse() {
	dw = 0;
	w = 0;
	F = NULL;
	B = NULL;
}

void synapse::ff() {
	F->receive(B->geta() * w);
}

void synapse::bp() {
	double oldDw = dw;
	dw = F->getDSI() * B->geta() * F->getSDW() / F->geta();
	B->receiveSDW(dw * w);
	w -= (dw * ETA) + (oldDw * ALPHA);
}

void synapse::setNeurons(neuron* B, neuron* F) {
	this->B = B;
	this->F = F;
}

void synapse::setInitialWeight(double w) {
	this->w = w;
}

void synapse::print() { //temp
	std::cout << ((B->getType() == 'b') ? "Bias " : "") << "Synapse connecting " << B->getid() << " to " << F->getid() << " Weight= " << w << std::endl;
}
// **********************************************************
// **********************************************************


// **********************************************************
//					ACTIVATION FUNCTIONS
// **********************************************************

double activations::sigmoid(double in) {
	return 1 / (1 + exp(-1 * in));
}

double activations::d_sigmoid(double in) {
	return sigmoid(in) * (1 - sigmoid(in));
}

double activations::identity(double in) {
	return in;
}

double activations::d_identity(double in) {
	return 1;
}

double activations::ReLU(double in) {
	return (in >= 0) ? in : 0;
}

double activations::d_ReLU(double in) {
	return (in >= 0) ? 1 : 0;
}

// **********************************************************
// **********************************************************



// **********************************************************
//					NEURAL NETWORK
// **********************************************************
nn::nn() {
	layerCount = 0;
	RMSE = 0;
	RMSRE = 0;
	topology = NULL;
	n = NULL;
	s = NULL;
	bn = NULL;
	bs = NULL;
}

nn::nn(int count, int* top) {
	RMSE = 0;
	RMSRE = 0;
	layerCount = count;
	topology = new int[layerCount];
	n = new neuron * [layerCount];
	s = new synapse * *[layerCount - 1];

	//Filling topology, and creating all neurons and synapses
	for (int i = 0; i < layerCount; i++) {
		topology[i] = top[i];
		n[i] = new neuron[top[i]];

		if (i == layerCount - 1) continue; //last layer, no synapses
		s[i] = new synapse * [top[i]];
		for (int j = 0; j < top[i]; j++)
			s[i][j] = new synapse[top[i + 1]];

	}

	//creating all bias neurons and their synapses
	bn = new neuron[layerCount - 1];
	bs = new synapse * [layerCount - 1];
	for (int i = 0; i < layerCount - 1; i++)
		bs[i] = new synapse[top[i + 1]];


	//Connecting synapses to neurons
	for (int i = 0; i < layerCount - 1; i++)
		//For each layer i (except last one)
		for (int j = 0; j < topology[i]; j++)
			//for each neuron j in current layer
			for (int k = 0; k < topology[i + 1]; k++)
				//for each neuron in k next layer
				s[i][j][k].setNeurons(&n[i][j], &n[i + 1][k]);
	
	
	//Connecting bias synapses to neurons
	for (int i = 0; i < layerCount - 1; i++)
		//For each layer i (except last one)
		for (int j = 0; j < topology[i + 1]; j++)
			//for each neuron j in next layer
			bs[i][j].setNeurons(&bn[i], &n[i + 1][j]);
}
	
nn::~nn() {
	if (layerCount == 0) return;
	
	//deleting synapses
	for (int i = 0; i < layerCount - 1; i++)
		for (int j = 0; j < topology[i]; j++)
			delete[] s[i][j];			
	for (int i = 0; i < layerCount - 1; i++)
		delete[] s[i];
	delete[]s;

	//deleting nodes and bias synapses
	for (int i = 0; i < layerCount - 1; i++) {
		delete[] n[i];
		delete[] bs[i];
	}
	delete[] n[layerCount - 1];
	delete[] n;
	delete[] bs;

	//deleting bias neurons
	delete[] bn;


	//deleting topology
	delete[] topology;

}

void nn::print() {
	//print neurons
	for (int i = 0; i < layerCount; i++) 
		for (int j = 0; j < topology[i]; j++)
			n[i][j].print();

	std::cout << "========================" << std::endl;
	//print synapses
	for (int i = 0; i < layerCount - 1; i++)
		for (int j = 0; j < topology[i]; j++)
			for (int k = 0; k < topology[i + 1]; k++)
				s[i][j][k].print();

	std::cout << "========================" << std::endl;
	//print bias neurons
	for (int i = 0; i < layerCount - 1; i++)
		bn[i].print();

	std::cout << "========================" << std::endl;
	//print bias synapses
	for (int i = 0; i < layerCount - 1; i++)
		for (int j = 0; j < topology[i + 1]; j++)
			bs[i][j].print();

	std::cout << "========================" << std::endl;
	std::cout << "RMSE = " << RMSE << std::endl;
	std::cout << "RMSRE = " << RMSRE << std::endl;
}

void nn::printRMSE() {
	std::cout << "RMSE = " << RMSE << std::endl;
}

void nn::printRMSRE() {
	std::cout << "RMSRE = " << RMSRE << std::endl;
}

void nn::initialize() {
	if (layerCount == 0) {
		std::cout << "Error: Empty neural network, initialization method possibly called prior to creating neurons!";
		return;
	}

	for (int i = 0; i < layerCount - 1; i++) {
		//Synapeses
		for (int j = 0; j < topology[i]; j++) 
			for (int k = 0; k < topology[i + 1]; k++) 
				s[i][j][k].setInitialWeight(INITIAL_WEIGHT);
		//Bias a and synapses
		bn[i].changeType('b');
		bn[i].receive(INITIAL_BIAS);
		for (int j = 0; j < topology[i + 1]; j++)
			bs[i][j].setInitialWeight(INITIAL_WEIGHT);

	}

	//setting type of input layers
	for (int i = 0; i < topology[0]; i++)
		n[0][i].changeType('i');
}

void nn::ff(double* input) {
	//Note: can't validate size of input unless we change code to use vectors instead of dynamic arrays, which I think we should do in the future
	//Currently, if size of input was incorrect, an out of bound memory read exception will occur

	//feed input to input layer and fire input neurons 
	for (int i = 0; i < topology[0]; i++) {
		n[0][i].receive(input[i]);
		n[0][i].fire();
	}

	//fire all bias neurons
	for (int i = 0; i < layerCount - 1; i++)
		bn[i].fire();

	//for each next layer, synapses feed forward and neurons fire
	for (int i = 0; i < layerCount - 1; i++) {
		//regular synapses
		for (int j = 0; j < topology[i]; j++)
			for (int k = 0; k < topology[i + 1]; k++)
				s[i][j][k].ff();
		//bias synapses
		for (int j = 0; j < topology[i + 1]; j++)
			bs[i][j].ff();
		//neurons
		for (int j = 0; j < topology[i + 1]; j++)
			n[i + 1][j].fire();
	}

	

}


void nn::bp(double* Y) {

	//set SDW for output layer and calculate RMSE
	RMSE = 0;
	RMSRE = 0;
	int lastLayerIndex = layerCount - 1;
	int numberOutputNeurons = topology[lastLayerIndex];
	double sdw = 0;
	double a = 0;
	double delta = 0;
	double sumY = 0;
	for (int i = 0; i < numberOutputNeurons; i++) {
		a = n[lastLayerIndex][i].geta();
		delta = a - Y[i];
		sdw = 2 * delta * a;
		n[lastLayerIndex][i].receiveSDW(sdw);
		RMSE += delta * delta;
		RMSRE += (delta / ((Y[i] == 0) ? a : Y[i])) * (delta / ((Y[i] == 0) ? a : Y[i]));
		sumY += Y[i];
	}
	RMSE = sqrt(RMSE / numberOutputNeurons);
	RMSRE = sqrt(RMSRE / numberOutputNeurons);


	//iterate backwards, for each layer call bp for synapses and normalize sdw for neurons
	for (int i = layerCount - 2; i >= 0; i--) {
		for (int j = 0; j < topology[i]; j++) {
			for (int k = 0; k < topology[i + 1]; k++)
				s[i][j][k].bp();
			//normalize sdw depending on number of neurons on next layer
			n[i][j].normalizeSDW(topology[i + 1]);
		}
	}

	//again, iterate backwards this time for bias synapses and neurons
	for (int i = layerCount - 2; i >= 0; i--) {
		for (int j = 0; j < topology[i + 1]; j++)
			bs[i][j].bp();

		//normalize sdw depending on number of neurons in next layer
		bn[i].normalizeSDW(topology[i + 1]);
	}
}

void nn::getOutput(double* output) {
	for (int i = 0; i < topology[layerCount - 1]; i++)
		output[i] = n[layerCount - 1][i].geta();
}

void nn::reset() {
	//call reset function for every neuron in network
	for (int i = 0; i < layerCount; i++) 
		for (int j = 0; j < topology[i]; j++)
			n[i][j].reset();

	for (int i = 0; i < layerCount - 1; i++)
		bn[i].reset();
}

// **********************************************************
// **********************************************************


//Testing in main
int main() {
	// AND/OR with selector
	int top[3] = { 3,2,1 };
	nn myNeuralNetwork(3, top);
	myNeuralNetwork.initialize();
	myNeuralNetwork.print();
	double dataIn[3] = { 0, 0 ,0 };
	double dataOut[1] = { 0 };
	double dataNeeded[1] = { 0 };
	int r1 = 0;
	int r2 = 0;
	int r3 = 0;

	//training for 1000000 iterations
	for (int i = 0; i < 1000000; i++) {
		//std::cout << "Training Iteration " << i << std::endl;
		r1 = rand() % 10;
		r2 = rand() % 10;
		r3 = rand() % 10;
		dataIn[0] = (r1 % 2 == 0);
		dataIn[1] = (r2 % 2 == 0);
		dataIn[2] = (r3 % 2 == 0);
		dataNeeded[0] = dataIn[2]? (dataIn[0] && dataIn[1]) : (dataIn[0] || dataIn[1]);

		myNeuralNetwork.ff(dataIn);
		myNeuralNetwork.getOutput(dataOut);
		/*
		std::cout << "Data in: " << dataIn[0] << " " << dataIn[1]  << std::endl;
		std::cout << "Data needed: " << dataNeeded[0] << std::endl;
		std::cout << "Data out: " << dataOut[0] << std::endl;
		*/
		myNeuralNetwork.bp(dataNeeded);
		myNeuralNetwork.ff(dataIn);

		myNeuralNetwork.reset();
		//myNeuralNetwork.printRMSE();
		//myNeuralNetwork.printRMSRE();
	}

	std::cout << "Training Complete" << std::endl;
	myNeuralNetwork.printRMSE();

	//Training complete, testing the nn:
	while (1) {
		std::cout << "Enter Input 1 [1/0], Input 2 [1/0], and Selector [0 : OR / 1 : AND]" << std::endl;
		std::cin >> dataIn[0] >> dataIn[1] >> dataIn[2];
		myNeuralNetwork.ff(dataIn);
		myNeuralNetwork.getOutput(dataOut);
		myNeuralNetwork.reset();
		std::cout << dataOut[0] << std::endl;
		std::cout << "---------------------" << std::endl;
	}

}
