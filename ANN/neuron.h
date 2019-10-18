#pragma once

class neuron {
private:
	char type;
	double a, SI, DSI, SDW;
	int id; //temp
public:
	void fire();
	void normalizeSDW(int);
	static int count;
	neuron();
	void receive(double);
	void receiveSDW(double);
	void reset();
	double getDSI();
	double geta();
	double getSDW();
	void changeType(char);
	int getid(); //temp
	void print(); //temp
	char getType();

};