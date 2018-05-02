#pragma once

#include "main.h"

//numero delle features estratte = numero degli input
#define nOfFeatures 15
//numero dei nodi "hidden"
#define nOfHiddenNodes 60
//numero delle classi = numero dei nodi di out
#define nOfOutputNodes 1
//parametro di apprendimento //"eta" da esempio
#define learningRate 0.1
//secondo parametro di apprendimento //"alpha" da esempio
#define alpha 0
typedef struct 
{
	float value;
	float weights[nOfHiddenNodes];
	//float bias;
}
InputNode;

typedef struct 
{
	float value;
	float weights[nOfOutputNodes];
	float bias;
}
HiddenNode;

typedef struct 
{
	float value;
	float bias;
}
OutputNode;

void feedForward(InputNode in[], HiddenNode hn[], OutputNode on[]);
//setup con pesi random nell'intervallo [-0.5; 0.5]
void randomSetupNodes(InputNode in[], HiddenNode hn[], OutputNode on[]);
void train(InputNode in[], HiddenNode hn[], OutputNode on[], float inputFeatures[], int label);
int calculateOutput(InputNode in[], HiddenNode hn[], OutputNode on[], float inputFeatures[]);

double sigmoid(double x);
float fsigmoid(float x);
float generateRandomWeights();