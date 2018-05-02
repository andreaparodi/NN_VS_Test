#include "node.h"

//funzione sigmoide utilizzata per il calcolo del valore del nodo
double sigmoid(double x)
{
	return tanh(x);
}
float fsigmoid(float x)
{
	float result = 1 / (1 + expf(-x));
	return result;
}
float ftanh(float x)
{
	return (float)tanh(x);
}
void feedForward(InputNode in[], HiddenNode hn[], OutputNode on[])
{
	float hValues[nOfHiddenNodes] = {0};
	float hSigmoidValues[nOfHiddenNodes] = { 0 };

	float oValues[nOfOutputNodes] = { 0 };
	/*
	for (int h = 0; h < nOfHiddenNodes; h++)//per ogni nodo del primo livello
	{
		for (int i = 0; i < nOfFeatures; i++)//per ogni nodo del livello hidden
		{
			values[h] = 0;
			values[h] = values[h] + in[i].value*in[i].weights[h];
			//temp = in[i].value;
			//temp = temp + in[i].value*in[i].weights[h];
			//temp = temp + hn[h].value;
			//temp = temp + in[i].bias; //nuovo
			//temp = 0;
			
			/////values[h] = values[h] + (in[i].value*in[i].weights[h]);
		}
		//hn[h].value = temp;
		//temp = 0;
	}
	*/
	for (int h = 0; h < nOfHiddenNodes; h++)
	{
		//bias dello specifico nodo hidden
		hValues[h] = hn[h].bias;
		for (int i = 0; i < nOfFeatures; i++)
		{
			//calcolo dell'argomento della funzione di attivazione come somma dei prodotti in*peso
			hValues[h] = hValues[h] + (in[i].value*in[i].weights[h]);

			//applicazione della funzione di attivazione
			hSigmoidValues[h]= fsigmoid(hValues[h]);
			hn[h].value = fsigmoid(hValues[h]);
		}
	}
	//int placeholder = 0;
	for (int o = 0; o < nOfOutputNodes; o++)
	{
		//bias dello specifico nodo di output
		oValues[o] = on[o].bias;
		for (int h = 0; h < nOfHiddenNodes; h++)
		{
			//calcolo dell'argomento della funzione di attivazione come somma dei prodotti hid*peso
			oValues[o] = oValues[o] + (hn[h].value*hn[h].weights[o]);
			//applicazione della funzione di attivazione
			on[o].value = fsigmoid(oValues[o]);
		}
	}
	int placeholder2 = 0;


	//va ora applicata la sigmoide ai valori dei nodi hidden
	/*
	for (int h = 0; h < nOfHiddenNodes; h++)
	{
		hn[h].value = fsigmoid(hn[h].value);
	}
	*/
	//ora si propaga da hidden a output
	/*
	for (int h = 0; h < nOfHiddenNodes; h++)//per ogni nodo del livello hidden
	{
		for (int o = 0; o < nOfOutputNodes; o++)//per ogni nodo del livello out
		{
			/*
			temp = hn[h].value;
			temp = temp*hn[h].weights[o];
			//temp = temp + on[o].value;
			temp = temp + hn[h].bias;
			on[o].value = temp;
			temp = 0;
			
		}
	}
	*/
	//va ora applicata la sigmoide ai valori dei nodi out
	/*
	for (int o = 0; o < nOfOutputNodes; o++)
	{
		on[o].value = fsigmoid(on[o].value);
	}
	*/
}
void randomSetupNodes(InputNode in[], HiddenNode hn[], OutputNode on[]) 
{
	for (int i = 0; i < nOfFeatures; i++)
	{
		in[i].value = defaultNodeValue;
		for (int h = 0; h < nOfHiddenNodes; h++)
		{
			in[i].weights[h] = generateRandomWeights();
		}
	}
	for (int h = 0; h < nOfHiddenNodes; h++)
	{
		hn[h].value = defaultNodeValue;
		hn[h].bias = defaultBias;
		for (int o = 0; o < nOfOutputNodes; o++)
		{
			hn[h].weights[o] = generateRandomWeights();
		}
	}
	for (int o = 0; o < nOfOutputNodes; o++)
	{
		on[o].value = defaultNodeValue;
		on[o].bias = defaultBias;
	}
}

//nel caso in cui il training set occupi troppo spazio
void loadTrainedNetwork(InputNode in[], HiddenNode hn[], OutputNode on[])
{
	in[0].weights[0]= 0.050364453;
	in[0].weights[1] = 0.140978098;
	in[0].weights[2] = -0.143539980,
	in[0].weights[3] = -0.334958464;
	in[0].weights[4] = -0.188528284;
	in[0].weights[5] = 0.003182704;
	in[0].weights[6] = 0.231281325;
	in[0].weights[7] = 0.339421511;
	in[0].weights[8] = -0.285962641;
	in[0].weights[9] = 0.321480662;
	in[0].weights[10] = -0.129770562;
	in[0].weights[11] = -0.172099218;
	in[0].weights[12] = -0.148472920;
	in[0].weights[13] = 0.084917776;
	in[0].weights[14] = -0.272993892;
	in[0].weights[15] = -0.330143213;
	in[0].weights[16] = -0.167630628;
	in[0].weights[17] = -0.327026933;
	in[0].weights[18] = -0.279903024;
	in[0].weights[19] = 0.407478631;
	in[0].weights[20] = -0.269944727;
	in[0].weights[21] = -0.114695296;
	in[0].weights[22] = -0.327084571;
	in[0].weights[23] = -0.300511777;
	in[0].weights[24] = -0.047700591;
	in[0].weights[25] = -0.419359028;
	in[0].weights[26] = -0.067375891;
	in[0].weights[27] = -0.451978326;
	in[0].weights[28] = -0.422831237;
	in[0].weights[29] = -0.491236448;
}

//questa implementazione suppone che le etichette siano due e il nodo di uscita sia solo uno, (contrariamente a quanto fatto finora) che risulta tuttavia più semplice
/*
void train(InputNode in[], HiddenNode hn[], OutputNode on[], float inputFeatures[], int label) 
{
	calculateOutput(in, hn, on, inputFeatures);
	float delta_out = 0;
	float delta_hid = 0;
	float weight_adj_hidden_to_out[nOfHiddenNodes] = {0};

	//per prima cosa vanno sistemati i pesi tra livello hidden e out
	float out = on[0].value;
	delta_out = out;
	//delta_out = delta_out*(label - out); //questa è da dispense
	delta_out = delta_out*(out- label); //questa è video yt nei preferiti
	delta_out = delta_out*(1 - out);

	for (int h = 0; h < nOfHiddenNodes; h++)
	{
		weight_adj_hidden_to_out[h] = learningRate*delta_out*hn[h].value;
	}
	for(int i = 0; i <nOfFeatures; i++)
	{
		for(int h = 0; h <nOfHiddenNodes; h++)
		{
			float weight_adj = 0;
			float delta = delta_out;
			delta = delta*hn[h].value;
			delta = delta*(1 - hn[h].value);
			delta = delta*hn[h].weights[0];

			weight_adj = learningRate;
			weight_adj = weight_adj *delta;
			weight_adj = weight_adj *in[i].value;

			//aggiornamento pesi
			in[i].weights[h] = in[i].weights[h] + weight_adj;
		}
	}
	//aggiornamento pesi tra hidden e out
	for (int h = 0; h < nOfHiddenNodes; h++)
	{
		hn[h].weights[0] = hn[h].weights[0] + weight_adj_hidden_to_out[h];
	}
}
*/
//siccome le etichette sono salvate come '0' e '1' bisogna portarle in maniera tale che siano '-1' e '1'
int convertLabel(int label)
{
	if (label == 0)
		return -1;
	else
		return 1;
}
int calculateOutput(InputNode in[], HiddenNode hn[], OutputNode on[], float inputFeatures[])
{
	for (int i = 0; i < nOfFeatures; i++)
	{
		in[i].value = inputFeatures[i];
	}
	feedForward(in, hn, on);
	float result = on[0].value;
	if (on[0].value >= 0.5)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}
//tentativo n2
void train(InputNode in[], HiddenNode hn[], OutputNode on[], float inputFeatures[], int label)
{
	float delta_h[nOfHiddenNodes] = {0};
	float delta_o[nOfOutputNodes] = {0};
	
	//giusto per verifica, eliminabili
	float bias_hid[nOfHiddenNodes] = { 0 };
	float bias_o[nOfOutputNodes] = {0};
	
	float weight_adj_hid[nOfHiddenNodes] = { 0 };


	calculateOutput(in, hn, on, inputFeatures);

	//calcolo dei delta per il livello di output
	for (int o = 0; o < nOfOutputNodes; o++)
	{
		delta_o[o] = (label - on[o].value)*on[o].value*(1.0 - on[o].value);
	}
	//calcolo dei delta per il livello hidden
	for (int h = 0; h < nOfHiddenNodes; h++)
	{
		for (int o = 0; o < nOfOutputNodes; o++)
		{
			delta_h[h] = delta_h[h] + (delta_o[o]*hn[h].weights[o]);
		}
		delta_h[h] = delta_h[h] * ((hn[h].value)*(1- hn[h].value));
	}
	int placeholderrr = 0;
	//aggiornamento pesi input-hidden layer
	for (int h = 0; h < nOfHiddenNodes; h++)
	{
		//giusto per verifica
		bias_hid[h] = learningRate*delta_h[h];
		hn[h].bias = hn[h].bias + learningRate*delta_h[h];
		//soluzione temporanea per tenere la modifica dei pesi, sarebbe da cambiare con una matrice per verificare gli aggiustamenti
		float temp = 0;
		for (int i = 0; i < nOfFeatures; i++)
		{
			temp = learningRate*in[i].value*delta_h[h];
			in[i].weights[h] = in[i].weights[h] + temp;
			temp = 0;
		}
		
	}
	int placeholderz = 0;
	for (int o = 0; o < nOfOutputNodes; o++)
	{
		bias_o[o] = learningRate*delta_o[o];
		on[o].bias = learningRate*delta_o[o];
		//soluzione temporanea per tenere la modifica dei pesi, sarebbe da cambiare con una matrice (vettore se nOutput=1) per verificare gli aggiustamenti
		float temp = 0;
		for (int h = 0; h < nOfHiddenNodes; h++)
		{
			temp = learningRate*hn[h].value*delta_o[o];
			hn[h].weights[o] = hn[h].weights[o] + temp;
			temp = 0;
		}
	}
	int end = 1;
}

float generateRandomWeights() 
{
	float result = 0;
	result = (float)(rand() % 1000);
	result = result - 500;
	result = result / 1000;
	return result;
}
void printNetwork(InputNode in[], HiddenNode hn[], OutputNode on[], float weights[], float biases[])
{
	int index = 0;
	//salvo i pesi tra nodi input e hidden
	//weights contiene pesi salvati in questo ordine: tutti i pesi tra nodo 1 e hidden, tutti pesi tra nodo 2 e hid e così via
	for (int i = 0; i < nOfFeatures; i++)
	{
		for (int h = 0; h < nOfHiddenNodes; h++)
		{
			weights[index] = in[i].weights[h];
			index++;
		}
	}
	//salvo pesi tra hidden e out
	for (int h = 0; h < nOfHiddenNodes; h++)
	{
		for (int o = 0; o < nOfOutputNodes; o++)
		{
			weights[index] = hn[h].weights[o];
			index++;
		}
	}
	index = 0;
	for (int h = 0; h < nOfHiddenNodes; h++)
	{
		biases[index] = hn[h].bias;
		index++;
	}
	for (int o = 0; o < nOfOutputNodes; o++)
	{
		biases[index] = on[o].bias;
		index++;
	}
	index = 0;
	for (int i = 0; i < nOfFeatures; i++)
	{
		printf("#Nodo input ");
		printf("%i", i);
		printf(": \n");
		for (int h = 0; h < nOfHiddenNodes; h++)
		{
			printf("%.9f\n", weights[index]);
			index++;
		}
	}
	for (int h = 0; h < nOfHiddenNodes; h++)
	{
		printf("#Nodo hidden ");
		printf("%i", h);
		printf(": \n");
		for (int o = 0; o < nOfOutputNodes; o++)
		{
			printf("%.9f\n", weights[index]);
			index++;
		}
	}
	index = 0;
	printf("#Bias nodi hidden:\n");
	for (int h = 0; h < nOfHiddenNodes; h++)
	{
		printf("%.9f\n", biases[index]);
		index++;
	}
	printf("#Bias nodi(o) output:\n");
	for (int o = 0; o < nOfOutputNodes; o++)
	{
		printf("%.9f\n", biases[index]);
		index++;
	}
}