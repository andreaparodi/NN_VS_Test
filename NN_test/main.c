#pragma once
#include "main.h"

void main()
{
	//test
	/*
	double x = sigmoid(-1);
	double y = sigmoid(0);
	double z = sigmoid(1);
	double h = sigmoid(30);

	float x_float = (float)x;
	float y_float = (float)y;
	float z_float = (float)z;
	float h_float = (float)h;

	float xx_float = fsigmoid(-1);
	float yy_float = fsigmoid(0);
	float zz_float = fsigmoid(1);
	float hh_float = fsigmoid(30);
	*/

	//test.value = 000;

	InputNode inputNodes[nOfFeatures];
	HiddenNode hiddenNodes[nOfHiddenNodes];
	OutputNode outputNodes[nOfOutputNodes];
	
	srand(time(NULL));

	//inizializzazione ai valori standard
	setupNodes(inputNodes, hiddenNodes, outputNodes);

	//inizializzazione dei nodi, portare a funzione possibilmente
	/*
	for (int i = 0; i < nOfFeatures; i++)
	{
		inputNodes[i].value = defaultNodeValue;
		for (int j = 0; j < nOfHiddenNodes; j++)
		{
			inputNodes[i].weights[j] = defaultWeight;
		}
	}
	for (int i = 0; i < nOfHiddenNodes; i++)
	{
		hiddenNodes[i].value = defaultNodeValue;
		for (int j = 0; j < nOfOutputNodes; j++)
		{
			hiddenNodes[i].weights[j] = defaultWeight;
		}
	}
	for (int i = 0; i < nOfOutputNodes; i++)
	{
		outputNodes[i].value = defaultNodeValue;
	}
	*/

	//test!

	/*
	inputNodes[0].value = 5;
	inputNodes[6].value = 2;
	inputNodes[3].value = 1;
	inputNodes[4].value = 0;
	inputNodes[1].value = -2;
	inputNodes[5].value = 3;
	inputNodes[9].value = 1;

	inputNodes[0].weights[0] = 0;
	inputNodes[0].weights[1] = 0;

	inputNodes[1].weights[0] = 0;
	inputNodes[3].weights[0] = -10;


	feedForward(inputNodes, hiddenNodes, outputNodes);
	*/
	//test 2 bias
	/*
	inputNodes[0].value = 1;
	inputNodes[1].value = 2;
	inputNodes[2].value = 3;
	inputNodes[3].value = 4;
	inputNodes[4].value = 5;
	inputNodes[5].value = 6;

	hiddenNodes[0].bias = -20.9;
	inputNodes[0].weights[0] = 0;
	inputNodes[0].weights[1] = 0;

	for (int i = 0; i < nOfFeatures; i++)
	{
		inputNodes[i].weights[7] = 0;	
	}
	*/
	inputNodes[5].value = 0;
	//inputNodes[1].weights[0] = 0;
	//inputNodes[3].weights[0] = -10;

	//int placeholder2 = 0;
	feedForward(inputNodes, hiddenNodes, outputNodes);
	//int placeHolder = 0;

	float test[nOfFeatures] = { 0.76, 0.20, 0.0, 0.8, -0.3, 0.19, 0.32, -0.7, -0.9, 0.43, -0.12, 0.10, 0.15, 0.2, -0.19 };
	int testLabel = 0;

	int notTrainedTestLabel = calculateOutput(inputNodes, hiddenNodes, outputNodes, test);
	int placeholder3 = 0;
	
	train(inputNodes, hiddenNodes, outputNodes, test, testLabel);
	
	int trainedTestLabel = calculateOutput(inputNodes, hiddenNodes, outputNodes, test);
	int end = 1;
	/*
	float trainingSetFeatures[nOfSamples][nOfFeatures] =
	{
		{ 0.713942,        	0.556642,		1.818262,		-3.407425,		0.015400,        143.818665,      2.918958,        2.650967,        6.445556,        2.918958,        2.650967,        6.445556,        -0.327651,       -0.417206,       0.052923 },
		{ 0.612171,        	0.425334,		0.326917,		-9.584925,		4.153275,        142.931946,      1.576644,        2.866804,        5.199918,        1.576644,        2.866804,        5.199918,        0.613374,        0.091030,        0.065601 },
		{ 0.174003,        	0.602562,		5.912107,		-5.844824,		3.933475,        143.741852,      3.012391,        2.920674,        5.908579,        3.012391,        2.920674,        5.908579,        -0.861947,       0.846054,        -0.558840 },
		{ 0.213344,        	0.281256,		3.716173,		-12.699049,		-0.014700,       143.142120,      1.550633,        2.409202,        2.486700,        1.550633,        2.409202,        2.486700,        0.336187,        -0.039629,       -0.651001 },
		{ 0.412951,        	0.409984,		3.131670,		-13.817649,		1.169350,        142.888733,      1.345184,        1.415121,        5.508087,        1.345184,        1.415121,        5.508087,        0.241658,        0.166274,        0.066907 },
		{ 1.175005,        	0.112283,		-5.735976,		-14.368900,		4.154850,        143.230682,      1.576398,        2.089843,        4.435951,        1.576398,        2.089843,        4.435951,        0.381381,        -0.497589,       -0.011275 },
		{ 0.125053,        	0.341492,		7.294523,		-13.451900,		0.559300,        143.446976,      1.577515,        1.930671,        8.666748,        1.577515,        1.930671,        8.666748,        0.543265,        0.913650,        -0.057672 },
		{ -0.644095,       	0.923739,		10.179117,		-14.338627,		3.524499,        143.384659,      1.523255,        1.970914,        7.138545,        1.523255,        1.970914,        7.138545,        -0.158627,       0.270767,        -0.045072 },
		{ 0.451712,     		0.131180,		2.372969,		-16.640051,		1.940750,        143.054977,      1.165434,        1.730089,        2.089404,        1.165434,        1.730089,        2.089404,        -0.829186,       0.076683,        -0.057197 },
		{ 0.633454,        	0.104544,		1.367647,		-16.536976,		4.694376,        142.350601,      1.437268,        1.892786,        2.974824,        1.437268,        1.892786,        2.974824,        -0.593238,       0.031985,        0.228636 },
		{ 1.072654,        	0.064816,		-3.773637,		-15.286247,		9.389277,        142.489700,      1.079115,        2.144551,        3.274465,        1.079115,        2.144551,        3.274465,        -0.297085,       0.075839,        0.282014 },
		{ 1.101353,        	0.021992,		-6.290104,		-14.307123,		6.177325,        143.037476,      1.807241,        2.130175,        3.182556,        1.807241,        2.130175,        3.182556,        0.344591,        0.206094,        0.085990 },
		{ 0.770696,        	-0.013028,		-2.002584,		-10.544625,		5.415726,        143.270584,      1.215982,        2.108036,        4.544981,        1.215982,        2.108036,        4.544981,        -0.876428,       -0.104399,       -0.043856 },
		{ 0.458806,        	0.488924,		0.731097,		-9.425850,		2.588774,        142.973434,      1.324637,        2.443218,        2.555637,        1.324637,        2.443218,        2.555637,        0.319182,        -0.033933,       -0.548366 },
		{ 0.728259,        	0.082487,		-0.375932,		-8.646399,		3.619701,        143.563339,      1.315418,        2.145947,        2.644065,        1.315418,        2.145947,        2.644065,        -0.606521,       0.072447,        -0.423163 },
		{ 1.252591,        	-0.068298,		-9.751205,		-8.371302,		5.943175,        143.376801,      1.094234,        2.348646,        4.176978,        1.094234,        2.348646,        4.176978,        0.014919,        -0.396599,       0.265199 },
		{ 0.465255,        	0.348006,		2.849705,		-6.127800,		4.033575,        143.540588,      1.175805,        2.112827,        4.014213,        1.175805,        2.112827,        4.014213,        -0.309226,       0.200109,        -0.138511 },
		{ 0.441715,        	0.220309,		1.480317,		-9.859324,		4.491899,        143.216141,      1.124292,        2.124532,        4.288002,        1.124292,        2.124532,        4.288002,        0.294888,        -0.364065,       -0.108953 },
		{ 1.106190,        	0.104737,		-5.357336,		-10.908800,		5.390525,        143.040115,      1.172850,        2.150508,        3.350538,        1.172850,        2.150508,        3.350538,        -0.870064,       -0.508305,       -0.077260 },
		{ 0.908454,        	-0.099449,		-1.886238,		-14.028877,		7.163450,        142.588776,      1.013098,        2.379738,        3.112856,        1.013098,        2.379738,        3.112856,        -0.057213,       -0.222283,       -0.019582 },
		{ 0.280353,        	0.898457,		4.380969,		-8.222025,		7.051450,        142.683807,      2.449189,        2.482212,        20.068745,       2.449189,        2.482212,        20.068745,       0.396890,        -0.579298,       0.039194 },
		{ 0.766762,        	0.228887,		-4.382648,		-9.282348,		5.051375,        142.729996,      1.465303,        2.158673,        13.726238,       1.465303,        2.158673,        13.726238,       0.265752,        -0.734828,       -0.006554 },
		{ 0.719617,        	0.956501,		1.274131,		-8.926048,		3.051125,        143.516418,      2.017431,        2.796547,        14.627744,       2.017431,        2.796547,        14.627744,       0.330295,        0.034697,        -0.000075 },
		{ 0.526459,        	0.470866,		1.200609,		-12.967150,		4.229576,        142.723358,      1.219039,        2.341258,        1.942219,        1.219039,        2.341258,        1.942219,        0.057744,        -0.123421,       0.364608 },
		{ 0.584439,        	0.147819,		1.232985,		-10.368575,		2.793000,        143.093796,      1.347355,        2.389926,        7.925227,        1.347355,        2.389926,        7.925227,        0.290707,        -0.164234,       0.038540 },
		{ 0.485377,        	0.432106,		-0.253717,		-13.082825,		4.856424,        143.145615,      1.674440,        2.714810,        6.767797,        1.674440,        2.714810,        6.767797,        0.557035,        -0.044336,       0.137005 },
		{ 0.650287,        	0.656736,		-1.546810,		-14.143499,		3.367351,        142.852661,      1.691030,        2.342486,        5.668236,        1.691030,        2.342486,        5.668236,        -1.000000,       -0.166604,       0.135145 },
		{ -0.324466,       	0.917483,		12.921828,		-13.105399,		0.792750,        143.196579,      1.716303,        2.509724,        5.288964,        1.716303,        2.509724,        5.288964,        0.947762,        0.580067,        -0.148281 },
		{ 0.242302,       	0.385928,		3.879342,		-13.320475,		4.807250,        142.642838,      1.348820,        2.299210,        4.773266,        1.348820,        2.299210,        4.773266,        0.156015,        0.083218,        0.136538 },
		{ 0.492600,       	0.182839,		1.517400,		-12.965050,		4.952150,        142.390869,      1.343269,        2.422253,        2.254429,        1.343269,        2.422253,        2.254429,        -0.248773,       0.246996,        0.425502 },
		{ 0.538842,        	0.172004,		1.178036,		-13.516476,		3.167150,        142.500565,      1.258169,        2.472537,        2.049111,        1.258169,        2.472537,        2.049111,        1.000000,        -0.084452,       0.581373 },
		{ 0.713103,        	0.037922,		0.228693,		-15.393001,     2.597875,        142.605560,      1.372984,        2.241966,        2.197016,        1.372984,        2.241966,        2.197016,        0.146785,        0.162356,        0.398473 },
		{ 0.780499,        	-0.116411,		0.213473,		-11.856952,     4.441851,        142.975189,      1.421696,        2.645764,        6.615953,        1.421696,        2.645764,        6.615953,        1.000000,        0.011264,        0.108255 },
		{ -0.099513,       	0.626424,		7.938684,		-12.361126,     2.189250,        142.530182,      1.182529,        2.616374,        2.879770,        1.182529,        2.616374,        2.879770,        0.060597,        0.319224,        0.007364 },
		{ 0.123440,        	0.640806,		5.486904,		-13.923174,     1.425200,        143.049545,      1.327052,        2.423693,        2.875464,        1.327052,        2.423693,        2.875464,        0.447794,        0.324599,        0.115439 },
		{ 0.226372,        	0.733806,		4.891436,		-12.304249,     2.392600,        143.048660,      1.199520,        2.187471,        3.645622,        1.199520,        2.187471,        3.645622,        0.601558,        0.173913,        0.045562 },
		{ 0.556062,        	0.127890,		-0.084873,		-11.211200,     4.427849,        142.810486,      1.081794,        2.823575,        3.579734,        1.081794,        2.823575,        3.579734,        1.000000,        -0.266439,       0.143556 },
		{ 0.640613,        	0.292606,		1.044728,		-11.685800,     2.738925,        143.236282,      1.030218,        2.436898,        2.815125,        1.030218,        2.436898,        2.815125,        -1.000000,       0.155980,        -0.035023 },
		{ -0.583794,       	0.700463,		13.323877,		-11.616151,     2.697100,        143.040619,      1.212546,        2.361864,        3.309625,        1.212546,        2.361864,        3.309625,        0.849133,        0.413490,        -0.091059 },
		{ 0.498211,       	0.145884,		1.034087,		-14.246923,     3.295075,        143.178497,      1.227577,        2.351060,        2.231011,        1.227577,        2.351060,        2.231011,        0.217173,        -0.018139,       0.384134 },
		{ 0.919740,        	0.045532,		-2.708529,		-12.575327,     5.551701,        142.603149,      1.132022,        2.209406,        3.747874,        1.132022,        2.209406,        3.747874,        0.615292,        -0.473673,       -0.077596 },
		{ 0.206508,        	0.193029,		5.802405,		-13.671700,     1.998150,        143.479858,      1.510056,        2.571752,        4.626489,        1.510056,        2.571752,        4.626489,        -0.369397,       0.310037,        -0.156593 },
		{ -0.022573,       	0.336720,		7.294652,		-12.579176,     3.162250,        143.032761,      1.210038,        2.823218,        3.601262,        1.210038,        2.823218,        3.601262,        0.076848,        0.149404,        0.134815 },
		{ 0.114669,        	0.683501,		6.835459,		-12.552573,     2.892925,        142.799835,      1.344595,        2.365141,        3.332313,        1.344595,        2.365141,        3.332313,        1.000000,        0.236408,        0.028017 },
		{ 0.093128,        	0.565800,		6.298102,		-13.206900,     3.826900,        143.004578,      1.338341,        2.352524,        4.239812,        1.338341,        2.352524,        4.239812,        1.000000,        0.416424,        -0.179170 },
		{ 0.106221,        	0.306279,		5.599702,		-12.661250,     3.454325,        143.078064,      1.379984,        2.582642,        5.758764,        1.379984,        2.582642,        5.758764,        0.587026,        0.188409,        -0.327012 },
		{ 0.874788,        	0.285383,		-1.509274,		-11.346297,     7.588349,        143.373474,      1.426815,        2.531761,        4.326740,        1.426815,        2.531761,        4.326740,        0.513129,        -0.395794,       0.167619 },
		{ 0.806490,        	-0.068621,		-1.960728,		-11.519376,     8.547700,        143.179413,      1.654357,        2.192947,        6.681862,        1.654357,        2.192947,        6.681862,        0.226287,        -0.062263,       -0.014735 },
		{ -0.626747,       	0.959855,		12.463412,		-12.495350,     5.260150,        142.939636,      1.541508,        1.838494,        11.007656,       1.541508,        1.838494,        11.007656,       -0.587069,       0.024260,        0.015319 },
		{ 0.256684,        	0.584890,		2.295513,		-15.675277,     4.198600,        142.809967,      1.661571,        2.285452,        6.726266,        1.661571,        2.285452,        6.726266,        -0.532575,       -0.130010,       -0.060698 },
		{ 0.555739,        0.638098,        1.765119,        -9.861600,       -3.451175,       140.669907,      5.816241,        5.630680,        11.691554,       5.816241,        5.630680,        11.691554,       1.000000,        -0.599303,       0.192032 },
		{ 2.353492,       0.611268,        -1.419435,       -4.348050,       -9.359177,       138.790924,      10.167748,       5.702479,        10.055149,      10.167748,       5.702479,        10.055149,       0.906043,        1.000000,        -0.090569 },
		{ 0.956759,        0.799718,       6.080693,       -9.748375,       -1.438150,       140.265656,      8.993824,        6.557792,        7.640732,        8.993824,        6.557792,        7.640732,      0.176867,        1.000000,        0.011361 },
		{ -0.441844,       -0.172262,       3.145858,        -7.840173,       -0.365050,       138.731079,      5.697924,        8.564024,        7.002821,        5.697924,        8.564024,        7.002821,        1.000000,        -0.197614,       -0.044093 },
		{ 0.760828,        -0.769019,       -0.278611,       -26.966797,      -11.589199,      137.582031,      8.460344,        5.110650,        7.829276,        8.460344,        5.110650,        7.829276,        0.652340,        0.759402,        -0.074626 },
		{ 0.012770,        -0.222696,       2.508018,        -9.701300,       -3.226300,       140.440140,      8.953351,        5.241531,        7.739034,        8.953351,        5.241531,        7.739034,       -0.076429,       -0.851497,       -0.572989 },
		{ -0.150979,       0.697947,        10.765234,       -6.678525,       -14.233277,      144.026596,      6.709936,        7.427036,        12.689460,       6.709936,        7.427036,        12.689460,       -0.362032,       0.134830,        0.231388 },
		{ -0.650545,       -0.983073,       5.307419,        -12.315800,      4.684575,        140.078552,      8.970109,        6.467745,        10.145947,       8.970109,        6.467745,        10.145947,       -0.606604,       -0.170063,       0.324408 },
		{ 3.810527,        0.308278,        -4.918458,       -18.347347,      8.996747,        139.466080,      6.113780,        6.640101,        9.568777,        6.113780,        6.640101,        9.568777,        0.877734,       -0.635399,       0.030053 },
		{ -0.876336,       0.366064,        -0.801459,       -12.181749,      -11.364676,      141.448853,      6.784817,        9.111094,        7.972156,        6.784817,        9.111094,        7.972156,        -0.205852,       -0.460793,       0.052825 },
		{ 1.728035,        0.563092,        0.346458,        -10.042724,      9.861424,        139.383270,      7.347293,        7.077485,        8.984610,        7.347293,        7.077485,        8.984610,        0.505647,       -1.000000,       0.262947 },
		{ 2.225796,        2.604178,        8.793028,        -21.801678,      -11.837001,      137.549820,      10.907845,       8.364674,        22.739645,       10.907845,       8.364674,        22.739645,       -0.697623,       -0.246411,       0.170674 },
		{ -0.370837,       0.475123,        -0.899618,       -31.667124,      -11.852225,      135.209030,      9.107673,        7.301165,        12.794623,       9.107673,        7.301165,        12.794623,       1.000000,        0.545387,        -0.492634 },
		{ -0.779467,       -1.125087,       7.462271,        -16.760626,      24.473923,       138.807907,      11.866672,       5.599699,        7.404382,        11.866672,       5.599699,        7.404382,        0.025307,        -0.101941,       0.223112 },
		{ -0.035923,       0.748188,        5.014618,        -18.192476,      -10.329379,      137.524979,      8.044081,        8.093778,        9.968178,        8.044081,        8.093778,        9.968178,        1.000000,        0.295019,        -0.061826 },
		{ 1.269359,        -0.581859,       0.348651,        -15.509374,      25.415081,       139.679596,      9.137956,        5.015593,        9.633342,        9.137956,        5.015593,        9.633342,        0.011243,        -0.442861,       0.382588 },
		{ 2.749739,        -0.394957,       -4.003813,       -3.073350,      -8.460202,       141.713409,      9.131365,        6.119473,        18.853302,       9.131365,        6.119473,        18.853302,       0.231939,        -0.707586,       -0.128024 },
		{ -1.692886,       -0.071458,       3.405508,        -16.435999,      5.236525,        139.691467,      9.169026,        9.106875,        9.261450,        9.169026,        9.106875,        9.261450,        -0.099544,       0.686718,        0.307159 },
		{ 1.761830,        -1.275357,       1.681213,        -15.260877,      19.804579,       137.792358,      10.092237,       6.060471,        11.212069,       10.092237,       6.060471,        11.212069,       0.173114,        1.000000,        0.264568 },
		{ 3.631428,        1.314569,        -13.844213,      -12.046650,      -8.710101,       139.034698,      10.980105,       7.253848,        19.166941,       10.980105,       7.253848,        19.166941,       -0.038668,       -1.000000,       0.035311 },
		{ -1.160752,       -0.710395,       -3.012292,       -5.346251,       5.617323,        139.713013,      7.491157,        7.899293,        8.208830,        7.491157,        7.899293,        8.208830,        0.583842,        -0.335056,       0.312356 },
		{ 2.209865,        -1.205833,       4.496864,        11.835949,       -16.583176,      137.834549,      7.387317,        5.147671,        9.375200,        7.387317,        5.147671,        9.375200,        0.357634,        1.000000,        -0.353957 },
		{ 1.671540,        0.646997,        -0.043146,       9.457700,        -1.060151,       138.537170,      7.902226,        4.551535,        11.301497,       7.902226,        4.551535,        11.301497,       0.503424,        1.000000,        -0.271054 },
		{ -0.703752,       0.900844,        0.703751,        -22.425901,      30.300898,       137.578522,      7.796834,        6.859141,        9.000793,        7.796834,        6.859141,        9.000793,        0.350850,        0.190963,        1.000000 },
		{ 0.571927,        0.443650,        -1.480768,       -29.505878,     -19.585300,      138.607178,      6.710761,        5.583373,        8.210958,        6.710761,        5.583373,        8.210958,        0.557047,        0.017359,        -0.169343 },
		{ 2.334274,        0.345233,        3.318572,        -16.134825,      27.220726,       136.624786,      11.588893,       7.681242,        12.856382,       11.588893,       7.681242,        12.856382,       -0.441676,       0.178148,        0.282629 },
		{ 0.769535,        0.873111,        -7.647367,       20.494423,       -19.973099,      134.749329,      9.082520,        10.015201,       13.759643,       9.082520,        10.015201,       13.759643,       0.630840,        -0.617455,       -0.016715 },
		{ -1.683406,       2.578252,        -7.917853,       -25.889673,      -31.567898,      132.282700,      11.283733,       8.410786,        16.660498,       11.283733,       8.410786,        16.660498,       -0.830150,       0.998795,        -0.122349 },
		{ -1.878563,       1.371710,        6.280881,        -46.015205,      7.802024,        131.658813,      9.342642,        6.425677,        10.366064,       9.342642,        6.425677,        10.366064,       -0.021496,       -0.813193,       0.549072 },
		{ 0.020702,        -1.260394,       3.790018,        -27.115376,      -46.932892,      130.592194,      6.647960,        5.472942,        5.701294,        6.647960,        5.472942,        5.701294,        0.227123,        -0.002415,       -0.607178 },
		{ 1.418338,        -2.129056,       -1.246592,       -5.162499,       34.169449,       136.147369,      8.685479,        7.576622,        8.131395,        8.685479,        7.576622,        8.131395,        -0.154642,       0.555569,        0.711780 },
		{ 1.621686,        -0.146980,       12.605621,       25.225376,       -33.377403,      133.766312,      6.177016,        5.991683,        10.232461,       6.177016,        5.991683,        10.232461,       -0.197445,       -0.038451,       0.001128 },
		{ -0.411790,       -0.067202,       -1.252268,       -26.776396,      29.700472,       136.068970,      9.084647,        7.503750,        11.454255,       9.084647,       7.503750,        11.454255,       0.115547,        -0.109180,       -0.151622 },
		{ 0.783466,        -1.665671,       0.775339,        -9.389626,       -13.618674,      140.629471,      11.628453,       5.731861,        15.235204,       11.628453,       5.731861,        15.235204,       0.257899,        1.000000,        -0.239917 },
		{ 1.713912,        -0.568122,       -7.533408,       -5.607525,       30.329424,       140.417191,      11.962025,       6.053595,        12.415768,       11.962025,       6.053595,        12.415768,       -0.240809,       -0.801863,       -0.110432 },
		{ -0.866985,       -0.233660,       0.247333,        -10.327975,      20.853525,       139.369644,      14.264810,       5.729699,        8.214917,        14.264810,       5.729699,        8.214917,        0.202862,        -0.170604,       0.190182 },
		{ 1.796334,        -1.949184,       6.359951,        18.711174,       -5.980273,       137.230438,      10.047881,       10.983697,       8.052500,        10.047881,       10.983697,       8.052500,        -0.504972,       0.262589,        -0.869873 },
		{ 2.158916,        2.720396,        -12.697779,      -10.742724,      -0.608650,      132.108551,      7.274404,        7.631165,        7.726500,        7.274404,        7.631165,        7.726500,        -1.000000,       -0.033915,       -0.321391 },
		{ 2.471322,        -1.126183,       -2.991396,       -24.947649,      23.314026,       137.628220,      9.554283,        9.251644,        13.653337,       9.554283,        9.251644,        13.653337,       0.006864,        -0.636835,       0.404830 },
		{ 1.067301,        1.272970,        5.221642,        2.292850,        -1.152201,       139.000916,      9.630769,        6.874348,        13.844775,       9.630769,        6.874348,        13.844775,       -0.049878,       -0.008005,       0.159429 },
		{ 2.256688,        0.887945,        1.521850,        -22.166374,      7.696324,        137.705231,      10.367709,       8.330601,        14.819776,       10.367709,       8.330601,        14.819776,       0.583099,        0.710437,        0.487588 },
		{ 2.433077,        0.654221,        -2.671252,       -32.824051,      -12.449150,      139.637924,      7.452524,        7.471720,        6.086591,        7.452524,        7.471720,        6.086591,        0.178462,        -0.079892,       0.216686 },
		{ -1.379513,       0.615009,        0.996423,        -13.381902,      -3.930148,       139.908295,      10.988699,       4.990869,        11.241116,       10.988699,       4.990869,        11.241116,       0.359484,        -0.235714,       0.132394 },
		{ 3.584026,        -0.772953,       -3.396286,       -21.035524,      28.302219,       136.599915,      10.230189,       4.467750,        7.168789,        10.230189,       4.467750,        7.168789,        0.092728,        -0.959730,       0.740558 },
		{ -2.785082,       1.192096,        5.340892,        -26.346949,      -17.908100,      136.178009,      9.429276,        4.747312,        13.156721,       9.429276,        4.747312,        13.156721,       0.283946,        1.000000,        0.483724 },
		{ 1.404344,        -0.930317,       -0.801073,       -11.966497,      19.844471,       134.251266,      16.187851,       7.635839,        18.098358,       16.187851,       7.635839,        18.098358,       -0.134369,       0.586672,        -0.021713 },
		{ 0.811068,        -0.385606,       9.317745,        2.456825,        -27.084578,      135.061340,      5.396933,        9.805336,        10.689906,       5.396933,        9.805336,        10.689906,       -1.000000,       -1.000000,       -0.219616 },
		{ 2.895882,        0.587083,        0.903552,        -11.703473,      8.152550,        139.845642,      8.818317,        8.416117,        12.259288,       8.818317,        8.416117,        12.259288,       -1.000000,       -1.000000,       -0.399782 },
		{ -1.328628,       0.956501,        13.580758,       -3.270925,       20.398876,       138.524948,      8.162104,        4.903562,        13.670374,       8.162104,        4.903562,        13.670374,       -0.106058,       -0.365110,       0.040404 },
		{ 1.546358,        -0.453517,       6.143638,        4.246376,        2.898177,        136.485123,      8.862868,        6.091926,        13.967298,       8.862868,        6.091926,        13.967298,       -0.203902,       1.000000,        -0.040541 },
	};
	int trainingLabels[nOfSamples] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
	//training set normalization
	float max_values[nOfFeatures] = { 0 };

	for (int i = 0; i < nOfFeatures; i++)
	{
		for (int sample = 0; sample < nOfSamples; sample++)
		{
			float temp_val = fabs(trainingSetFeatures[sample][i]);
			if (temp_val > max_values[i])
			{
				max_values[i] = temp_val;
			}
		}
	}
	for (int i = 0; i < nOfFeatures; i++)
	{
		for (int sample = 0; sample < nOfSamples; sample++)
		{
			trainingSetFeatures[sample][i] = trainingSetFeatures[sample][i] / max_values[i];
		}
	}

	float test[nOfFeatures] = {0.76, 0.20, 0.0, 0.8, -0.3, 0.19, 0.32, -0.7, -0.9, 0.43, -0.12, 0.10, 0.15, 0.2, -0.19};
	int testLabel = 0;

	int notTrainedTestLabel = calculateOutput(inputNodes, hiddenNodes, outputNodes, test);
	for (int i = 0; i < 20; i++)
	{
		train(inputNodes, hiddenNodes, outputNodes, test, testLabel);
	}
	int trainedTestLabel = calculateOutput(inputNodes, hiddenNodes, outputNodes, test);
	*/
}