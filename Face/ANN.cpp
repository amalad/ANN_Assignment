#include<iostream>
#include<stdlib.h>
#include<vector>
#include<time.h>
#include<math.h>
#include<fstream>
#include<string>
#include<sstream>
#include<algorithm>
using namespace std;

#define OLN 20 //Number of output nodes
#define HLN 20 //Number of hidden layer nodes
#define IN 960 //Length of input vector
int inpn = 80; //Number of training examples
double hlw[IN+1][HLN], olw[HLN+1][OLN]; //hidden layer weights and output layer weights
double hlwp[IN+1][HLN], olwp[HLN+1][OLN]; //delta hidden layer weights and output layer weights in previous iteration for momentum term
double testinp1[36][IN+1], testout1[36][OLN], testinp2[40][IN+1], testout2[40][OLN];
int epochs = 500;
double learningRate = 0.3, momentum = 0.3;


void prediction(double olop[], double hlop[], double inp[]) //feedforward
{
	hlop[0] = 1;
	//to find hidden layer outout
	for(int i=0; i<HLN; i++)
	{
		double temp = 0;
		for(int j=0; j<=IN; j++)
		{
			temp+=inp[j]*hlw[j][i];
		}
		hlop[i+1] = 1/(1+exp(-temp));
	}
	//to find output layer outout
	for(int i=0; i<OLN; i++)
	{
		double temp = 0;
		for(int j=0; j<=HLN; j++)
		{
			temp+=hlop[j]*olw[j][i];
		}
		olop[i] = 1/(1+exp(-temp));
	}
}

void getAcc(double inp[][IN+1], double target[][OLN], int count)
{
	int p=0;
	double hlop[HLN+1], olop[OLN];
	for(int i=0; i<count; i++)
	{
		prediction(olop, hlop, inp[i]);
		double max = 0;
		int maxInd = -1,j;
		for(j=0; j<OLN; j++)
			if(olop[j]>max)
			{
				max = olop[j];
				maxInd = j;
			}
		for(j=0; j<OLN; j++)
			if(j==maxInd)
				olop[j] = 1;
			else
				olop[j] = 0;
		for(j=0; j<OLN; j++)
			if(target[i][j]==1)
				break;
		if(maxInd==j)
		{
			p++;
		}
	}
	cout << ((double)p)/count << endl;
	
}



void randomizeWeights() //initialize weights to random values
{
	srand(time(NULL));
	for(int i=0; i<=IN; i++)
		for(int j=0; j<HLN; j++)
			hlw[i][j] = (2*(double)(rand()%100)/100)-1;
	for(int i=0; i<=HLN; i++)
		for(int j=0; j<OLN; j++)
			olw[i][j] = (2*(double)(rand()%100)/100)-1;
}


void updateNetwork(double prediction[], double hiddenlayerop[], double targetOutput[], double input[])
{
	//hidden layer updation
	for(int j=0; j<HLN; j++)
	{
		double delkw = 0;
		for(int k=0; k<OLN; k++)
			delkw+=(targetOutput[k]-prediction[k])*prediction[k]*(1-prediction[k])*olw[j+1][k];
		for(int i=0; i<=IN; i++)
		{ 
			double del = learningRate*delkw*hiddenlayerop[j+1]*(1-hiddenlayerop[j+1])*input[i];
			hlwp[i][j]= del+ momentum*hlwp[i][j];
			hlw[i][j]+= hlwp[i][j];

		}
	}
	//output layer updation
	for(int j=0; j<OLN; j++)
	{
		double delj = (targetOutput[j]-prediction[j])*prediction[j]*(1-prediction[j]);
		for(int i=0; i<=HLN; i++)
		{	
			double del = learningRate*delj*hiddenlayerop[i]; 
			olwp[i][j]= del + momentum*olwp[i][j];
			olw[i][j]+= olwp[i][j]; 

		}
	}
}

void stochasticGradientDescent(double inp[][IN+1], double actualOutput[][OLN]) 
{	
	srand(time(NULL));
	randomizeWeights();
	int shuf[inpn];
	for(int i=0; i<inpn; i++)
		shuf[i] = i;
	for(int k = 1; k<=epochs; k++)
	{
		random_shuffle(&shuf[0],&shuf[inpn]); //to shuffle the order for every epoch
		double olop[OLN];
		double hlop[HLN+1];
		hlop[0] = 1;
		//cout << "Iteration " << k << ":" << endl;
		for(int i=0; i<inpn; i++)
		{
			prediction(olop, hlop, inp[shuf[i]]);
			updateNetwork(olop, hlop, actualOutput[shuf[i]], inp[shuf[i]]);
			
		}
	}
	cout << "The training set accuracy is: ";
	getAcc(inp,actualOutput,inpn);
	cout << "Testing set 1 accuracy is: ";
	getAcc(testinp1, testout1, 36);
	cout << "Testing set 2 accuracy is: ";
	getAcc(testinp2, testout2, 40);
}

int main ()
{	
    	string line,line1;
	stringstream ss,ss1;
    	double inp[inpn][IN+1], target[inpn][OLN], olop[OLN], hlop[HLN+1];
	//load data in matrix
	ifstream myfile("trainingInpF.txt");
	ifstream file("trainingOutF.txt");
	ss << myfile.rdbuf();
	
	for(int i=0; i<inpn; i++)
	{
		for(int j=0; j<=IN; j++)
		{
			ss >> inp[i][j];
			inp[i][j]/=255;
		}
	}
	ss1 << file.rdbuf();
	for(int i=0; i<inpn; i++)
	{
		for(int j=0; j<OLN; j++)
		{
			ss1 >> target[i][j];
		}
	}
	myfile.close();
	file.close();
	//load testing data 1
	myfile.open("testInpF1.txt");
	file.open("testOutF1.txt");
	ss << myfile.rdbuf();
	
	for(int i=0; i<36; i++)
	{
		for(int j=0; j<=IN; j++)
		{
			ss >> testinp1[i][j];
			testinp1[i][j]/=255;
		}
	}
	ss1 << file.rdbuf();
	for(int i=0; i<36; i++)
	{
		for(int j=0; j<OLN; j++)
		{
			ss1 >> testout1[i][j];
		}
	}
	myfile.close();
	file.close();
	//load testing data 2
	myfile.open("testInpF2.txt");
	file.open("testOutF2.txt");
	ss << myfile.rdbuf();
	
	for(int i=0; i<40; i++)
	{
		for(int j=0; j<=IN; j++)
		{
			ss >> testinp2[i][j];
			testinp2[i][j]/=255;
		}
	}
	ss1 << file.rdbuf();
	for(int i=0; i<40; i++)
	{
		for(int j=0; j<OLN; j++)
		{
			ss1 >> testout2[i][j];
		}
	}
	myfile.close();
	file.close();
	//ANN call	
	stochasticGradientDescent(inp,target);


	return 0;
}
