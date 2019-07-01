#include <iostream>

using namespace std;

extern int getlinenum(string fileName, char separator);
extern void train(string fileName, char separator, int lines, int _grid_x, int _grid_y);
extern void predict(string fileName, char separator, int lines, int _grid_x, int _grid_y);

void main(){
	int trainline = getlinenum("P:\\train.txt", ';');
	int predictline = getlinenum("P:\\predict.txt", ';');
	//int Rin = 1, Rex = 4;
	//float threshold = 5.0;
	//bool adaption = false;
	int _grid_x = 8, _grid_y = 8;
	train("P:\\train.txt", ';', trainline, _grid_x, _grid_y);
	predict("P:\\predict.txt", ';', predictline, _grid_x, _grid_y);
	system("pause");
}