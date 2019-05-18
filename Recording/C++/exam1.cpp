// exam1.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <random>
#include <time.h>
#include <iomanip>
#include <stdlib.h>

using namespace std;

/*
int _tmain(int argc, _TCHAR* argv[])
{
	return 0;
}*/


/** 请完成下面这个函数，实现题目要求的功能 **/
/** 当然，你也可以不按照这个模板来作答，完全按照自己的想法来 ^-^  **/

double leartCurve(double mu1, double sigma1, double mu2, double sigma2) {
	default_random_engine e;
	normal_distribution<double> n1(mu1, sigma1);
	normal_distribution<double> n2(mu2, sigma2);
	double a = 0;
	double b = 0;
	double n = 0;
	double in(0);
	double mid(0);
	for (int i = 0; i<100000; i++)
	{
	//	if (i % 100 == 1)
		//	cout << i << endl;
		a = n1(e);
		b = n2(e);
		double temp = a*a + b*b - 1.0;
		double f_val = temp*temp - a*a*b*b;
		if (f_val<0)
			mid += 1;
		if (abs(a) < 1 && abs(b)<1 && f_val>0)
			in += 1;
	}
	return (in+mid) / 100000.0;

}

int main() {
	double res;

	double _mu1;
	cin >> _mu1;
	cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	double _sigma1;
	cin >> _sigma1;
	cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	double _mu2;
	cin >> _mu2;
	cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	double _sigma2;
	cin >> _sigma2;
	cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');


	res = leartCurve(_mu1, _sigma1, _mu2, _sigma2);
	cout <<fixed<<setprecision(2)<< res << endl;
	printf("%.1f\n", res);
	system("pause");
	return 0;

}