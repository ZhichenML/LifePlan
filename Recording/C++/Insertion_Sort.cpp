// SORT.cpp : 定义控制台应用程序的入口点。
// Insertion sort is to iteratively insert an element into a sorted array.
// The time complexity is O(n^2), space complexity is O(1), because the target element need to be stored outside the array, such that 
// the position is available for larger elements (assuming non-decreasing order).
// The most efficient case is when the array is already sorted, becase the "while" condition is not met.
// The worst case is when the array is reversely ordered, becase the inner "while" condition is always met.


#include "stdafx.h"
#include <iostream>
#include <random>
#include <time.h>
#include <iomanip>
using namespace std;

class ARRAY{
public:
	double * a;
	size_t length;
	ARRAY(size_t le):length(le)
	{
		size_t i;
		a = new double[length];
		for(i = 0; i < length; i++)
			a[i] = 0;
	};
};

void init_rand(ARRAY array, int low, int high)
{
	size_t i(0);
	default_random_engine e;
	srand((unsigned)time(NULL));
	for (; i < array.length; i++)
		array.a[i] = e()%(high-low+1)+low;
}

void PrintArray(ARRAY array)
{
	size_t i;
	for (i = 0; i < array.length; i++)
		cout << setprecision(4) << array.a[i] << endl;
}

void Insertion_Sort(ARRAY array)
{
	size_t i, j;
	double key;
	for (j = 1; j < array.length ; j++)
	{
		key = array.a[j];
		i = j - 1;
		while (i >= 0 && array.a[i] > key)
		{
			array.a[i + 1] = array.a[i];
			i = i - 1;
		}
		array.a[i + 1] = key;
	}
}


int main(int argc, _TCHAR* argv[])
{
	ARRAY array(10);
	init_rand(array, 0, 1000);
	PrintArray(array);
	Insertion_Sort(array);
	cout << "===========================================" << endl;
	PrintArray(array);
	system("pause");



	return 0;
}

