// MergeSort.cpp : 定义控制台应用程序的入口点。
// The time complexity of merge sort is O(nlogn). Storage complexity is O(n). 
// It is a stable sort algorithm, because when left<=righrt, left is added to the array, see Merge.
// The idea is divide and conqure, which recursively break the array into subarray until one element is left.
// Then combine them. T(n) = 2T(n/2)+O(n).

#include "stdafx.h"
#include <iostream>
#include <iomanip>
#include <time.h>
#include <random>

using namespace std;

class ARRAY{
public:
	size_t length;
	double* a;
	ARRAY(size_t len) :length(len){
		a = new double[length];
	}
};

void init_rand(ARRAY array, int low, int high)
{
	size_t i(0);
	default_random_engine e;
	srand((unsigned)time(NULL));
	while (i < array.length)
	{
		array.a[i] = e() % (high - low + 1) + low;
		i++;
	}
}


void PrintArray(ARRAY array)
{
	size_t i(0);
	while (i < array.length)
	{
		cout << array.a[i] << endl;
		i++;
	}
}

void Merge(ARRAY array, size_t p, size_t q, size_t r)
{
	size_t len1 = q - p + 1;
	size_t len2 = r - q;
	double *arr1 = new double[len1];
	double *arr2 = new double[len2];
	size_t i,j,k;
	for (i = p; i <= q; i++)
		arr1[i-p] = array.a[i];
	for (i = q + 1; i <= r; i++)
		arr2[i - q - 1] = array.a[i];
	for (i = 0, j = 0, k = p; i < len1&&j < len2;)
	{
		if (arr1[i] <= arr2[j])
			array.a[k++] = arr1[i++];
		else
			array.a[k++] = arr2[j++];
	}
	if (i == len1) //be careful about the difference between == and =
	{
		while (j < len2)
			array.a[k++] = arr2[j++];

	}
	else
	{
		while (i < len1)
			array.a[k++] = arr1[i++];
	}
}


void Merge_Sort(ARRAY array, size_t p, size_t r)
{
	if (p < r) // remember the termination condition of merge sort.
	{
		size_t q = (p + r) / 2;
		Merge_Sort(array, p, q);
		
		Merge_Sort(array, q + 1, r);
		Merge(array, p, q, r);
	}
	
}

int main(int argc, _TCHAR* argv[])
{
	ARRAY array(10);
	init_rand(array, 0, 1000);
	PrintArray(array);
	Merge_Sort(array,0,array.length-1);
	cout << "===================================" << endl;
	PrintArray(array);
	system("pause");
	return 0;
}

