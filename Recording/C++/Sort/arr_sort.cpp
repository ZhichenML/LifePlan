#include "stdafx.h"
#include<iostream>
#include<time.h>
#include<random>

using namespace std;

template <typename T> class arr_sort{ // template 声明
private:
	unsigned int length;
	T * arr;
public:

	arr_sort() //默认构造函数
	{
		length = 100;
		arr = new T[length];
	}
	arr_sort(unsigned int len) : length(len) //带参数构造函数
	{
		arr = new T[length];
	}


	void  Rand() // 随即初始化
	{
		unsigned int i = 0;
		default_random_engine e;
		srand((unsigned)time(NULL));
		while (i != length)
			arr[i++] = e();
	}

	void Rand(int low, int high) //指定范围随即初始化，用于计数排序
	{
		unsigned int i(0);
		default_random_engine e;
		srand((unsigned)time(NULL));
		while (i != length)
			arr[i++] = e() % (high - low + 1) + low;
	}

	void print()
	{
		int i = 0;
		while (i != length)
		{
			cout << arr[i] << " ";
			i++;
		}
		cout << endl;
	}

	void insertion_sort()
	{
		int i, j;
		T key;
		for (j = 2; j != length; j++)
		{
			key = arr[j];
			i = j - 1;
			while (i != -1 && key<arr[i])
			{
				arr[i + 1] = arr[i];
				i = i - 1;
			}
			arr[i + 1] = key;
		}

	}

	void merge(unsigned int p, unsigned int q, unsigned int r)
	{
		unsigned int i, j, k;
		unsigned int nl = q - p + 1;
		unsigned int nr = r - q;
		T *left = new T[nl];//数组动态分配，只能用指针
		T *right = new T[nr];
		for (i = 0; i != nl; i++)
			left[i] = arr[p + i];
		for (i = 0; i != nr; i++)
			right[i] = arr[q + 1 + i];
		for (i = 0, j = 0, k = p; i != nl&&j != nr;) //自增分情况，左边拷贝左边自增，右边拷贝右边自增
		{
			if (left[i] <= right[j])
				arr[k++] = left[i++];
			else
				arr[k++] = right[j++];
		}
		if (i != nl)
		{
			while (i!=nl)
				arr[k++] = left[i++];
		}
		else
		{
			while (j != nr)
				arr[k++] = right[j++];
		}

	}

	void merge_sort(unsigned int p, unsigned int r) // 归并排序
	{
		int q;
		if (p < r)
		{
			q = (p + r) / 2;
			merge_sort(p,q);
			merge_sort(q + 1, r);
			merge(p,q,r);
		}

	}

	void bubble_sort()
	{
		for (unsigned int i = 0; i != length - 2; i++) //前n-1个元素。因为只要把n-1个小元素推到前面
		{
			for (unsigned int j = length - 1; j != i; j--) // wrong, j-- 把小元素从后向前推
			{
				if (arr[j] < arr[j - 1]) //wrong, j-1,习惯性写成j+1
				{
					T temp = arr[j];
					arr[j] = arr[j - 1];
					arr[j - 1] = temp; //wrong, 习惯性写成j+1；
				}
			}
		}
	}
	
	int Random_generator(const unsigned int low, const unsigned int high) //返回一个指定范围的随机整数
	{
		default_random_engine e;
		return e() % (high - low + 1) + low;

	}

	void swap(const unsigned int i, const unsigned int j)//交换数组中下标分别是i和j的两个元素
	{
		T temp = arr[i];
		arr[i] = arr[j];
		arr[j] = temp;
	}

	void Randomize() // 将一个数组变成随机排列的
	{
		for (unsigned int i = 0; i != length; i++)
			swap(i, Random_generator(i, length - 1));
	}

	// 堆排序
	unsigned int Parent(unsigned int i)
	{
		return (i - 1) / 2;
	}

	unsigned int Left(unsigned int i)
	{
		return 2 * i + 1;
	}

	unsigned int Right(unsigned int i)
	{
		return 2 * i + 2;
	}
	void Max_Heapify(unsigned int i)
	{
		unsigned int largest = i;
		unsigned int l, r;
		l = Left(i);
		r = Right(i);
		if (l<length&&arr[l]>arr[largest])
			largest = l;
		if (r<length&&arr[r]>arr[largest])
			largest = r;
		if (largest != i)
		{
			swap(i, largest);
			Max_Heapify(largest);
		}
	}

	void Build_Max_Heap()
	{
		for (unsigned int i = (length - 1) / 2; i != -1; i--)
			Max_Heapify(i);

	}
	void HeapSort()
	{
		unsigned int temp_length = length;
		Build_Max_Heap();
		for (unsigned int i = length - 1; i >= 1; i--)
		{
			swap(0, i); //wrong, 下标从0开始
			cout <<"arr["<<i<<"]=" <<arr[i] << endl;
			length--;
			Max_Heapify(0);
		}
		length = temp_length;
	}
		
	//快速排序
	int partition(unsigned int p, unsigned int q)
	{
		T x = arr[q];
		int i(p-1);
		for (int j = p; j <= q; j++)
		{
			if (arr[j] <= x)
			{
				i++;
				swap(i, j);
			}
		}
		//swap(i + 1, q);
		return i;// +1;
	}

	void QuickSort(int p, int r)
	{
		if (p < r)
		{
			int q = partition(p, r);
			QuickSort(p, q - 1);
			QuickSort(q + 1, r);
		}
		
	}

	//返回数组长度
	int ReadLength()
	{
		return length;
	}

	//计数排序
	void CountSort(int low, int high)
	{
		T *B = new T[length];
		int *C = new int[high - low + 1];
		for (unsigned int i = 0; i != high - low + 1; i++)
			C[i] = 0;
		for (unsigned int i = 0; i != length; i++)
			C[arr[i] - low]++;

		for (unsigned int i = 1; i != high - low + 1; i++) //wrong, 从下标1 开始
			C[i] = C[i - 1] + C[i];


		for (unsigned int i = length - 1; i != -1; i--)
		{
			B[C[arr[i] - low]-1] = arr[i]; //wrong, C最小元素是1，B下标从0开始。
			C[arr[i] - low]--;
		}

		for (unsigned int i = length - 1; i != -1; i--)
		{
			arr[i] = B[i];
		}
	}

	// 指定范围内质数的个数。保存已发现的素数，测试小于sqrt(n)的素数。
	int Prime_count(int num)
	{
		vector<int> prime;
		unsigned int count = 0;
		prime.push_back(2); //放入第一个素数
		for (int cur = 3; cur <= num; cur++)
		{
			bool flag = 1;
			for (int prime_index = 0; prime[prime_index]<=sqrt(cur); prime_index++) //遍历已经找到的素数
			{
				if (cur%prime[prime_index] == 0)
				{
					flag = false;
					break;
				}
			}
			if (flag)
			{
				prime.push_back(cur);
				count++;
			}
		}
		for (int i = 0; i != prime.size(); i++)
			cout << i << ": " << prime[i]<<endl;
		return count; //返回素数个数 
	}
	
	//求幂运算，对数时间复杂度,通过把指数分解成2的幂。满足结合律的运算都可以进行快速幂运算，如矩阵乘法。
	int qPower(int a, int n)
	{
		int ret = 1;
		for (; n; n = n >> 1, a = a*a)
		{
			if (n % 2)
				ret = ret * a;
		}
		return ret;
	}

	//查找第k个顺序统计量,O(N)
	//由于数组下标从0开始，第i个顺序统计量在第i-1位。
	T RandomSelect(int k, int low, int high)
	{
		if (low == high)
			return arr[low];
		int mid = partition(low, high);
		int q = mid - low + 1;
		if (q == k)
			return arr[mid]; //wrong, 返回mid
		if (q > k)
			return RandomSelect( k, low, mid - 1);
		else
			return RandomSelect(k-q, mid + 1, high);
	}

};


template <typename T> class Sim_Stack{
private: 
	unsigned int length;
	vector<T> arr;
	size_t top;
public:
	Sim_Stack() :length(10){
		arr.resize(length);
		top = -1;
	}
	Sim_Stack(unsigned int len) :length(len)
	{
		arr.resize(length);
		top = -1;
	}
	T Push(T num)
	{
		if (top == length - 1)
		{
			cout << "Stack Overflow!" << endl;
			return num;
		}
		top++;
		arr[top] = num;
		return arr[top];
	}
	T Pop()
	{
		if (top == -1)
		{
			cout << "Stack Empty!" << endl;
			return T(0);
		}
		top--;
		return arr[top+1];
	}
	
	void Print()
	{
		if (top == -1) { cout << "Empty." << endl; return; }
		for (int i = 0; i != top + 1; i++)
			cout << i << ": " << arr[i] << endl;
	}

};






int main()
{
	unsigned int length = 13;
	arr_sort<int> arr_int(length);
	arr_int.Rand();
	arr_int.print();
	cout << "=========================================================" << endl;
	//	arr_int.insertion_sort();
	//	arr_int.merge_sort(0,length-1);
	cout << "Bubble Sort" << endl;
	arr_int.bubble_sort();
	arr_int.print();

	cout << "=========================================================" << endl;
	cout << "Randomized array." << endl;
	arr_int.Randomize();
	arr_int.print();
	cout << "HeapSort()." << endl;
	arr_int.HeapSort();
	arr_int.print();

	cout << "=========================================================" << endl;
	cout << "Quick Sort." << endl;
	arr_int.Randomize();
	arr_int.print();
	arr_int.QuickSort(0, arr_int.ReadLength() - 1); //wrong, 调用参数范围
	arr_int.print();

	cout << "=========================================================" << endl;
	unsigned int length1 = 10;
	arr_sort<int> arr_int1(length1);
	arr_int1.Rand(5, 10);
	arr_int1.print();
	arr_int1.CountSort(0, 10);
	arr_int1.print();

	arr_int.Prime_count(100);
	cout << "Quick power:" << endl
		<< arr_int.qPower(3, 10) << endl;

	int k;
	//arr_int.Randomize();
	arr_int.print();
	cout << endl << "Input the k for ordered number:";
	while (cin >> k)
	{
	cout << arr_int.RandomSelect(k - 1, 0, length - 1) << endl;
	}

	Sim_Stack<int> stack_int;
	for (int i = 0; i != 10; i++)
		stack_int.Push(i);
	stack_int.Print();
	for (int i = 0; i != 11; i++)
		stack_int.Pop();
	stack_int.Print();
	

	system("pause");


	/*	int arr[100] = {0};
	print(arr,100);
	cout<<"=========================================================="<<endl;
	Rand(arr,100);
	print(arr,100);
	cout<<"=========================================================="<<endl;
	insertion_sort(arr,100);
	print(arr,100);
	cout<<"=========================================================="<<endl;
	*/
	return 0;
}