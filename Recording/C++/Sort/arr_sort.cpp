#include "stdafx.h"
#include<iostream>
#include<time.h>
#include<random>

using namespace std;

template <typename T> class arr_sort{ // template ����
private:
	unsigned int length;
	T * arr;
public:

	arr_sort() //Ĭ�Ϲ��캯��
	{
		length = 100;
		arr = new T[length];
	}
	arr_sort(unsigned int len) : length(len) //���������캯��
	{
		arr = new T[length];
	}


	void  Rand() // �漴��ʼ��
	{
		unsigned int i = 0;
		default_random_engine e;
		srand((unsigned)time(NULL));
		while (i != length)
			arr[i++] = e();
	}

	void Rand(int low, int high) //ָ����Χ�漴��ʼ�������ڼ�������
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
		T *left = new T[nl];//���鶯̬���䣬ֻ����ָ��
		T *right = new T[nr];
		for (i = 0; i != nl; i++)
			left[i] = arr[p + i];
		for (i = 0; i != nr; i++)
			right[i] = arr[q + 1 + i];
		for (i = 0, j = 0, k = p; i != nl&&j != nr;) //�������������߿�������������ұ߿����ұ�����
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

	void merge_sort(unsigned int p, unsigned int r) // �鲢����
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
		for (unsigned int i = 0; i != length - 2; i++) //ǰn-1��Ԫ�ء���ΪֻҪ��n-1��СԪ���Ƶ�ǰ��
		{
			for (unsigned int j = length - 1; j != i; j--) // wrong, j-- ��СԪ�شӺ���ǰ��
			{
				if (arr[j] < arr[j - 1]) //wrong, j-1,ϰ����д��j+1
				{
					T temp = arr[j];
					arr[j] = arr[j - 1];
					arr[j - 1] = temp; //wrong, ϰ����д��j+1��
				}
			}
		}
	}
	
	int Random_generator(const unsigned int low, const unsigned int high) //����һ��ָ����Χ���������
	{
		default_random_engine e;
		return e() % (high - low + 1) + low;

	}

	void swap(const unsigned int i, const unsigned int j)//�����������±�ֱ���i��j������Ԫ��
	{
		T temp = arr[i];
		arr[i] = arr[j];
		arr[j] = temp;
	}

	void Randomize() // ��һ��������������е�
	{
		for (unsigned int i = 0; i != length; i++)
			swap(i, Random_generator(i, length - 1));
	}

	// ������
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
			swap(0, i); //wrong, �±��0��ʼ
			cout <<"arr["<<i<<"]=" <<arr[i] << endl;
			length--;
			Max_Heapify(0);
		}
		length = temp_length;
	}
		
	//��������
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

	//�������鳤��
	int ReadLength()
	{
		return length;
	}

	//��������
	void CountSort(int low, int high)
	{
		T *B = new T[length];
		int *C = new int[high - low + 1];
		for (unsigned int i = 0; i != high - low + 1; i++)
			C[i] = 0;
		for (unsigned int i = 0; i != length; i++)
			C[arr[i] - low]++;

		for (unsigned int i = 1; i != high - low + 1; i++) //wrong, ���±�1 ��ʼ
			C[i] = C[i - 1] + C[i];


		for (unsigned int i = length - 1; i != -1; i--)
		{
			B[C[arr[i] - low]-1] = arr[i]; //wrong, C��СԪ����1��B�±��0��ʼ��
			C[arr[i] - low]--;
		}

		for (unsigned int i = length - 1; i != -1; i--)
		{
			arr[i] = B[i];
		}
	}

	// ָ����Χ�������ĸ����������ѷ��ֵ�����������С��sqrt(n)��������
	int Prime_count(int num)
	{
		vector<int> prime;
		unsigned int count = 0;
		prime.push_back(2); //�����һ������
		for (int cur = 3; cur <= num; cur++)
		{
			bool flag = 1;
			for (int prime_index = 0; prime[prime_index]<=sqrt(cur); prime_index++) //�����Ѿ��ҵ�������
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
		return count; //������������ 
	}
	
	//�������㣬����ʱ�临�Ӷ�,ͨ����ָ���ֽ��2���ݡ��������ɵ����㶼���Խ��п��������㣬�����˷���
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

	//���ҵ�k��˳��ͳ����,O(N)
	//���������±��0��ʼ����i��˳��ͳ�����ڵ�i-1λ��
	T RandomSelect(int k, int low, int high)
	{
		if (low == high)
			return arr[low];
		int mid = partition(low, high);
		int q = mid - low + 1;
		if (q == k)
			return arr[mid]; //wrong, ����mid
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
	arr_int.QuickSort(0, arr_int.ReadLength() - 1); //wrong, ���ò�����Χ
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