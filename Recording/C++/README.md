# C-code
Basic algorithm implementation.

Insertion sort is to iteratively insert an element into a sorted array.
The time complexity is O(n^2), space complexity is O(1), because the target element need to be stored outside the array, such that 
the position is available for larger elements (assuming non-decreasing order).
The most efficient case is when the array is already sorted, becase the "while" condition is not met.
The worst case is when the array is reversely ordered, becase the inner "while" condition is always met.



The time complexity of merge sort is O(nlogn). Storage complexity is O(n). 
It is a stable sort algorithm, because when left<=righrt, left is added to the array, see Merge.
The idea is divide and conqure, which recursively break the array into subarray until one element is left.
Then combine them. T(n) = 2T(n/2)+O(n).
