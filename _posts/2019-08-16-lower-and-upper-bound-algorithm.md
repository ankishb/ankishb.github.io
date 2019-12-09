---
layout: post
title: "Lower and Upper Bound Algorithm"
date: 2019-08-16
tags: Algorithm c++
---

This is a tricky question, it seems pretty straight-forward. But it played with me a lot. That's why, i wrote this post. Let's dive into this.

### Problem Statement: 
Find the lower bound and upper bound of given element in a given array.  
`Lower Bound: arr[i] <= element`  
`Upper Bound: arr[i] >= element`

>Note: Our objective is to solve this in `O(LogN)`

For example:
Given: An array `[0 2 5 8]`, find bounds for each one of this array `[-1 0 1 2 3 5 6]`  
Lower Bound: `[-1,0,0,1,2,3,4]`   
Upper Bound: `[0,0,1,1,2,3,4]`   


### Lower Bound:
1. If element is smaller than target, we need to shift its left bound at an index, one more than current one. **Now you may think that this isn't correct, because it will take our answer to next element. But our last statement will rescue us**
2. If element is greater than or equal to target, we need to tight its right bound at that index
3. return low or low-1, depend on ist step condition.

```c++
int lower_bound(int arr[], int n, int target){
    int lo = 0, hi = n-1, mid;
    while(lo < hi){
        mid = lo + (hi - lo)/2;
        if(arr[mid] < target) lo = mid+1;
        else hi = mid;
    }
    return (arr[lo] <= target) ? lo : lo-1;
}
```


### Upper Bound:
1. If element is greater or equal to target, we tight its right bound at that point.
2. If element is lesser than, we need to shift its left bound at an index, one more than current. (As element is smaller and we need upper bound of that element)
3. return mid

```c++
int upper_bound(int arr[], int n, int target){
    int lo = 0, hi = n-1, mid;
    while(lo < hi){
        mid = lo + (hi - lo)/2;
        if(arr[mid] >= target) hi = mid;
        else lo = mid+1;
    }
    return (arr[lo] >= target) ? lo : lo+1;
}
```

**Note**: If we want upper bound function as in c++ doc, than acc to their statement, arr[i] < target, so we can just change the `return` statement as `return (arr[lo] > target) ? lo : lo+1;`


Note: return statement also handle edge cases such as if element doesn't exit in given array.



```c++
Sample iterative solution in C++:

int binary_search(vector<int> A, int key){
    // considering 1 based index
    int low , high , mid ;
    low = 1 ;
    high = A.size() ;
    while(low <= high){
        mid = (low + high)/2 ;
        if(A[mid] == key){
            return mid ; // a match is found
        }else if(A[mid] < key){ // if middle element is less than the key 
            low = mid + 1 ;     // key will be right part if it exists
        }else{
            high = mid - 1 ;    // otherwise key will be in left part if it exists
        }
    }
    return -1 ; // indicating no such key exists 
}

Common Variations of BinarySearch:

    LowerBound: find an element 

in a sorted array of intergers such that , where key
is given.
UpperBound: find an element
in a sorted array of intergers such that , where key

    is given.

UpperBound Implementation in C++:


int UpperBound(vector<int> A, int K){
    int low , high , mid ;
    low = 1 ;
    high = A.size() ;
    while(low <= high){
        mid = ( low + high ) / 2 ; // finding middle element 
        if(A[mid] > K && ( mid == 1 || A[mid-1] <= K )) // checking conditions for upperbound
            return mid ;
        else if(A[mid] > K) // answer should be in left part 
            high = mid - 1 ;
        else                // answer should in right part if it exists
            low = mid + 1 ;
    }
    return mid ; // this will execute when there is no element in the given array which > K
}

LowerBound Implementation in C++:

int LowerBound(vector<int> A, int K){
    int low , high , mid ;
    low = 1 ;
    high = A.size() ;
    while(low <= high){
        mid = ( low + high ) / 2 ; // finding middle element 
        if(A[mid] >= K && ( mid == 1 || A[mid-1] < K )) // checking conditions for lowerbound
            return mid ;
        else if(A[mid] >= K) // answer should be in left part 
            high = mid - 1 ;
        else                // answer should in right part if it exists
            low = mid + 1 ;
    }
    return mid ; // this will execute when there is no element in the given array which >= K
}
```






```c++
int lowerBound(int *a,int n,int key){
    int s =0,e=n-1;
    int ans = -1;

    while(s<=e){
        int mid = (s+e)/2;

        if(a[mid]==key){
            ans = mid;
            e = mid - 1;
        }
        else if(a[mid]>key){
            e = mid - 1;
        }
        else{
            s = mid + 1;
        }
    }

    return ans;
}

int upperBound(int *a,int n,int key){
    int s =0,e=n-1;
    int ans = -1;

    while(s<=e){
        int mid = (s+e)/2;

        if(a[mid]==key){
            ans = mid;
            s = mid+1;
        }
        else if(a[mid]>key){
            e = mid - 1;
        }
        else{
            s = mid + 1;
        }
    }

    return ans;
}
```