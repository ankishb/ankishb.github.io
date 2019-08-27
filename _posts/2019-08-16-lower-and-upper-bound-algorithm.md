---
layout: post
title: "Lower and Upper Bound Algorithm"
date: 2019-08-16
tags: Algorithm c++
---

This is a tricky question, it seems pretty straight-forward. But it played it me a lot. Let's dive into this.

#### Problem Statement: 
Find the lower bound and upper bound of given element in a given array. 
**Lower Bound**: arr[i] <= element
**Upper Bound**: arr[i] >= element

Note: Our objective is to solve this in `O(NLogN)`

For example:
Given: Arr [0 2 5 8], find bounds for [-1 0 1 2 3 5 6]
Lower Bound: [-1,0,0,1,2,3,4] 
Upper Bound: [0,0,1,1,2,3,4] 

## Solution

### Lower Bound:
Logic:
    1. If element is smaller to target, we need to left bound at index, one more than current index. **Now you may think that this is not correct asit will take our answer to next element. But my last statement on return statement saves me**
    2. If element is greater than or equal to target, we need to right bound at that index
    - return low or low-1, by checking condition of ist step.

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
Logic:
    1. If element is greater or equal to target, we right bound it at that point.
    2. If element is lesser than, we need to left bound at index, one more than current. (As element is smaller and we need upper bound of that element)
    - return mid

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