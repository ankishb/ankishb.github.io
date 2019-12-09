---
layout: post
title: "largest-rectangle-in-histogram"
date: 2019-08-01
tags: leetcode stack c++
---

## Problem Statement
Given n non-negative integers representing the histogram's bar height where the width of each bar is 1, find the area of largest rectangle in the histogram.


Example 1:

Input: [3,4,5,3,4]
Output: 15

Example 2:

Input: [2,1,5,6,2,3]
Output: 10

Example 3:

Input: [2,1,5,6,2,3]
Output: 10


     
     |22|
  |22|22|  |22|
22|22|22|22|22|
22|22|22|22|22|
22|22|22|22|22|
----------------

### Approach:
1. In the above numbered-figure, notice that what and longest rectangle can be formed. 
Think this way, we go back on the position where element is less than the current value. And from there, we see the area of rectangle on the way right till current index. 
- For example: in [3,4,5], when we are 5, we go back its prev smaller element that is 4, then we see the length of rectangle in forward direction that is `5`.
- Another example: [3,4,9,5], when we are 5, we go back its prev smaller element that is 4, then we see the length of rectangle forward, that is `10`.
- Another case: [3,4,1,5], Now, when we are at 5, we go back its prev smaller element that is 1, then we see the length of rectangle forward, that is `5`.


Approach:
1. If element is in increasing order, add that element on stack. `Note: we add index of element, random than element, to keep track its position in the array.`
2. Else 
    1. we pop the element from stack, till we get smaller element in stack than current position. 
    2. We also calculate the area of rectangle using above logic:
    ```
    if(s.empty()) curArea = h[lastTop]*i;
    else curArea = h[lastTop]*(i-s.top()-1);
    ```



```c++
long largestRectangle(vector<int> h) {
    int n = h.size();
    stack<long> s;
    long maxArea = INT_MIN, curArea, idx;
    int i;
    for(i=0; i<n; i++){
        if(s.empty()) s.push(i);
        else if(h[i] >= h[s.top()]) s.push(i);
        else{
            while(!s.empty() && h[s.top()] > h[i]){
                idx = s.top(); s.pop();
                if(s.empty()) curArea = h[idx]*i;
                else curArea = h[idx]*(i-s.top()-1);
                maxArea = max(maxArea, curArea);
            }
            s.push(i);
        }
    }
    while(!s.empty()){
        idx = s.top(); s.pop();
        if(s.empty()) curArea = h[idx]*i;
        else curArea = h[idx]*(i-s.top()-1);
        maxArea = max(maxArea, curArea);
    }
    return maxArea;
}
```

