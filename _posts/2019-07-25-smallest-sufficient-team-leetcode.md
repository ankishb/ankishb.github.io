---
layout: post
title: "smallest-sufficient-team-leetcode"
date: 2019-07-25
tags: leetcode dp bit-mask
---

## Problem Statement
In a project, you have a list of required skills req_skills, and a list of people.  The i-th person people[i] contains a list of skills that person has.

Consider a sufficient team: a set of people such that for every required skill in req_skills, there is at least one person in the team who has that skill.  We can represent these teams by the index of each person: for example, team = [0, 1, 3] represents the people with skills people[0], people[1], and people[3].

Return any sufficient team of the smallest possible size, represented by the index of each person.


Example 1:

Input: req_skills = ["java","nodejs","reactjs"], people = [["java"],["nodejs"],["nodejs","reactjs"]]
Output: [0,2]

Example 2:

Input: req_skills = ["algorithms","math","java","reactjs","csharp","aws"], people = [["algorithms","math","java"],["algorithms","math","reactjs"],["java","csharp","aws"],["reactjs","csharp"],["csharp","math"],["aws","java"]]
Output: [1,2]

 

Constraints:

    1 <= req_skills.length <= 16
    1 <= people.length <= 60
    1 <= people[i].length, req_skills[i].length, people[i][j].length <= 16


### Approach:
1. As length of vector is very small, and we need to check each subset, we can use bit masking to represnt the whole skill_set.
2. Now our task boils down to take OR of skills of each subset, if it equal to all 1 or target then we find the team.
3. Can we break down this problem into smaller parts, where we can optimize subprobolem. Yes, we guess right. We use **DP** to do that.

### Time Complexity: O(n^2)
### Space Complexity: O(n^2)
Note: Space complexity can be optimized to O(n), by storing only the last optimization step. For exp, first we check, if only 1 person can fill for all skill_set, then we check for 2 person, and so on.


```c++
#include <bits/stdc++.h>
using namespace std;

void smallestSufficientTeam(vector<string>& req_skills, vector<vector<string>>& people) {
        unordered_map<string, int> mapping;
        int target = 0;
        for(int i=0; i<req_skills.size(); i++){
            // target = (target<<1)|1;
            target += pow(2,i);
            mapping[req_skills[i]] = i;
        }
        cout<<target<<endl;
        int n = people.size();
        vector<int> skill_people(n,0);
        int temp;
        for(int i=0; i<n; i++){
            temp = 0;
            for(int j=0; j<people[i].size(); j++){
                temp += pow(2, mapping[people[i][j]]);
            }
            skill_people[i] = temp;
        }
        
        for(auto itr : skill_people) cout<<itr<<" ";
        cout<<endl;
        // return skill_people;
        cout<<(skill_people[1] | skill_people[2] | skill_people[3])<<endl;
        int ans;
        vector<vector<int>> dp(n, vector<int>(n,0));
        for(int k=0; k<n; k++){
            for(int i=0; i<n-k; i++){
                int j = i+k;
                if(i == j) dp[i][j] = skill_people[i];
                else{
                    int result = (dp[i+1][j] | dp[i][j-1]);
                    dp[i][j] = result;
                    if(result == target){
                        cout<<i<<"-----"<<j<<endl;
                        i = n;
                        k = n;
                    }
                }
            }
        }
        
        cout<<endl;
        for(auto itr1 : dp){
            for(auto itr2 : itr1){
                cout<<itr2<<" ";
            }
            cout<<endl;
        }
    }
    
int main()
{
    int test; cin>>test;
    while(test--){
        int n; cin>>n;
        vector<string> req_skills(n);
        for(int i=0; i<n; i++){
            cin>>req_skills[i];
        }
        int m; cin>>m;
        vector<vector<string>> people;
        for(int i=0; i<m; i++){
            int p; cin>>p;
            vector<string> temp(p);
            for(int j=0; j<p; j++){
                cin>>temp[j];
            }
            people.push_back(temp);
            temp.clear();
        }
        smallestSufficientTeam(req_skills, people);
        
    }
    return 0;
}
```


Input:
```
1
6 "algorithms" "math" "java" "reactjs" "csharp" "aws"
6
3 "algorithms" "math" "java"
2 "algorithms" "math" 
3 "java" "csharp" "aws"
2 "reactjs" "csharp"
2 "csharp" "math"
2 "aws" "java"
```

Output:
```
63
7 3 52 24 18 36 
63
1-----3

7 7 55 0 0 0 
0 3 55 63 0 0 
0 0 52 60 0 0 
0 0 0 24 26 0 
0 0 0 0 18 54 
0 0 0 0 0 36 
```


