---
layout: post
title: "STL-container-summary-with-time-complexity"
date: 2019-06-12
tags: STL-container time-complexity c++
---

It is a brief summary and syntax of STL containers and algorithms.

## Containers
- vectors
- lists
- deques
- stacks
- queues
- priority queues
- sets and multisets
- maps and multimaps

---
## Vector 

#### Constructors

Syntax | Detail | Time-Comp
--- | --- | ---
vector<T> v | 	Make an empty vector. |	O(1)
vector<T> v(n) | 	Make a vector with N elements. |	O(n)
vector<T> v(n, value) | 	Make a vector with N elements, initialized to value. |	O(n)
vector<T> v(begin, end) | 	Make a vector and copy the elements from begin to end. |	O(n)

#### Accessors

Syntax | Detail | Time-Comp
--- | --- | ---
v[i] | 	Return (or set) the I'th element. |	O(1)
v.at(i) | 	Return (or set) the I'th element, with bounds checking. |	O(1)
v.size() | 	Return current number of elements. |	O(1)
v.empty() | 	Return true if vector is empty. |	O(1)
v.begin() | 	Return random access iterator to start. |	O(1)
v.end() | 	Return random access iterator to end. |	O(1)
v.front() | 	Return the first element. |	O(1)
v.back() | 	Return the last element. |	O(1)
v.capacity() | 	Return maximum number of elements. |	O(1)

#### Modifiers

Syntax | Detail | Time-Comp
--- | --- | ---
v.push_back(value) | 	Add value to end. |	O(1) (amortized)
v.insert(iterator, 
value) | 	Insert value at the position indexed by iterator. |	O(n)
v.pop_back() | 	Remove value from end. |	O(1)
v.erase(iterator) | 	Erase value indexed by iterator. |	O(n)
v.erase(begin, end) | 	Erase the elements from begin to end. |	O(n)

#### Advances

Syntax | Detail | 
--- | --- | ---
copy(oldVect + first, oldVect + last, newVect) | copy value from range first to last in newVect


  int myints[] = {10,20,30,30,20,10,10,20};
  std::vector<int> v(myints,myints+8);           // 10 20 30 30 20 10 10 20

  std::sort (v.begin(), v.end());                // 10 10 10 20 20 20 30 30

  std::vector<int>::iterator low,up;
  low=std::lower_bound (v.begin(), v.end(), 20); //          ^
  up= std::upper_bound (v.begin(), v.end(), 20); //                   ^

  std::cout << "lower_bound at position " << (low- v.begin()) << '\n';
  std::cout << "upper_bound at position " << (up - v.begin()) << '\n';



min_element(myints,myints+7)
max_element(myints,myints+7)



sort( vec.begin(), vec.end() );
vec.erase( unique( vec.begin(), vec.end() ), vec.end() );

Convert to set (using a constructor)

set<int> s( vec.begin(), vec.end() );
vec.assign( s.begin(), s.end() );



int myints[] = {10,20,20,20,30,30,20,20,10};           // 10 20 20 20 30 30 20 20 10
  std::vector<int> myvector (myints,myints+9);

  // using default comparison:
  std::vector<int>::iterator it;
  it = std::unique (myvector.begin(), myvector.end());   // 10 20 30 20 10 ?  ?  ?  ?
  
  myvector.resize( std::distance(myvector.begin(),it) ); // 10 20 30 20 10

  // using predicate comparison:
  std::unique (myvector.begin(), myvector.end(), myfunction);   // (no changes)

#### string function:
- `isdigit(ch)`
- `isalpha(ch)`
- `tolower(ch)`
- `toupper(ch)`
- `islower(ch)`
- `isupper(ch)`



---

## Deque 

`#include <deque>`

#### Constructors

Syntax | Detail | Time-Comp
--- | --- | ---
deque<T> d | 	Make an empty deque. |	O(1)
deque<T> d(n) | 	Make a deque with N elements. |	O(n)
deque<T> d(n, value) | 	Make a deque with N elements, initialized to value. |	O(n)
deque<T> d(begin, end) | 	Make a deque and copy the values from begin to end. |	O(n)

#### Accessors

Syntax | Detail | Time-Comp
--- | --- | ---
d[i] | 	Return (or set) the I'th element. |	O(1)
d.at(i) | 	Return (or set) the I'th element, with bounds checking. |	O(1)
d.size() | 	Return current number of elements. |	O(1)
d.empty() | 	Return true if deque is empty. |	O(1)
d.begin() | 	Return random access iterator to start. |	O(1)
d.end() | 	Return random access iterator to end. |	O(1)
d.front() | 	Return the first element. |	O(1)
d.back() | 	Return the last element. |	O(1)

#### Modifiers

Syntax | Detail | Time-Comp
--- | --- | ---
d.push_front(value) | 	Add value to front. |	O(1) (amortized)
d.push_back(value) | 	Add value to end. |	O(1) (amortized)
d.insert(iterator, value) | 	Insert value at the position indexed by iterator. |	O(n)
d.pop_front() | 	Remove value from front. |	O(1)
d.pop_back() | 	Remove value from end. |	O(1)
d.erase(iterator) | 	Erase value indexed by iterator. |	O(n)
d.erase(begin, end) | 	Erase the elements from begin to end. |	O(n)

---


# List 

`#include <list>`

#### Constructors

Syntax | Detail | Time-Comp
--- | --- | ---
list<T> l | 	Make an empty list. |	O(1)
list<T> l(begin, end) | 	Make a list and copy the values from begin to end. |	O(n)

#### Accessors

Syntax | Detail | Time-Comp
--- | --- | ---
l.size() | 	Return current number of elements. |	O(1)
l.empty() | 	Return true if list is empty. |	O(1)
l.begin() | 	Return bidirectional iterator to start. |	O(1)
l.end() | 	Return bidirectional iterator to end. |	O(1)
l.front() | 	Return the first element. |	O(1)
l.back() | 	Return the last element. |	O(1)

#### Modifiers

Syntax | Detail | Time-Comp
--- | --- | ---
l.push_front(value) | 	Add value to front. |	O(1)
l.push_back(value) | 	Add value to end. |	O(1)
l.insert(iterator, value) | 	Insert value after position indexed by iterator. |	O(1)
l.pop_front() | 	Remove value from front. |	O(1)
l.pop_back() | 	Remove value from end. |	O(1)
l.erase(iterator) | 	Erase value indexed by iterator. |	O(1)
l.erase(begin, end) | 	Erase the elements from begin to end. |	O(1)
l.remove(value) | 	Remove all occurrences of value. |	O(n)
l.remove_if(test) | 	Remove all element that satisfy test. |	O(n)
l.reverse() | 	Reverse the list. |	O(n)
l.sort() | 	Sort the list. |	O(n log n)
l.sort(comparison) | 	Sort with comparison function. |	O(n logn)
l.merge(l2) | 	Merge sorted lists. |	O(n)



---

# Stack

In the C++ STL, a stack is a container adaptor. That means there is no primitive stack data structure. Instead, you create a stack from another container, like a list, and the stack's basic operations will be implemented using the underlying container's operations.


`#include <stack>`

#### Constructors

Syntax | Detail | Time-Comp
--- | --- | ---
stack< container<T> > s | 	Make an empty stack. |	O(1)

#### Accessors

Syntax | Detail | Time-Comp
--- | --- | ---
s.top() | 	Return the top element. |	O(1)
s.size() | 	Return current number of elements. |	O(1)
s.empty() | 	Return true if stack is empty. |	O(1)

#### Modifiers

Syntax | Detail | Time-Comp
--- | --- | ---
s.push(value) | 	Push value on top. 	Same as push_back() for underlying container.
s.pop() | 	Pop value from top. |	O(1)

---



# Queue

In the C++ STL, a queue is a container adaptor. That means there is no primitive queue data structure. Instead, you create a queue from another container, like a list, and the queue's basic operations will be implemented using the underlying container's operations.

Don't confuse a queue with a deque.


`#include <queue>`

#### Constructors

Syntax | Detail | Time-Comp
--- | --- | ---
queue< container<T> > q | 	Make an empty queue. |	O(1)

#### Accessors

Syntax | Detail | Time-Comp
--- | --- | ---
q.front() | 	Return the front element. |	O(1)
q.back() | 	Return the rear element. |	O(1)
q.size() | 	Return current number of elements. |	O(1)
q.empty() | 	Return true if queue is empty. |	O(1)

#### Modifiers

Syntax | Detail | Time-Comp
--- | --- | ---
q.push(value) | 	Add value to end. 	Same for push_back() for underlying container.
q.pop() | 	Remove value from front. |	O(1)
Priority Queue

In the C++ STL, a priority queue is a container adaptor. That means there is no primitive priorty queue data structure. Instead, you create a priority queue from another container, like a deque, and the priority queue's basic operations will be implemented using the underlying container's operations.

Priority queues are neither first-in-first-out nor last-in-first-out. You push objects onto the priority queue. The top element is always the "biggest" of the elements currently in the priority queue. Biggest is determined by the comparison predicate you give the priority queue constructor.

    If that predicate is a "less than" type predicate, then biggest means largest.

    If it is a "greater than" type predicate, then biggest means smallest.



---


# priority queue

`#include <queue>` -- not a typo!

#### Constructors

Syntax | Detail | Time-Comp
--- | --- | ---
priority_queue<T, container<T>, comparison<T> > q | 	Make an empty priority queue using the given container to hold values, and comparison to compare values. container defaults to vector<T> and comparison defaults to less<T>. |	O(1)

#### Accessors

Syntax | Detail | Time-Comp
--- | --- | ---
q.top() | 	Return the "biggest" element. |	O(1)
q.size() | 	Return current number of elements. |	O(1)
q.empty() | 	Return true if priority queue is empty. |	O(1)

#### Modifiers

Syntax | Detail | Time-Comp
--- | --- | ---
q.push(value) | 	Add value to priority queue. |	O(log n)
q.pop() | 	Remove biggest value. |	O(log n)


---



# Set and Multiset

Sets store objects and automatically keep them sorted and quick to find. In a set, there is only one copy of each objects. multisets are declared and used the same as sets but allow duplicate elements.

Anything stored in a set has to have a comparison predicate. This will default to whatever operator<() has been defined for the item you're storing. Alternatively, you can specify a predicate to use when constructing the set.


`#include <set>`

#### Constructors

Syntax | Detail | Time-Comp
--- | --- | ---
set< type, compare > s | 	Make an empty set. compare should be a binary predicate for ordering the set. It's optional and will default to a function that uses operator<. |	O(1)
set< type, compare > s(begin, end) | 	Make a set and copy the values from begin to end. |	O(n log n)

#### Accessors

Syntax | Detail | Time-Comp
--- | --- | ---
s.find(key) | 	Return an iterator pointing to an occurrence of key in s, or s.end() if key is not in s. |	O(log n)
s.lower_bound(key) | 	Return an iterator pointing to the first occurrence of an item in s not less than key, or s.end() | if no such item is found. |	O(log n)
s.upper_bound(key) | 	Return an iterator pointing to the first occurrence of an item greater than key in s, or s.end() | if no such item is found. |	O(log n)
s.equal_range(key) | 	Returns pair<lower_bound(key), upper_bound(key)>. |	O(log n)
s.count(key) | 	Returns the number of items equal to key in s. |	O(log n)
s.size() | 	Return current number of elements. |	O(1)
s.empty() | 	Return true if set is empty. |	O(1)
s.begin() 	Return an iterator pointing to the first element. |	O(1)
s.end() 	Return an iterator pointing one past the last element. |	O(1)

#### Modifiers

Syntax | Detail | Time-Comp
--- | --- | ---
s.insert(iterator, key) | 	Inserts key into s. iterator is taken as a "hint" but key will go in the correct position no matter what. Returns an iterator pointing to where key went. |	O(log n)
s.insert(key) | 	Inserts key into s and returns a pair<iterator, bool>, where iterator is where key went and bool is true if key was actually inserted, i.e., was not already in the set. |	O(log n)





---



# Map and Multimap

Maps can be thought of as generalized vectors. They allow map[key] = value for any kind of key, not just integers. Maps are often called associative tables in other languages, and are incredibly useful. They're even useful when the keys are integers, if you have very sparse arrays, i.e., arrays where almost all elements are one value, usually 0.

Maps are implemented with balanced binary search trees, typically red-black trees. Thus, they provide logarithmic storage and retrieval times. Because they use search trees, maps need a comparison predicate to sort the keys. operator<() will be used by default if none is specified a construction time.

Maps store <key, value> pair's. That's what map iterators will return when dereferenced. To get the value pointed to by an iterator, you need to say

(*mapIter).second

Usually though you can just use map[key] to get the value directly.

    Warning: map[key] creates a dummy entry for key if one wasn't in the map before. Use map.find(key) if you don't want this to happen.

multimaps are like map except that they allow duplicate keys. map[key] is not defined for multimaps. Instead you use lower_bound() and upper_bound(), or equal_range(), to get the iterators for the beginning and end of the range of values stored for the key. To insert a new entry, use map.insert(pair<key_type, value_type>(key, value)).


`#include <map>`

#### Constructors

Syntax | Detail | Time-Comp
--- | --- | ---
map< key_type, value_type, key_compare > m | 	Make an empty map. key_compare should be a binary predicate for ordering the keys. It's optional and will default to a function that uses operator<. |	O(1)
map< key_type, value_type, key_compare > m(begin, end) | 	Make a map and copy the values from begin to end. |	O(n log n)

#### Accessors

Syntax | Detail | Time-Comp
--- | --- | ---
m[key] | 	Return the value stored for key. This adds a default value if key not in map. |	O(log n)
m.find(key) | 	Return an iterator pointing to a key-value pair, or m.end() if key is not in map. |	O(log n)
m.lower_bound(key) | 	Return an iterator pointing to the first pair containing key, or m.end() if key is not in map. |	O(log n)
m.upper_bound(key) | 	Return an iterator pointing one past the last pair containing key, or m.end() if key is not in map. |	O(log n)
m.equal_range(key) | 	Return a pair containing the lower and upper bounds for key. This may be more efficient than calling those functions separately. |	O(log n)
m.size() | 	Return current number of elements. |	O(1)
m.empty() | 	Return true if map is empty. |	O(1)
m.begin() | 	Return an iterator pointing to the first pair. |	O(1)
m.end() | 	Return an iterator pointing one past the last pair. |	O(1)

#### Modifiers

Syntax | Detail | Time-Comp
--- | --- | ---
m[key] = value | 	Store value under key in map. |	O(log n)
m.insert(pair) | 	Inserts the <key, value> pair into the map. Equivalent to the above operation. |	O(log n)

---


Ref: http://users.cs.northwestern.edu/~riesbeck/programming/c++/stl-summary.html
