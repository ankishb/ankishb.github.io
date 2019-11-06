

## `ACID` properties in DBMS:
To ensure the integrity of data during a transaction (A transaction is a unit of program that updates various data items, read more about it here), the database system maintains the following properties. These properties are widely known as ACID properties:
1. `Atomicity`: This property ensures that either all the operations of a transaction reflect in database or none. Let’s take an example of banking system to understand this: Suppose Account A has a balance of 400$ & B has 700$. Account A is transferring 100$ to Account B. This is a transaction that has two operations a) Debiting 100$ from A’s balance b) Creating 100$ to B’s balance. Let’s say first operation passed successfully while second failed, in this case A’s balance would be 300$ while B would be having 700$ instead of 800$. This is unacceptable in a banking system. Either the transaction should fail without executing any of the operation or it should process both the operations. The Atomicity property ensures that.
2. `Consistency`: To preserve the consistency of database, the execution of transaction should take place in isolation (that means no other transaction should run concurrently when there is a transaction already running). For example account A is having a balance of 400$ and it is transferring 100$ to account B & C both. So we have two transactions here. Let’s say these transactions run concurrently and both the transactions read 400$ balance, in that case the final balance of A would be 300$ instead of 200$. This is wrong. If the transaction were to run in isolation then the second transaction would have read the correct balance 300$ (before debiting 100$) once the first transaction went successful.
3. `Isolation`: For every pair of transactions, one transaction should start execution only when the other finished execution. I have already discussed the example of Isolation in the Consistency property above.
4. `Durability`: Once a transaction completes successfully, the changes it has made into the database should be permanent even if there is a system failure. The recovery-management component of database systems ensures the durability of transaction.



## `Normalization` in DBMS:
#### Problems Without Normalization
1. Insertion Anomaly
2. Updation Anomaly
3. Deletion Anomaly


#### `Normalization Rule` are divided into the following normal forms:
1. First Normal Form (1NF)
    1. It should only have single(atomic) valued attributes/columns.
    2. Values stored in a column should be of the same domain
    3. All the columns in a table should have unique names.
    4. And the order in which data is stored, does not matter.

2. Second Normal Form (2NF)
    1. It should be in the First Normal form.
    2. And, it should not have Partial Dependency.

3. Third Normal Form (3NF)
    1. It is in the Second Normal form.
    2. And, it doesn't have Transitive Dependency.
    **A transitive functional dependency is when changing a non-key column, might cause any of the other non-key columns to change**

4. Boyce and Codd Normal Form (BCNF)
Boyce and Codd Normal Form is a higher version of the Third Normal form. This form deals with certain type of anomaly that is not handled by 3NF. A 3NF table which does not have multiple overlapping candidate keys is said to be in BCNF. For a table to be in BCNF, following conditions must be satisfied:
    1. R must be in 3rd Normal Form
    2. and, for each functional dependency ( X → Y ), X should be a super Key.


5. Fourth Normal Form
    1. It is in the Boyce-Codd Normal Form.
    2. And, it doesn't have Multi-Valued Dependency.



## `Keys`
### Primary Key
A primary is a single column value used to identify a database record uniquely. 
It has following attributes
    1. A primary key cannot be NULL
    2. A primary key value must be unique
    3. The primary key values should rarely be changed
    4. The primary key must be given a value when a new record is inserted.

### What is Composite Key?
A composite key is a primary key composed of multiple columns used to identify a record uniquely 

Foreign Key references the primary key of another Table! It helps connect your Tables
    1. A foreign key can have a different name from its primary key
    2. It ensures rows in one table have corresponding rows in another
    3. Unlike the Primary key, they do not have to be unique. Most often they aren't
    4. Foreign keys can be null even though primary keys can not 

#### Why do you need a foreign key?
Suppose, a novice inserts a record in Table such as
- You will only be able to insert values into your foreign key that exist in the unique key in the parent table. This helps in `referential integrity`
- Foreign Key references the primary key of another Table! It helps connect your Tables

### Super Key in DBMS: 
A super key is a set of one or more attributes (columns), which can uniquely identify a row in a table.

### Candidate keys
Candidate keys are selected from the set of super keys, the only thing we take care while selecting candidate key is: It should not have any redundant attribute. That’s the reason they are also termed as minimal super key.