
## SQL (Structured Query Language) 
- SQL is a specifically designed language for defining, accessing and manipulating the data. It is non-procedural, where the essential elements and its outcome is first specified without describing how results are computed.

There are two different groups of commands included in the SQL – DDL and DML. DDL expands to `Data Definition Language` used for defining and modifying several data structures. While DML (`Data Manipulation Language`) is intended to access and manipulate the data stored within the data structures previously defined by DDL.

---

## `ACID` properties in DBMS:
Following properties plays an important role, `to ensure the accuracy, completeness and data integrity, during a transaction` (A transaction is a very small unit of a program and it may contain several lowlevel tasks)
1. `Atomicity`: 
    - a transaction must be treated as an `atomic unit`, i.e. `either all of its operations are executed or none`
    - There must be `no` state in a database where a transaction is left `partially completed`. 
    - For exp: lets say i tranfer money to some friend. But if that money is cut from my account, but my friend does not receive it, this is transaction failure. So atomicity ensure this as atomic event.
2. `Consistency`: 
    - To preserve the consistency of database, the execution of transaction should take place in isolation.
    - it means that no other transaction should run concurrently when there is a transaction already running. 
    - In other words, `No transaction should have any adverse effect on the data residing in the database`
    - For example account A is having a balance of 400$ and it is transferring 100$ to account B & C both. So we have two transactions here. Let’s say these transactions run concurrently and both the transactions read 400$ balance, in that case the final balance of A would be 300$ instead of 200$. This is wrong. If the transaction were to run in isolation then the second transaction would have read the correct balance 300$ (before debiting 100$) once the first transaction went successful.
3. `Isolation`: 
    - For every pair of transactions, one transaction should start execution only when the other finished execution.
    - We have discussed the example of Isolation in the Consistency property above.
4. `Durability`: 
    - Once a transaction completes successfully, the changes it has made into the database should be permanent even if there is a system failure. 
    - The recovery-management component of database systems ensures the durability of transaction.
    - If a transaction commits but the system fails before the data could be written on to the disk, then that data will be updated once the system springs back into action.

---

## Pivoting:
```sql
<!-- simple -->
SELECT teams.conference AS conference,
       players.year,
       COUNT(1) AS players
  FROM benn.college_football_players players
  JOIN benn.college_football_teams teams
    ON teams.school_name = players.school_name
 GROUP BY 1,2
 ORDER BY 1,2

<!-- pivoting -->
SELECT conference,
       SUM(players) AS total_players,
       SUM(CASE WHEN year = 'FR' THEN players ELSE NULL END) AS fr,
       SUM(CASE WHEN year = 'SO' THEN players ELSE NULL END) AS so,
       SUM(CASE WHEN year = 'JR' THEN players ELSE NULL END) AS jr,
       SUM(CASE WHEN year = 'SR' THEN players ELSE NULL END) AS sr
  FROM (
        SELECT teams.conference AS conference,
               players.year,
               COUNT(1) AS players
          FROM benn.college_football_players players
          JOIN benn.college_football_teams teams
            ON teams.school_name = players.school_name
         GROUP BY 1,2
       ) sub
 GROUP BY 1
 ORDER BY 2 DESC
```

---

## Casting:

```sql
SELECT CAST(funding_total_usd AS varchar) AS funding_total_usd_string,
       founded_at_clean::varchar AS founded_at_string
  FROM tutorial.crunchbase_companies_clean_date
```


## Data Types:[Ref](mode-analytics)

Datatype | Keyword | explanation
--- | --- | ---
- String | `VARCHAR(1024)` | `Any characters`
- Date/Time | `TIMESTAMP`  | `Stores year, month, day, hour, minute and second values as YYYY-MM-DD hh:mm:ss`
- Number | DOUBLE PRECISION  |  Numerical, with up to 17 significant digits decimal precision.
- Boolean | BOOLEAN | Only TRUE or FALSE values.

## Timestemp functionality:
```sql
col_a - col_b::timestamp AS new_col
```

- Interval usuage: as '10 seconds' or '5 months'
```sql
col_a::timestamp AS + INTERVAL '1 week' new_col
```

- NOW: current time
```sql
NOW() - col_b::timestamp AS new_col
```

---

## cases:
```sql
SELECT 
    player_name,
    weight,
    CASE 
        WHEN weight > 250 THEN 'over 250'
        WHEN weight > 200 THEN '201-250'
        WHEN weight > 175 THEN '176-200'
        ELSE '175 or under' END AS weight_group
FROM benn.college_football_players
```

- Alternatively, you can use the column's alias in the GROUP BY clause like this:
```sql
SELECT 
    CASE 
        WHEN year = 'FR' THEN 'FR'
        WHEN year = 'SO' THEN 'SO'
        WHEN year = 'JR' THEN 'JR'
        WHEN year = 'SR' THEN 'SR'
        ELSE 'No Year Data' END AS year_group,
    COUNT(1) AS count
FROM benn.college_football_players
GROUP BY year_group
```

```sql
SELECT 
    CASE 
        WHEN year IN ('FR', 'SO') THEN 'underclass'
        WHEN year IN ('JR', 'SR') THEN 'upperclass'
        ELSE NULL END AS class_group,
    SUM(weight) AS combined_player_weight
FROM benn.college_football_players
WHERE state = 'CA'
GROUP BY 1
```

```sql
SELECT 
    CASE 
        WHEN state IN ('CA', 'OR', 'WA') THEN 'West Coast'
        WHEN state = 'TX' THEN 'Texas'
        ELSE 'Other' END AS arbitrary_regional_designation,
    COUNT(1) AS players
FROM benn.college_football_players
WHERE weight >= 300
GROUP BY 1
```

---

## Structured Query Language (SQL)

SQL, which is an abbreviation for Structured Query Language, is a language to request data from a database, to add, update, or remove data within a database, or to manipulate the metadata of the database.

SQL is a declarative language in which the expected result or operation is given without the specific details about how to accomplish the task. The steps required to execute SQL statements are handled transparently by the SQL database. Sometimes SQL is characterized as non-procedural because procedural languages generally require the details of the operations to be specified, such as opening and closing tables, loading and searching indexes, or flushing buffers and writing data to filesystems. Therefore, SQL is considered to be designed at a higher conceptual level of operation than procedural languages because the lower level logical and physical operations aren't specified and are determined by the SQL engine or server process that executes it.

Instructions are given in the form of statements, consisting of a specific SQL statement and additional parameters and operands that apply to that statement. SQL statements and their modifiers are based upon official SQL standards and certain extensions to that each database provider implements. Commonly used statements are grouped into the following categories:

Data Query Language (DQL)

    SELECT - Used to retrieve certain records from one or more tables.

Data Manipulation Language (DML)

    INSERT - Used to create a record.
    UPDATE - Used to change certain records.
    DELETE - Used to delete certain records.

Data Definition Language (DDL)

    CREATE - Used to create a new table, a view of a table, or other object in database.
    ALTER - Used to modify an existing database object, such as a table.
    DROP - Used to delete an entire table, a view of a table or other object in the database.

Data Control Language (DCL)

    GRANT - Used to give a privilege to someone.
    REVOKE - Used to take back privileges granted to someone.

###
## Difference between SQL and T-SQL
SQL
    
T-SQL

    SQL is a programming language which focuses on managing relational databases.

    

    T-SQL is a procedural extension used by SQL Server.

    This is used for controlling and manipulating data where large amounts of information are stored about products, clients, etc.

    

    T-SQL has some features that are not available in SQL. Like procedural programming elements and a local variable to provide more flexible control of how the application flows.

    SQL queries submitted individually to the database server.

    

    T-SQL writes a program in such a way that all commands are submitted to the server in a single go

    The syntax was formalized for many commands; some of these are SELECT, INSERT, UPDATE, DELETE, CREATE, and DROP.

    

    It also includes special functions like the converted date () and some other functions which are not part of the regular SQL.


## SQL as non-precedural language
- - SQL is called as a non Procedural language because the programmer or the user only specify what is needed and not tell the compiler how to do it, as done in Procedural language.

- Procedural capabilities are give to SQL using T-SQL or PL/SQL.

1. SQL | means Structured Query Language and.. well name says it all right :D
2) "structured" | well.. in fact one can say all languages are structured as they all defined by a structure called BNF which compilers use to validate syntax.
3) "procedural" | this is the more fuzy part. At higher level one simply executes one single SQL statement at any time vs a procedural language that can execute multiple "statements" controled by loops and conditional statements that execute the only thing any language able to do at its most lowest level whuch is to assign value(s) to a variable.
4) So from programmer high level one can in a simplistic way say procedural= way to call many statements while non-procedural= 1 single statement.
5) but 4) interpretation is too simplistic since in fact one can implement typical "procedural" mind thinking but at not so easy to read by another human looping, conditional behaviour right only using SQL.
6) But most human programmers got used/felt need to express more complex data behaviours using a more human readable language typically procedural which gives "reusability" in a more easy way or because -sadly- its harder for most human coders to express full data behaviours purely only using 1 single SQL statement which is and always be the most performant way to manipulate data (as its the language directly above data access).
7) So the way one single SQL statement gets processed can in fact be -and it is- seen as procedural sequence of actions since it also give us the same semantic concepts as "loops", and "conditional filtering" accepting a set of inputs and returning a set of ouputs.
8) The only "difference" I would say is "reusability" in where so called "procedural languages" programmer explicitly define reusable pieces of code (procedure or function for example) while "reusability" at SQL execution is much much more less human explicitly dictated. A good SQL programmer knows many many technical implementation details of how a SQL works at lowest level.
9) Imagine a similarity between how assembler language is for micro processors = same way as SQL is to access more complex data structures.



---
## NoSQL vs. SQL Summary
    SQL Databases   NoSQL Databases
Types   One type (SQL database) with minor variations   Many different types including key-value stores, document databases, wide-column stores, and graph databases
Development History Developed in 1970s to deal with first wave of data storage applications Developed in late 2000s to deal with limitations of SQL databases, especially scalability, multi-structured data, geo-distribution and agile development sprints
Examples    MySQL, Postgres, Microsoft SQL Server, Oracle Database  MongoDB, Cassandra, HBase, Neo4j
Data Storage Model  Individual records (e.g., 'employees') are stored as rows in tables, with each column storing a specific piece of data about that record (e.g., 'manager,' 'date hired,' etc.), much like a spreadsheet. Related data is stored in separate tables, and then joined together when more complex queries are executed. For example, 'offices' might be stored in one table, and 'employees' in another. When a user wants to find the work address of an employee, the database engine joins the 'employee' and 'office' tables together to get all the information necessary.    Varies based on database type. For example, key-value stores function similarly to SQL databases, but have only two columns ('key' and 'value'), with more complex information sometimes stored as BLOBs within the 'value' columns. Document databases do away with the table-and-row model altogether, storing all relevant data together in single 'document' in JSON, XML, or another format, which can nest values hierarchically.
Schemas Structure and data types are fixed in advance. To store information about a new data item, the entire database must be altered, during which time the database must be taken offline.   Typically dynamic, with some enforcing data validation rules. Applications can add new fields on the fly, and unlike SQL table rows, dissimilar data can be stored together as necessary. For some databases (e.g., wide-column stores), it is somewhat more challenging to add new fields dynamically.
Scaling Vertically, meaning a single server must be made increasingly powerful in order to deal with increased demand. It is possible to spread SQL databases over many servers, but significant additional engineering is generally required, and core relational features such as JOINs, referential integrity and transactions are typically lost.   Horizontally, meaning that to add capacity, a database administrator can simply add more commodity servers or cloud instances. The database automatically spreads data across servers as necessary.
Development Model   Mix of open technologies (e.g., Postgres, MySQL) and closed source (e.g., Oracle Database)  Open technologies
Supports multi-record ACID transactions Yes Mostly no. MongoDB 4.0 and beyond support multi-document ACID transactions. Learn more
Data Manipulation   Specific language using Select, Insert, and Update statements, e.g. SELECT fields FROM table WHERE… Through object-oriented APIs
Consistency Can be configured for strong consistency    Depends on product. Some provide strong consistency (e.g., MongoDB, with tunable consistency for reads) whereas others offer eventual consistency (e.g., Cassandra).

ref: https://www.mongodb.com/nosql-explained
---

## indexing:
Summary:

    Indexing is a small table which is consist of two columns.
    Two main types of indexing methods are 1)Primary Indexing 2) Secondary Indexing.
    Primary Index is an ordered file which is fixed length size with two fields.
    The primary Indexing is also further divided into two types 1)Dense Index 2)Sparse Index.
    In a dense index, a record is created for every search key valued in the database.
    A sparse indexing method helps you to resolve the issues of dense Indexing.
    The secondary Index is an indexing method whose search key specifies an order different from the sequential order of the file.
    Clustering index is defined as an order data file.
    Multilevel Indexing is created when a primary index does not fit in memory.
    The biggest benefit of Indexing is that it helps you to reduce the total number of I/O operations needed to retrieve that data.
    The biggest drawback to performing the indexing database management system, you need a primary key on the table with a unique value.




Indexing is defined as a data structure technique which allows you to quickly retrieve records from a database file. It is based on the same attributes on which the Indices has been done.

An index

    Takes a search key as input
    Efficiently returns a collection of matching records.

An Index is a small table having only two columns. The first column comprises a copy of the primary or candidate key of a table. Its second column contains a set of pointers for holding the address of the disk block where that specific key value stored.
Secondary Index

The secondary Index can be generated by a field which has a unique value for each record, and it should be a candidate key. It is also known as a non-clustering index
Example of secondary Indexing

In a bank account database, data is stored sequentially by acc_no; you may want to find all accounts in of a specific branch of ABC bank.

Here, you can have a secondary index for every search-key. Index record is a record point to a bucket that contains pointers to all the records with their specific search-key value. 


Clustering Index

In a clustered index, records themselves are stored in the Index and not pointers

Example:

Let's assume that a company recruited many employees in various departments. In this case, clustering indexing should be created for all employees who belong to the same dept.

It is considered in a single cluster, and index points point to the cluster as a whole. Here, Department _no is a non-unique key.
What is Multilevel Index?

Multilevel Indexing is created when a primary index does not fit in memory. In this type of indexing method, you can reduce the number of disk accesses to short any record and kept on a disk as a sequential file and create a sparse base on that file.
B-Tree Index

B-tree index is the widely used data structures for Indexing. It is a multilevel index format technique which is balanced binary search trees. All leaf nodes of the B tree signify actual data pointers.

Moreover, all leaf nodes are interlinked with a link list, which allows a B tree to support both random and sequential access. 
---


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


## Remove `Duplicate without` using `distinct`
```sql
Remove Duplicates using self Join
YourTable

emp_name   emp_address  sex  matial_status  
uuuu       eee          m    s
iiii       iii          f    s
uuuu       eee          m    s

SELECT emp_name, emp_address, sex, marital_status
from YourTable a
WHERE NOT EXISTS (select 1 
         from YourTable b
         where b.emp_name = a.emp_name and
               b.emp_address = a.emp_address and
               b.sex = a.sex and
               b.create_date >= a.create_date)

3. Remove Duplicates using group By

SELECT FirstName, LastName, MobileNo, COUNT(*) as CNT
FROM  CUSTOMER
GROUP BY FirstName, LastName, MobileNo;
HAVING COUNT(*)  = 1
```


## Handling NULL values in query results (the NVL function)
- NULL values represent the absence of any actual value
- The syntax of testing for NULL values in a WHERE clause
For example, to return all employees who do not receive a commission, the query would be:
```sql
SELECT EMPNO, ENAME, SAL
FROM EMP
WHERE COMM IS NULL;
```

> It is important to remember that NULL is not the same as, say, zero for a numeric attribute.

### NVL function
- it is used to substitute other values in place of NULLs in the results of queries. 
- This may be required for a number of reasons:
    1. By default, arithmetic and aggregate functions ignore NULL values in query results. Sometimes this is what is required, but at other times we might explicitly wish to consider a NULL in a numeric column as actually representing the value zero, for example.
    2. We may wish to replace a NULL value, which will appear as a blank column in the displayed results of a query, with a more explicit indication that there was no value for that column instance.

Examples of using the NVL function

1. An example of using NVL to treat all employees with NULL commissions as if they had zero commission:
```sql
SELECT EMPNO,NVL(COMM, 0)
FROM EMP;
```

2. To display the word 'unassigned' wherever a NULL value is retrieved from the JOB attribute:
```sql
SELECT EMPNO,NVL(job, 'unassigned')
FROM EMP;
```




## Cloud Deployment Models

The cloud deployment models summarised below are the following:

    Private Cloud: the cloud services used by a single organization, which are not exposed to the public. A private cloud resides inside the organization and must be behind a firewall, so only the organization has access to it and can manage it.
    Public Cloud: the cloud services are exposed to the public and can be used by anyone. Virtualization is typically used to build the cloud services that are offered to the public. An example of a public cloud is Amazon Web Services (AWS).
    Hybrid Cloud: the cloud services can be distributed among public and private clouds, where sensitive applications are kept inside the organization’s network (by using a private cloud), whereas other services can be hosted outside the organization’s network (by using a public cloud). Users can them interchangeably use private as well as public cloud services in every day operations.

The biggest differences between public, private and hybrid cloud are described in the table below.
Difference  Private     Public  Hybrid
Tenancy     Single tenancy: there’s only the data of a single organization stored in the cloud.     Multi-tenancy: the data of multiple organizations in stored in a shared environment.    The data stored in the public cloud is usually multi-tenant, which means the data from multiple organizations is stored in a shared environment. The data stored in private cloud is kept private by the organization.
Exposed to the Public   No: only the organization itself can use the private cloud services.    Yes: anyone can use the public cloud services.  The services running on a private cloud can be accessed only the organization’s users, while the services running on public cloud can be accessed by anyone.
Data Center Location    Inside the organization’s network.  Anywhere on the Internet where the cloud service provider’s services are located.   Inside the organization’s network for private cloud services as well as anywhere on the Internet for public cloud services.
Cloud Service Management    The organization must have their own administrators managing their private cloud services.  The cloud service provider manages the services, where the organization merely uses them.   The organization itself must manage the private cloud, while the public cloud is managed by the CSP.
Hardware Components     Must be provided by the organization itself, which has to buy physical servers to build the private cloud on.   The CSP provides all the hardware and ensures it’s working at all times.    The organization must provide hardware for the private cloud, while the hardware of CSP is used for public cloud services.
Expenses    Can be quite expensive, since the hardware, applications and network have to be provided and managed by the organization itself.    The CSP has to provide the hardware, set-up the application and provide the network accessibility according to the SLA.     The private cloud services must be provided by the organization, including the hardware, applications and network, while the CSP manages the public cloud services.

 

As you can see, the hybrid cloud is a combination of private, as well as public cloud, used together by the same organization to pull the best features from each.
Which one should you choose?

It’s important to keep in mind when deciding whether to build a private or public cloud, to properly weigh the differences against each other. In most cases they can be thought of as advantages or disadvantages, depending on the usage required. If we’d like to store our backup data somewhere in the cloud, it’s important to determine the sensitivity of said data. For example, if we are storing confidential information such as credit card information or medical records we absolutely must store that data in a private cloud but when it comes to non-sensitive info, we can store it in a public cloud if it keeps costs down considerably.






## 
What is a public cloud?

Public clouds are the most common way of deploying cloud computing. The cloud resources (like servers and storage) are owned and operated by a third-party cloud service provider and delivered over the Internet. Microsoft Azure is an example of a public cloud. With a public cloud, all hardware, software and other supporting infrastructure is owned and managed by the cloud provider. In a public cloud, you share the same hardware, storage and network devices with other organisations or cloud “tenants.” You access services and manage your account using a web browser. Public cloud deployments are frequently used to provide web-based email, online office applications, storage and testing and development environments.
Advantages of public clouds:

    Lower costs—no need to purchase hardware or software and you pay only for the service you use.
    No maintenance—your service provider provides the maintenance.
    Near-unlimited scalability—on-demand resources are available to meet your business needs.
    High reliability—a vast network of servers ensures against failure.

What is a private cloud?

A private cloud consists of computing resources used exclusively by one business or organisation. The private cloud can be physically located at your organisation’s on-site datacenter or it can be hosted by a third-party service provider. But in a private cloud, the services and infrastructure are always maintained on a private network and the hardware and software are dedicated solely to your organisation. In this way, a private cloud can make it easier for an organisation to customise its resources to meet specific IT requirements. Private clouds are often used by government agencies, financial institutions, any other mid- to large-size organisations with business-critical operations seeking enhanced control over their environment.
Advantages of a private clouds:

    More flexibility—your organisation can customise its cloud environment to meet specific business needs.
    Improved security—resources are not shared with others, so higher levels of control and security are possible.
    High scalability—private clouds still afford the scalability and efficiency of a public cloud.

What is a hybrid cloud?

Often called “the best of both worlds,” hybrid clouds combine on-premises infrastructure, or private clouds, with public clouds so organisations can reap the advantages of both. In a hybrid cloud, data and applications can move between private and public clouds for greater flexibility and more deployment options. For instance, you can use the public cloud for high-volume, lower-security needs such as web-based email and the private cloud (or other on-premises infrastructure) for sensitive, business-critical operations like financial reporting. In a hybrid cloud, “cloud bursting” is also an option. This is when an application or resource runs in the private cloud until there is a spike in demand (such as seasonal event like online shopping or tax filing), at which point the organisation can “burst through” to the public cloud to tap into additional computing resources.
Advantages of hybrid clouds:

    Control—your organisation can maintain a private infrastructure for sensitive assets.
    Flexibility—you can take advantage of additional resources in the public cloud when you need them.
    Cost-effectiveness—with the ability to scale to the public cloud, you pay for extra computing power only when needed.
    Ease—transitioning to the cloud does not have to be overwhelming because you can migrate gradually—phasing in workloads over time.

