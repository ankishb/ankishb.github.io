---
layout: post
title: "sql-tutorial"
date: 2019-09-05
tag: sql
---

In this post, i will cover some fundamentals of sql and their commands. This won't be an extensive tutorial of sql. It can be viewed as an go though tutorial or refreshing your sql concepts. Let's dive into it.

Sql commands are divided into `4` subgroups, which are
as follows:
1. DDL (Data Definition Language)
It deals with database schemas and descriptions, of how the data should reside in the database. 
    1. CREATE - to create a database and its objects like (table, index, views, store procedure, function, and triggers)
    2. ALTER - alters the structure of the existing database
    3. DROP - delete objects from the database
    4. TRUNCATE - remove all records from a table, including all spaces allocated for the records are removed
    5. COMMENT - add comments to the data dictionary
    6. RENAME - rename an object

2. DML (Data Manipulation language)
it is used to store, modify, retrieve, delete and update data in a database. 
    1. SELECT - retrieve data from a database
    2. INSERT - insert data into a table
    3. UPDATE - updates existing data within a table
    4. DELETE - Delete all records from a database table
    5. MERGE - insert or update
    6. CALL - call a PL/SQL or Java subprogram
    7. EXPLAIN PLAN - interpretation of the data access path
    8. LOCK TABLE - concurrency Control

3. DCL (Data Control Language)
It includes commands such as GRANT and mostly concerned with rights, permissions and other controls of the database system.
    1. GRANT - allow users access privileges to the database
    2. REVOKE - withdraw users access privileges given by using the GRANT command

4. TCL (Transaction Control Language)
It deals with a transaction within a database.
    1. COMMIT - commits a Transaction
    2. ROLLBACK - rollback a transaction in case of any error occurs
    3. SAVEPOINT - to rollback the transaction making points within groups
    4. SET TRANSACTION - specify characteristics of the transaction




## JSON (javascript object notation)
- serialize data
- semi-structured data

Objects: label-pair

{ "Books":
    [
        {   "title": "the wise monk",
            "Price": 600,
            "Author": [{"first_name": "Robin",
                        "last_name": "sharma"}]
        },
        {   "title": "12 year as slave",
            "Price": 1890,
            "ISBN No": 1234-345,
            "Author": [{"first_name": "soloman",
                        "last_name": "northup"}]
        }

    ]
}