# Migrating sqlite to postgresql
We started testing the migration of SQLite db to PostgreSQL db to simplify the multiprocessing architecture.

## Running the migration
We tried to run the migration on our VM by just installing `pgloader` using apt.

This did not work out as the program was throwing errors about running out of memory. We tested the methods built into `pgloader` 
for batch-size and memory-management but the problems persisted. After reading through some GitHub issues about people having the
same problem the general suggestion was to use the docker container published by the autor. 

Using the docker container we were able to migrate our db from SQLite to PostgreSQL. The relevant docker-compose file is
present in this directory.

## The general file for migration
Contents of `db.load` file
```
load database
     from sqlite:///var/lib/postgresql/data/datadir-100/crawl-data.sqlite
     into pgsql://postgres@localhost/webgraph

with include drop, create tables, create indexes, reset sequences

CAST type string to text drop typemod;
```

```
pgloader db.load
```