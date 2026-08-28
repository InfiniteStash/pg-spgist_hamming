pg-spgist_hamming
==============

Forked from https://github.com/fake-name/pg-spgist_hamming, all credit for orignal implementation goes to [fake-name](https://github.com/fake-name)

This fork improves tree balancing and adds a batched join plan for perceptual
hash searches. The low-level traversal shares index page visits across up to 64
hashes; larger `UNNEST` inputs are automatically processed in 64-hash chunks.

Tested on PostgreSQL 18

------

Quickstart:

This module has a simple makefile that uses `pg_config` to do it's magic. Check you have the `pg_config` shell command, and that it's output looks reasonable.

If you do, installing is a two steps:

```
cd bktree
make
sudo make install
sudo make installcheck   # to run tests that check everything installed correctly.
```


Once you have it installed:


```
# Enable extension in current database (Note: This is per-database, so if you want to use it on 
# multiple DBs, you'll have to enable it in each.
CREATE EXTENSION bktree;

# Use the enabled extension to create an index. 
# phash_column MUST be a int64 ("bigint") type.
CREATE INDEX bk_index_name ON table_name USING spgist (phash_column bktree_ops);

# Query across the table within a specified edit distance.
SELECT <columns here> FROM table_name WHERE phash_column <@ (target_phash_int64, search_distance_int);

# Query a table for a batch of values within a specified edit distance.
SELECT <columns>
FROM UNNEST(<target_phash_int64[]>) AS query(phash)
JOIN table_name
  ON phash_column <@ (query.phash, search_distance_int)::bktree_area;
```

You'll need to replace things like `bk_index_name`, `table_name`, `target_phash_int64`, `search_distance_int`, 
and `phash_column` with appropriate values for your database.

`phash_column` must be a column of type `bigint`. Currently, only 64-bit phash values are supported, and they're 
stored in signed format. 

The planner optimization is enabled by default. It can be disabled per session
for correctness checks or comparisons:

```sql
SET bktree.enable_customscan = off;
```

For a partial index, include the predicate in the query. For example:

```sql
CREATE INDEX fingerprints_phash_idx
ON fingerprints USING spgist (hash bktree_ops)
WHERE algorithm = 'PHASH';

SELECT fp.*, query.phash
FROM UNNEST($1::bigint[]) AS query(phash)
JOIN fingerprints AS fp
  ON fp.hash <@ (query.phash, $2)::bktree_area
 AND fp.algorithm = 'PHASH';
```
