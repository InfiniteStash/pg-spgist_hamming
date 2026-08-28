--
-- Test CustomScan join optimization for UNNEST + bktree pattern
--

-- Create test table
CREATE TABLE test_batch_join(id serial PRIMARY KEY, hash int8);

-- Insert test data
INSERT INTO test_batch_join(hash)
SELECT (random() * 1e18)::int8 FROM generate_series(1, 10000);

-- Create bktree index
CREATE INDEX test_batch_join_hash_idx ON test_batch_join USING spgist(hash bktree_ops);

-- Analyze for accurate statistics
ANALYZE test_batch_join;

-- Basic functionality test: UNNEST + JOIN with bktree
-- This should use the batch join optimization
SELECT count(*) FROM (
    SELECT t.*, target
    FROM UNNEST(ARRAY[12345::int8, 67890::int8, 11111::int8]) AS target
    JOIN test_batch_join t ON t.hash <@ (target, 10)::bktree_area
) subq;

-- Test with actual matching values
-- Insert some known values we can query for
INSERT INTO test_batch_join(hash) VALUES
    (100), (101), (102), (103),  -- distance 0-3 from 100
    (200), (201), (202), (203);  -- distance 0-3 from 200

-- Query for exact matches
SELECT t.id, t.hash, target
FROM UNNEST(ARRAY[100::int8, 200::int8]) AS target
JOIN test_batch_join t ON t.hash <@ (target, 0)::bktree_area
ORDER BY t.hash, target;

-- Query with distance 1 (should get more matches)
SELECT count(*) FROM (
    SELECT t.id, t.hash, target
    FROM UNNEST(ARRAY[100::int8, 200::int8]) AS target
    JOIN test_batch_join t ON t.hash <@ (target, 3)::bktree_area
) subq;

-- Test with more targets
SELECT count(*) FROM (
    SELECT t.*, target
    FROM UNNEST(ARRAY[1::int8, 2, 3, 4, 5, 6, 7, 8, 9, 10]) AS target
    JOIN test_batch_join t ON t.hash <@ (target, 2)::bktree_area
) subq;

-- Test with empty array (should return no rows)
SELECT count(*) FROM (
    SELECT t.*, target
    FROM UNNEST(ARRAY[]::int8[]) AS target
    JOIN test_batch_join t ON t.hash <@ (target, 5)::bktree_area
) subq;

-- Verify results match between custom scan and regular nested loop
-- by comparing counts with custom scan disabled vs enabled
SET bktree.enable_customscan = off;
SELECT count(*) AS nested_loop_count FROM (
    SELECT t.*, target
    FROM UNNEST(ARRAY[100::int8, 200::int8]) AS target
    JOIN test_batch_join t ON t.hash <@ (target, 3)::bktree_area
) subq;

SET bktree.enable_customscan = on;
SELECT count(*) AS custom_scan_count FROM (
    SELECT t.*, target
    FROM UNNEST(ARRAY[100::int8, 200::int8]) AS target
    JOIN test_batch_join t ON t.hash <@ (target, 3)::bktree_area
) subq;

-- Runtime parameters can contain more than 64 rows even when the planner's
-- UNNEST estimate is smaller.  The executor must search all chunks.
INSERT INTO test_batch_join(hash)
SELECT -g FROM generate_series(1000, 1064) AS g;

SELECT count(*) AS multi_chunk_count FROM (
    SELECT t.hash, target
    FROM UNNEST(ARRAY(SELECT -g::int8 FROM generate_series(1000, 1064) AS g)) AS target
    JOIN test_batch_join t ON t.hash <@ (target, 0)::bktree_area
) subq;

-- A partial index must not be used unless its predicate is present.
CREATE TABLE test_batch_partial(id serial PRIMARY KEY, algorithm text, hash int8);
INSERT INTO test_batch_partial(algorithm, hash)
VALUES ('PHASH', 42), ('OTHER', 42);
CREATE INDEX test_batch_partial_hash_idx
ON test_batch_partial USING spgist(hash bktree_ops)
WHERE algorithm = 'PHASH';
ANALYZE test_batch_partial;

SELECT count(*) AS without_predicate FROM (
    SELECT t.id
    FROM UNNEST(ARRAY[42::int8]) AS target
    JOIN test_batch_partial t ON t.hash <@ (target, 0)::bktree_area
) subq;

SELECT count(*) AS with_predicate FROM (
    SELECT t.id
    FROM UNNEST(ARRAY[42::int8]) AS target
    JOIN test_batch_partial t ON t.hash <@ (target, 0)::bktree_area
    WHERE t.algorithm = 'PHASH'
) subq;

-- PostgreSQL may scan a partial SP-GiST index without a distance scan key.
-- The consistent functions must not prune the entire tree in that case.
SET enable_seqscan = off;
SELECT count(*) AS unconstrained_index_count FROM (
    SELECT hash
    FROM test_batch_partial
    WHERE algorithm = 'PHASH'
    LIMIT 1
) subq;
RESET enable_seqscan;

-- Cleanup
DROP TABLE test_batch_partial;
DROP TABLE test_batch_join;
