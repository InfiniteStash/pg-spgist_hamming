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

-- Check that EXPLAIN shows our custom scan (when enabled)
-- Note: This might show nested loop if cost estimation prefers it
EXPLAIN (COSTS OFF)
SELECT t.*, target
FROM UNNEST(ARRAY[12345::int8, 67890::int8]) AS target
JOIN test_batch_join t ON t.hash <@ (target, 5)::bktree_area;

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
SET enable_customscan = off;
SELECT count(*) AS nested_loop_count FROM (
    SELECT t.*, target
    FROM UNNEST(ARRAY[100::int8, 200::int8]) AS target
    JOIN test_batch_join t ON t.hash <@ (target, 3)::bktree_area
) subq;

SET enable_customscan = on;
SELECT count(*) AS custom_scan_count FROM (
    SELECT t.*, target
    FROM UNNEST(ARRAY[100::int8, 200::int8]) AS target
    JOIN test_batch_join t ON t.hash <@ (target, 3)::bktree_area
) subq;

-- Cleanup
DROP TABLE test_batch_join;
