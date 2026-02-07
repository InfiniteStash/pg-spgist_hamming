/*
 * bktree.h - Shared types and functions for BK-tree SP-GiST index
 */

#ifndef BKTREE_H
#define BKTREE_H

#include "postgres.h"
#include "access/spgist.h"
#include "access/spgist_private.h"
#include "storage/itemptr.h"

/* Maximum number of targets in a single batch (limited by bitmask) */
#define MAX_BATCH_TARGETS 64

/* Result item from batch search */
typedef struct BatchResult
{
	int64		target_hash;
	int64		match_hash;
	ItemPointerData heap_tid;
} BatchResult;

/* Batch search state - used by both SRF and CustomScan */
typedef struct BatchSearchState
{
	int64	   *targets;		/* array of target hashes */
	int			ntargets;		/* number of targets */
	int64		distance;		/* search distance */

	BatchResult *results;		/* accumulated results */
	int			nresults;		/* number of results */
	int			max_results;	/* allocated size */

	Relation	index;			/* the SP-GiST index */
	SpGistState spgstate;		/* SP-GiST state */
} BatchSearchState;

/*
 * Core batch search execution - shared by both SRF and CustomScan.
 * Caller must initialize targets, ntargets, distance, and open the index.
 * Results are stored in state->results/nresults.
 */
extern void bktree_batch_execute(BatchSearchState *state);

/*
 * OID cache functions for extension types and operators.
 * These are cached after first lookup for performance.
 */
extern Oid bktree_area_type_oid(void);
extern Oid bktree_containment_op_oid(void);  /* <@ operator */

/*
 * Register planner hooks for CustomScan join optimization.
 * Called from _PG_init().
 */
extern void bktree_register_hooks(void);

#endif /* BKTREE_H */
