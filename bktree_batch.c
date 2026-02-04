/*
 * bktree_batch.c - Batch search for BK-tree SP-GiST index
 *
 * Provides a function to search for multiple target hashes in a single
 * index traversal, significantly faster than separate queries.
 */

#include "postgres.h"
#include "funcapi.h"
#include "access/spgist_private.h"
#include "access/relscan.h"
#include "access/heapam.h"
#include "access/tableam.h"
#include "access/htup_details.h"
#include "catalog/namespace.h"
#include "catalog/pg_type.h"
#include "catalog/indexing.h"
#include "executor/executor.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/fmgroids.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "utils/syscache.h"
#include "storage/bufmgr.h"

#include <string.h>

PG_FUNCTION_INFO_V1(bktree_batch_search);
PG_FUNCTION_INFO_V1(bktree_search);

/* Maximum number of targets in a single batch (limited by bitmask) */
#define MAX_BATCH_TARGETS 64

/* Work item for traversal stack */
typedef struct BatchStackItem
{
	BlockNumber blkno;
	OffsetNumber offset;
	uint64		activeTargets;	/* bitmask of targets still searching this path */
} BatchStackItem;

/* Result item */
typedef struct BatchResult
{
	int64		target_hash;
	int64		match_hash;
	ItemPointerData heap_tid;
} BatchResult;

/* Batch search state */
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
 * Compute hamming distance between two int64 values
 */
static inline int
hamming_distance(int64 a, int64 b)
{
	return __builtin_popcountll((uint64)(a ^ b));
}

/*
 * Add a result to the results array
 */
static void
add_result(BatchSearchState *state, int64 target, int64 match, ItemPointer tid)
{
	if (state->nresults >= state->max_results)
	{
		state->max_results *= 2;
		state->results = repalloc(state->results,
								  sizeof(BatchResult) * state->max_results);
	}

	state->results[state->nresults].target_hash = target;
	state->results[state->nresults].match_hash = match;
	state->results[state->nresults].heap_tid = *tid;
	state->nresults++;
}

/*
 * Process leaf tuples starting at given offset, following the chain
 */
static void
process_leaf_chain(BatchSearchState *state, Relation index,
				   BlockNumber blkno, OffsetNumber startOffset,
				   uint64 activeTargets)
{
	Buffer		buffer;
	Page		page;
	OffsetNumber offset;

	buffer = ReadBuffer(index, blkno);
	LockBuffer(buffer, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buffer);

	offset = startOffset;
	while (offset != InvalidOffsetNumber)
	{
		SpGistLeafTuple leafTuple;
		ItemId		itemId;
		int64		leafHash;
		int			i;
		uint64		mask;

		itemId = PageGetItemId(page, offset);
		if (!ItemIdIsUsed(itemId))
			break;

		leafTuple = (SpGistLeafTuple) PageGetItem(page, itemId);

		if (leafTuple->tupstate != SPGIST_LIVE)
		{
			/* Handle redirect */
			if (leafTuple->tupstate == SPGIST_REDIRECT)
			{
				SpGistDeadTuple dt = (SpGistDeadTuple) leafTuple;
				BlockNumber newBlk = ItemPointerGetBlockNumber(&dt->pointer);
				OffsetNumber newOff = ItemPointerGetOffsetNumber(&dt->pointer);
				UnlockReleaseBuffer(buffer);
				/* Follow redirect recursively */
				process_leaf_chain(state, index, newBlk, newOff, activeTargets);
				return;
			}
			offset = SGLT_GET_NEXTOFFSET(leafTuple);
			continue;
		}

		/* Extract the leaf datum (int64 hash) */
		leafHash = DatumGetInt64(SGLTDATUM(leafTuple, &state->spgstate));

		/* Check against all active targets */
		mask = 1;
		for (i = 0; i < state->ntargets; i++, mask <<= 1)
		{
			if (activeTargets & mask)
			{
				int dist = hamming_distance(leafHash, state->targets[i]);
				if (dist <= state->distance)
				{
					add_result(state, state->targets[i], leafHash,
							   &leafTuple->heapPtr);
				}
			}
		}

		/* Follow chain to next tuple */
		offset = SGLT_GET_NEXTOFFSET(leafTuple);
	}

	UnlockReleaseBuffer(buffer);
}

/*
 * Recursive traversal of inner nodes
 */
static void
traverse_inner(BatchSearchState *state, Relation index,
			   BlockNumber blkno, OffsetNumber offset,
			   uint64 activeTargets,
			   BatchStackItem *stack, int *stackDepth, int maxStack)
{
	Buffer		buffer;
	Page		page;
	SpGistInnerTuple innerTuple;
	ItemId		itemId;
	int64		prefix;
	int			i;
	uint64		mask;
	SpGistNodeTuple node;
	int			nodeN;
	uint64		nodeTargets[65];

	if (activeTargets == 0 || *stackDepth >= maxStack - 65)
		return;

	buffer = ReadBuffer(index, blkno);
	LockBuffer(buffer, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buffer);

	/* Check if this is actually a leaf page */
	if (SpGistPageIsLeaf(page))
	{
		UnlockReleaseBuffer(buffer);
		process_leaf_chain(state, index, blkno, offset, activeTargets);
		return;
	}

	itemId = PageGetItemId(page, offset);
	if (!ItemIdIsUsed(itemId))
	{
		UnlockReleaseBuffer(buffer);
		return;
	}

	innerTuple = (SpGistInnerTuple) PageGetItem(page, itemId);

	if (innerTuple->tupstate != SPGIST_LIVE)
	{
		/* Handle redirect */
		if (innerTuple->tupstate == SPGIST_REDIRECT)
		{
			SpGistDeadTuple dt = (SpGistDeadTuple) innerTuple;
			BlockNumber newBlk = ItemPointerGetBlockNumber(&dt->pointer);
			OffsetNumber newOff = ItemPointerGetOffsetNumber(&dt->pointer);
			UnlockReleaseBuffer(buffer);
			traverse_inner(state, index, newBlk, newOff, activeTargets,
						   stack, stackDepth, maxStack);
			return;
		}
		UnlockReleaseBuffer(buffer);
		return;
	}

	/* Initialize nodeTargets */
	memset(nodeTargets, 0, sizeof(nodeTargets));

	/* Handle allTheSame case */
	if (innerTuple->allTheSame)
	{
		/* All nodes equivalent - visit first node with all active targets */
		SGITITERATE(innerTuple, nodeN, node)
		{
			BlockNumber childBlk = ItemPointerGetBlockNumber(&node->t_tid);
			OffsetNumber childOff = ItemPointerGetOffsetNumber(&node->t_tid);

			if (BlockNumberIsValid(childBlk))
			{
				stack[*stackDepth].blkno = childBlk;
				stack[*stackDepth].offset = childOff;
				stack[*stackDepth].activeTargets = activeTargets;
				(*stackDepth)++;
			}
		}
		UnlockReleaseBuffer(buffer);
		return;
	}

	/* Get the prefix datum (the pivot hash for this node) */
	prefix = DatumGetInt64(SGITDATUM(innerTuple, &state->spgstate));

	/*
	 * For each target, compute distance to prefix and determine which
	 * child nodes need to be visited.
	 */
	mask = 1;
	for (i = 0; i < state->ntargets; i++, mask <<= 1)
	{
		if (activeTargets & mask)
		{
			int dist = hamming_distance(prefix, state->targets[i]);
			int minDist = (dist > state->distance) ? (dist - state->distance) : 0;
			int maxDist = (dist + state->distance > 64) ? 64 : (dist + state->distance);
			int d;

			for (d = minDist; d <= maxDist; d++)
			{
				nodeTargets[d] |= mask;
			}
		}
	}

	/* Queue children that have active targets */
	SGITITERATE(innerTuple, nodeN, node)
	{
		if (nodeN < 65 && nodeTargets[nodeN] != 0)
		{
			BlockNumber childBlk = ItemPointerGetBlockNumber(&node->t_tid);
			OffsetNumber childOff = ItemPointerGetOffsetNumber(&node->t_tid);

			if (BlockNumberIsValid(childBlk))
			{
				stack[*stackDepth].blkno = childBlk;
				stack[*stackDepth].offset = childOff;
				stack[*stackDepth].activeTargets = nodeTargets[nodeN];
				(*stackDepth)++;
			}
		}
	}

	UnlockReleaseBuffer(buffer);
}

/*
 * Main batch search function
 */
Datum
bktree_batch_search(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	BatchSearchState *state;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		Oid			indexOid;
		ArrayType  *targetArray;
		int64	   *targets;
		int			ntargets;
		int64		distance;
		Relation	index;
		TupleDesc	tupdesc;
		Datum	   *targetDatums;
		bool	   *targetNulls;
		int			i;

		/* Stack for iterative traversal */
		BatchStackItem *stack;
		int			stackDepth;
		int			maxStack;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/* Get arguments */
		indexOid = PG_GETARG_OID(0);
		targetArray = PG_GETARG_ARRAYTYPE_P(1);
		distance = PG_GETARG_INT64(2);

		/* Deconstruct target array */
		deconstruct_array(targetArray, INT8OID, 8, true, 'd',
						  &targetDatums, &targetNulls, &ntargets);

		if (ntargets > MAX_BATCH_TARGETS)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("too many targets, maximum is %d", MAX_BATCH_TARGETS)));

		/* Copy targets to int64 array */
		targets = palloc(sizeof(int64) * ntargets);
		for (i = 0; i < ntargets; i++)
		{
			if (targetNulls[i])
				ereport(ERROR,
						(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
						 errmsg("target array cannot contain NULL values")));
			targets[i] = DatumGetInt64(targetDatums[i]);
		}

		/* Open the index */
		index = index_open(indexOid, AccessShareLock);

		/* Verify it's a SP-GiST index */
		if (index->rd_rel->relam != SPGIST_AM_OID)
			ereport(ERROR,
					(errcode(ERRCODE_WRONG_OBJECT_TYPE),
					 errmsg("index is not a SP-GiST index")));

		/* Initialize state */
		state = palloc0(sizeof(BatchSearchState));
		state->targets = targets;
		state->ntargets = ntargets;
		state->distance = distance;
		state->max_results = 1024;
		state->results = palloc(sizeof(BatchResult) * state->max_results);
		state->nresults = 0;
		state->index = index;

		initSpGistState(&state->spgstate, index);

		/* Allocate traversal stack */
		maxStack = 65 * 100;  /* Should be plenty */
		stack = palloc(sizeof(BatchStackItem) * maxStack);
		stackDepth = 0;

		/* Start with root */
		stack[stackDepth].blkno = SPGIST_ROOT_BLKNO;
		stack[stackDepth].offset = FirstOffsetNumber;
		stack[stackDepth].activeTargets = (ntargets == 64) ? ~0ULL : ((1ULL << ntargets) - 1);
		stackDepth++;

		/* Iterative DFS traversal using stack */
		while (stackDepth > 0)
		{
			BatchStackItem item;

			stackDepth--;
			item = stack[stackDepth];

			if (item.activeTargets == 0)
				continue;

			traverse_inner(state, index, item.blkno, item.offset,
						   item.activeTargets, stack, &stackDepth, maxStack);
		}

		pfree(stack);

		/* Build result tuple descriptor */
		tupdesc = CreateTemplateTupleDesc(3);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "target_hash", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "match_hash", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "match_tid", TIDOID, -1, 0);

		funcctx->tuple_desc = BlessTupleDesc(tupdesc);
		funcctx->user_fctx = state;

		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	state = (BatchSearchState *) funcctx->user_fctx;

	if (funcctx->call_cntr < state->nresults)
	{
		Datum		values[3];
		bool		nulls[3] = {false, false, false};
		HeapTuple	tuple;
		BatchResult *res = &state->results[funcctx->call_cntr];

		values[0] = Int64GetDatum(res->target_hash);
		values[1] = Int64GetDatum(res->match_hash);
		values[2] = ItemPointerGetDatum(&res->heap_tid);

		tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);

		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
	}
	else
	{
		/* Clean up */
		index_close(state->index, AccessShareLock);
		SRF_RETURN_DONE(funcctx);
	}
}

/*
 * Find a SP-GiST bktree index on the specified table and column.
 * Returns InvalidOid if not found.
 */
static Oid
find_bktree_index(Oid tableOid, const char *columnName)
{
	Relation	indexRelation;
	SysScanDesc scan;
	HeapTuple	indexTuple;
	Oid			resultOid = InvalidOid;
	AttrNumber	colAttNum = InvalidAttrNumber;
	int			i;

	/* First, find the column's attribute number */
	for (i = 1; i <= MaxHeapAttributeNumber; i++)
	{
		HeapTuple	attTuple;

		attTuple = SearchSysCache2(ATTNUM,
								   ObjectIdGetDatum(tableOid),
								   Int16GetDatum(i));
		if (HeapTupleIsValid(attTuple))
		{
			Form_pg_attribute att = (Form_pg_attribute) GETSTRUCT(attTuple);

			if (strcmp(NameStr(att->attname), columnName) == 0)
			{
				colAttNum = att->attnum;
				ReleaseSysCache(attTuple);
				break;
			}
			ReleaseSysCache(attTuple);
		}
		else
		{
			break;  /* No more attributes */
		}
	}

	if (colAttNum == InvalidAttrNumber)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_COLUMN),
				 errmsg("column \"%s\" does not exist", columnName)));

	/* Scan pg_index for indexes on this table */
	indexRelation = table_open(IndexRelationId, AccessShareLock);
	scan = systable_beginscan(indexRelation, IndexIndrelidIndexId, true,
							  NULL, 0, NULL);

	while ((indexTuple = systable_getnext(scan)) != NULL)
	{
		Form_pg_index indexForm = (Form_pg_index) GETSTRUCT(indexTuple);
		HeapTuple	classTuple;
		Form_pg_class classForm;

		/* Check if this index is on our table */
		if (indexForm->indrelid != tableOid)
			continue;

		/* Check if the column is in this index */
		if (indexForm->indkey.dim1 < 1 || indexForm->indkey.values[0] != colAttNum)
			continue;

		/* Check if it's a SP-GiST index */
		classTuple = SearchSysCache1(RELOID, ObjectIdGetDatum(indexForm->indexrelid));
		if (!HeapTupleIsValid(classTuple))
			continue;

		classForm = (Form_pg_class) GETSTRUCT(classTuple);
		if (classForm->relam == SPGIST_AM_OID)
		{
			resultOid = indexForm->indexrelid;
			ReleaseSysCache(classTuple);
			break;
		}
		ReleaseSysCache(classTuple);
	}

	systable_endscan(scan);
	table_close(indexRelation, AccessShareLock);

	return resultOid;
}

/*
 * Core batch search logic - shared by both entry points
 */
static void
do_batch_search(BatchSearchState *state, Relation index)
{
	BatchStackItem *stack;
	int			stackDepth;
	int			maxStack;

	initSpGistState(&state->spgstate, index);

	/* Allocate traversal stack */
	maxStack = 65 * 100;
	stack = palloc(sizeof(BatchStackItem) * maxStack);
	stackDepth = 0;

	/* Start with root */
	stack[stackDepth].blkno = SPGIST_ROOT_BLKNO;
	stack[stackDepth].offset = FirstOffsetNumber;
	stack[stackDepth].activeTargets = (state->ntargets == 64) ? ~0ULL : ((1ULL << state->ntargets) - 1);
	stackDepth++;

	/* Iterative DFS traversal using stack */
	while (stackDepth > 0)
	{
		BatchStackItem item;

		stackDepth--;
		item = stack[stackDepth];

		if (item.activeTargets == 0)
			continue;

		traverse_inner(state, index, item.blkno, item.offset,
					   item.activeTargets, stack, &stackDepth, maxStack);
	}

	pfree(stack);
}

/*
 * Ergonomic batch search function: bktree_search(table, column, targets, distance)
 */
Datum
bktree_search(PG_FUNCTION_ARGS)
{
	FuncCallContext *funcctx;
	BatchSearchState *state;

	if (SRF_IS_FIRSTCALL())
	{
		MemoryContext oldcontext;
		Oid			tableOid;
		Name		columnName;
		ArrayType  *targetArray;
		int64	   *targets;
		int			ntargets;
		int64		distance;
		Oid			indexOid;
		Relation	index;
		TupleDesc	tupdesc;
		Datum	   *targetDatums;
		bool	   *targetNulls;
		int			i;

		funcctx = SRF_FIRSTCALL_INIT();
		oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

		/* Get arguments */
		tableOid = PG_GETARG_OID(0);
		columnName = PG_GETARG_NAME(1);
		targetArray = PG_GETARG_ARRAYTYPE_P(2);
		distance = PG_GETARG_INT64(3);

		/* Find the SP-GiST index */
		indexOid = find_bktree_index(tableOid, NameStr(*columnName));
		if (!OidIsValid(indexOid))
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_OBJECT),
					 errmsg("no SP-GiST index found on column \"%s\"",
							NameStr(*columnName))));

		/* Deconstruct target array */
		deconstruct_array(targetArray, INT8OID, 8, true, 'd',
						  &targetDatums, &targetNulls, &ntargets);

		if (ntargets > MAX_BATCH_TARGETS)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("too many targets, maximum is %d", MAX_BATCH_TARGETS)));

		if (ntargets == 0)
		{
			/* Empty array - return no results */
			funcctx->user_fctx = NULL;
			MemoryContextSwitchTo(oldcontext);
			SRF_RETURN_DONE(funcctx);
		}

		/* Copy targets to int64 array */
		targets = palloc(sizeof(int64) * ntargets);
		for (i = 0; i < ntargets; i++)
		{
			if (targetNulls[i])
				ereport(ERROR,
						(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
						 errmsg("target array cannot contain NULL values")));
			targets[i] = DatumGetInt64(targetDatums[i]);
		}

		/* Open the index */
		index = index_open(indexOid, AccessShareLock);

		/* Initialize state */
		state = palloc0(sizeof(BatchSearchState));
		state->targets = targets;
		state->ntargets = ntargets;
		state->distance = distance;
		state->max_results = 1024;
		state->results = palloc(sizeof(BatchResult) * state->max_results);
		state->nresults = 0;
		state->index = index;

		/* Do the batch search */
		do_batch_search(state, index);

		/* Build result tuple descriptor */
		tupdesc = CreateTemplateTupleDesc(3);
		TupleDescInitEntry(tupdesc, (AttrNumber) 1, "target_hash", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 2, "match_hash", INT8OID, -1, 0);
		TupleDescInitEntry(tupdesc, (AttrNumber) 3, "match_tid", TIDOID, -1, 0);

		funcctx->tuple_desc = BlessTupleDesc(tupdesc);
		funcctx->user_fctx = state;

		MemoryContextSwitchTo(oldcontext);
	}

	funcctx = SRF_PERCALL_SETUP();
	state = (BatchSearchState *) funcctx->user_fctx;

	if (state == NULL)
		SRF_RETURN_DONE(funcctx);

	if (funcctx->call_cntr < state->nresults)
	{
		Datum		values[3];
		bool		nulls[3] = {false, false, false};
		HeapTuple	tuple;
		BatchResult *res = &state->results[funcctx->call_cntr];

		values[0] = Int64GetDatum(res->target_hash);
		values[1] = Int64GetDatum(res->match_hash);
		values[2] = ItemPointerGetDatum(&res->heap_tid);

		tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);

		SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
	}
	else
	{
		/* Clean up */
		index_close(state->index, AccessShareLock);
		SRF_RETURN_DONE(funcctx);
	}
}
