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
#include "catalog/pg_operator.h"
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

#include "bktree.h"

PG_FUNCTION_INFO_V1(bktree_batch_search);
PG_FUNCTION_INFO_V1(bktree_search);

/* Work item for traversal stack */
typedef struct BatchStackItem
{
	BlockNumber blkno;
	OffsetNumber offset;
	uint64		activeTargets;	/* bitmask of targets still searching this path */
} BatchStackItem;

/* Dynamically sized traversal stack. */
typedef struct BatchTraversalStack
{
	BatchStackItem *items;
	int			depth;
	int			capacity;
} BatchTraversalStack;

#define INITIAL_STACK_CAPACITY 256

/* Cached OIDs for extension types/operators */
static Oid	CachedBktreeAreaTypeOid = InvalidOid;
static Oid	CachedBktreeContainmentOpOid = InvalidOid;

/*
 * Compute hamming distance between two int64 values
 */
static inline int
hamming_distance(int64 a, int64 b)
{
	return __builtin_popcountll((uint64)(a ^ b));
}

/*
 * Get OID of the bktree_area composite type.
 * Cached after first lookup.
 */
Oid
bktree_area_type_oid(void)
{
	if (!OidIsValid(CachedBktreeAreaTypeOid))
	{
		CachedBktreeAreaTypeOid = TypenameGetTypid("bktree_area");
		if (!OidIsValid(CachedBktreeAreaTypeOid))
			elog(ERROR, "type bktree_area not found");
	}
	return CachedBktreeAreaTypeOid;
}

/*
 * Get OID of the <@ (containment) operator for int8/bktree_area.
 * Cached after first lookup.
 */
Oid
bktree_containment_op_oid(void)
{
	if (!OidIsValid(CachedBktreeContainmentOpOid))
	{
		/* Look up operator <@(int8, bktree_area) */
		CachedBktreeContainmentOpOid = OpernameGetOprid(
			list_make1(makeString("<@")),
			INT8OID,
			bktree_area_type_oid());
		if (!OidIsValid(CachedBktreeContainmentOpOid))
			elog(ERROR, "operator <@(int8, bktree_area) not found");
	}
	return CachedBktreeContainmentOpOid;
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

/* Add an item, growing the stack rather than silently dropping work. */
static void
push_stack_item(BatchTraversalStack *stack, BlockNumber blkno,
				OffsetNumber offset, uint64 activeTargets)
{
	BatchStackItem *item;

	if (activeTargets == 0 || !BlockNumberIsValid(blkno))
		return;

	if (stack->depth == stack->capacity)
	{
		stack->capacity *= 2;
		stack->items = repalloc(stack->items,
								  sizeof(BatchStackItem) * stack->capacity);
	}

	item = &stack->items[stack->depth++];
	item->blkno = blkno;
	item->offset = offset;
	item->activeTargets = activeTargets;
}

/* Remove a queued item for a particular block, if one exists. */
static bool
pop_stack_item_for_block(BatchTraversalStack *stack, BlockNumber blkno,
						 BatchStackItem *item)
{
	int			i;

	for (i = stack->depth - 1; i >= 0; i--)
	{
		if (stack->items[i].blkno == blkno)
		{
			*item = stack->items[i];
			stack->depth--;
			if (i != stack->depth)
				stack->items[i] = stack->items[stack->depth];
			return true;
		}
	}

	return false;
}

/*
 * Process an on-page leaf chain.  The caller owns the buffer lock; redirects
 * are queued so other pending work can continue to share the current page.
 */
static void
process_leaf_chain(BatchSearchState *state, Page page,
				   OffsetNumber startOffset, uint64 activeTargets,
				   BatchTraversalStack *stack)
{
	OffsetNumber offset = startOffset;
	OffsetNumber maxOffset = PageGetMaxOffsetNumber(page);

	while (offset != InvalidOffsetNumber)
	{
		SpGistLeafTuple leafTuple;
		ItemId		itemId;
		int64		leafHash;
		uint64		remainingTargets;

		if (offset < FirstOffsetNumber || offset > maxOffset)
			break;

		itemId = PageGetItemId(page, offset);
		if (!ItemIdIsUsed(itemId))
			break;

		leafTuple = (SpGistLeafTuple) PageGetItem(page, itemId);

		if (leafTuple->tupstate != SPGIST_LIVE)
		{
			if (leafTuple->tupstate == SPGIST_REDIRECT)
			{
				SpGistDeadTuple dt = (SpGistDeadTuple) leafTuple;

				push_stack_item(stack,
							ItemPointerGetBlockNumber(&dt->pointer),
							ItemPointerGetOffsetNumber(&dt->pointer),
							activeTargets);
				return;
			}
			offset = SGLT_GET_NEXTOFFSET(leafTuple);
			continue;
		}

		leafHash = DatumGetInt64(SGLTDATUM(leafTuple, &state->spgstate));

		/* Check only targets whose bits remain active on this path. */
		remainingTargets = activeTargets;
		while (remainingTargets != 0)
		{
			int			i = __builtin_ctzll(remainingTargets);
			int			dist = hamming_distance(leafHash, state->targets[i]);

			if (dist <= state->distance)
				add_result(state, state->targets[i], leafHash,
						   &leafTuple->heapPtr);

			remainingTargets &= remainingTargets - 1;
		}

		offset = SGLT_GET_NEXTOFFSET(leafTuple);
	}
}

/* Process one inner tuple while the caller holds the page buffer lock. */
static void
process_inner_tuple(BatchSearchState *state, Page page, OffsetNumber offset,
					uint64 activeTargets, BatchTraversalStack *stack)
{
	SpGistInnerTuple innerTuple;
	ItemId		itemId;
	int64		prefix;
	uint64		remainingTargets;
	SpGistNodeTuple node;
	int			nodeN;
	uint64		nodeTargets[65];
	OffsetNumber maxOffset = PageGetMaxOffsetNumber(page);

	if (activeTargets == 0 || offset < FirstOffsetNumber || offset > maxOffset)
		return;

	itemId = PageGetItemId(page, offset);
	if (!ItemIdIsUsed(itemId))
		return;

	innerTuple = (SpGistInnerTuple) PageGetItem(page, itemId);

	if (innerTuple->tupstate != SPGIST_LIVE)
	{
		if (innerTuple->tupstate == SPGIST_REDIRECT)
		{
			SpGistDeadTuple dt = (SpGistDeadTuple) innerTuple;

			push_stack_item(stack,
						ItemPointerGetBlockNumber(&dt->pointer),
						ItemPointerGetOffsetNumber(&dt->pointer),
						activeTargets);
		}
		return;
	}

	memset(nodeTargets, 0, sizeof(nodeTargets));

	if (innerTuple->allTheSame)
	{
		SGITITERATE(innerTuple, nodeN, node)
		{
			push_stack_item(stack,
							ItemPointerGetBlockNumber(&node->t_tid),
							ItemPointerGetOffsetNumber(&node->t_tid),
							activeTargets);
		}
		return;
	}

	prefix = DatumGetInt64(SGITDATUM(innerTuple, &state->spgstate));

	remainingTargets = activeTargets;
	while (remainingTargets != 0)
	{
		int			i = __builtin_ctzll(remainingTargets);
		uint64		mask = UINT64CONST(1) << i;
		int			dist = hamming_distance(prefix, state->targets[i]);
		int			minDist = (dist > state->distance) ?
			(dist - state->distance) : 0;
		int			maxDist = (dist + state->distance > 64) ?
			64 : (dist + state->distance);
		int			d;

		for (d = minDist; d <= maxDist; d++)
			nodeTargets[d] |= mask;

		remainingTargets &= remainingTargets - 1;
	}

	SGITITERATE(innerTuple, nodeN, node)
	{
		if (nodeN < 65 && nodeTargets[nodeN] != 0)
		{
			push_stack_item(stack,
							ItemPointerGetBlockNumber(&node->t_tid),
							ItemPointerGetOffsetNumber(&node->t_tid),
							nodeTargets[nodeN]);
		}
	}
}

/* Process all currently queued work for one block under one buffer lock. */
static void
process_block(BatchSearchState *state, BatchTraversalStack *stack,
			  BatchStackItem item)
{
	Buffer		buffer;
	Page		page;
	BlockNumber blkno = item.blkno;

	buffer = ReadBuffer(state->index, blkno);
	LockBuffer(buffer, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buffer);

	do
	{
		if (SpGistPageIsLeaf(page))
			process_leaf_chain(state, page, item.offset,
							   item.activeTargets, stack);
		else
			process_inner_tuple(state, page, item.offset,
								item.activeTargets, stack);
	} while (pop_stack_item_for_block(stack, blkno, &item));

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

		bktree_batch_execute(state);

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
 * Core batch search logic - exported for use by CustomScan
 *
 * Caller must have initialized:
 *   - state->targets, state->ntargets, state->distance
 *   - state->index (opened with at least AccessShareLock)
 *   - state->results, state->max_results (pre-allocated)
 *   - state->nresults = 0
 *
 * On return, state->results contains all matches and state->nresults
 * indicates how many were found.
 */
void
bktree_batch_execute(BatchSearchState *state)
{
	BatchTraversalStack stack;
	uint64		activeTargets;

	if (state->ntargets < 0 || state->ntargets > MAX_BATCH_TARGETS)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("too many targets, maximum is %d", MAX_BATCH_TARGETS)));

	if (state->ntargets == 0)
		return;

	initSpGistState(&state->spgstate, state->index);

	stack.capacity = INITIAL_STACK_CAPACITY;
	stack.depth = 0;
	stack.items = palloc(sizeof(BatchStackItem) * stack.capacity);

	/* Start with root */
	activeTargets = (state->ntargets == MAX_BATCH_TARGETS) ?
		UINT64_MAX : ((UINT64CONST(1) << state->ntargets) - 1);
	push_stack_item(&stack, SPGIST_ROOT_BLKNO, FirstOffsetNumber,
					activeTargets);

	while (stack.depth > 0)
	{
		BatchStackItem item;

		item = stack.items[--stack.depth];
		process_block(state, &stack, item);
	}

	pfree(stack.items);
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
		bktree_batch_execute(state);

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
