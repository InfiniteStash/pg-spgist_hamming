/*
 * bktree_customscan.c - CustomScan join optimization for BK-tree batch search
 *
 * This module intercepts the planner's join path generation to detect the
 * common UNNEST + bktree JOIN pattern and inject a CustomPath that uses
 * batch traversal instead of N separate index scans.
 *
 * Target pattern:
 *   SELECT ... FROM UNNEST($1::int8[]) AS target
 *   JOIN table t ON t.hash <@ (target, distance)::bktree_area
 *
 * Instead of N nested loop iterations with N index scans, we collect all
 * targets first and do a single batch traversal of the index.
 */

#include "postgres.h"

#include <math.h>

#include "access/heapam.h"
#include "access/htup_details.h"
#include "access/relscan.h"
#include "access/spgist_private.h"
#include "access/tableam.h"
#include "access/genam.h"
#include "access/table.h"
#include "catalog/indexing.h"
#include "catalog/pg_class.h"
#include "catalog/pg_operator.h"
#include "catalog/pg_opclass.h"
#include "executor/executor.h"
#include "executor/nodeCustom.h"
#include "executor/tuptable.h"
#include "nodes/extensible.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/planmain.h"
#include "optimizer/restrictinfo.h"
#include "parser/parsetree.h"
#include "commands/explain.h"
#include "lib/stringinfo.h"
#include "storage/bufmgr.h"
#include "utils/builtins.h"
#include "utils/fmgroids.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "utils/syscache.h"

#include "bktree.h"

/* Forward declarations */
static void bktree_join_pathlist_hook(PlannerInfo *root,
									  RelOptInfo *joinrel,
									  RelOptInfo *outerrel,
									  RelOptInfo *innerrel,
									  JoinType jointype,
									  JoinPathExtraData *extra);

/* Previous hook for chaining */
static set_join_pathlist_hook_type prev_join_hook = NULL;

/*
 * Information extracted from a matching join pattern
 */
typedef struct BktreeBatchJoinInfo
{
	Oid			indexOid;		/* OID of the bktree SP-GiST index */
	AttrNumber	indexedAttno;	/* attribute number of indexed column */
	Expr	   *distanceExpr;	/* expression for distance value */
	Var		   *outerVar;		/* the UNNEST variable */
	RestrictInfo *joinClause;	/* the <@ join condition */
	Index		innerRelid;		/* inner relation's relid (for var matching) */
} BktreeBatchJoinInfo;

/*
 * Custom path for batch join
 */
typedef struct BktreeBatchJoinPath
{
	CustomPath	cpath;

	Path	   *outerpath;		/* Path for UNNEST result */
	Oid			indexOid;		/* Bktree index on inner table */
	AttrNumber	indexedAttno;	/* Indexed column */
	Expr	   *distanceExpr;	/* Distance value expression */
	Var		   *outerVar;		/* Variable from outer (for projection) */
	Index		innerRelid;		/* Inner relation's relid */
} BktreeBatchJoinPath;

/*
 * Column mapping entry - describes where each output column comes from
 */
typedef struct ColumnMapping
{
	Index		varno;			/* Source relation (innerRelid or outerRelid, or -1) */
	AttrNumber	varattno;		/* Attribute number in source */
} ColumnMapping;

/*
 * Private data stored in CustomScan->custom_private.
 * Stored as a List for serialization, but accessed via helper functions.
 *
 * List structure:
 *   [0] = indexOid (Const OID)
 *   [1] = indexedAttno (Const INT2)
 *   [2] = innerRelid (Const INT4)
 *   [3] = outerVarAttno (Const INT2)
 *   [4] = distanceExpr (Expr node)
 *   [5] = outerRelid (Const INT4)
 *   [6..] = column mappings as pairs of (varno INT4, varattno INT2)
 */
#define PRIVATE_INDEX_OID		0
#define PRIVATE_INDEXED_ATTNO	1
#define PRIVATE_INNER_RELID		2
#define PRIVATE_OUTER_VAR_ATTNO	3
#define PRIVATE_DISTANCE_EXPR	4
#define PRIVATE_OUTER_RELID		5
#define PRIVATE_COL_MAPPINGS	6	/* Start of column mapping pairs */

/* Helper to extract Const value from custom_private */
static inline Datum
private_get_datum(List *private, int index)
{
	Const *c = (Const *) list_nth(private, index);
	Assert(IsA(c, Const) && !c->constisnull);
	return c->constvalue;
}

#define PRIVATE_GET_OID(priv, idx)		DatumGetObjectId(private_get_datum(priv, idx))
#define PRIVATE_GET_INT32(priv, idx)	DatumGetInt32(private_get_datum(priv, idx))
#define PRIVATE_GET_INT16(priv, idx)	DatumGetInt16(private_get_datum(priv, idx))

/*
 * Custom scan state for execution
 */
typedef struct BktreeBatchJoinState
{
	CustomScanState css;

	/* Outer scan state */
	PlanState  *outerPlan;

	/* Batch search state */
	Relation	index;
	int64	   *targets;
	int			ntargets;
	int			max_targets;
	int64		distance;
	ExprState  *distanceExprState;

	/* Results - allocated in batchContext for easy reset on rescan */
	MemoryContext batchContext;
	BatchResult *results;
	int			nresults;
	int			curr_result;
	bool		search_done;

	/* For heap tuple fetch */
	Relation	innerRel;
	TupleTableSlot *innerSlot;	/* Slot for fetched heap tuples */
	Oid			indexOid;
	AttrNumber	indexedAttno;

	/* Outer var attno for extracting target value */
	AttrNumber	outerVarAttno;

	/* Relation indices */
	Index		innerRelid;		/* Inner relation's RT index */
	Index		outerRelid;		/* Outer relation's RT index */

	/* Column mappings - how to build output tuple */
	ColumnMapping *colMappings;	/* Array of column mappings */
	int			ncolMappings;	/* Number of output columns */
} BktreeBatchJoinState;

/* CustomScan callbacks */
static Plan *bktree_batch_join_plan(PlannerInfo *root,
									RelOptInfo *rel,
									CustomPath *best_path,
									List *tlist,
									List *clauses,
									List *custom_plans);
static Node *bktree_batch_join_create_state(CustomScan *cscan);
static void bktree_batch_join_begin(CustomScanState *node,
									EState *estate,
									int eflags);
static TupleTableSlot *bktree_batch_join_exec(CustomScanState *node);
static void bktree_batch_join_end(CustomScanState *node);
static void bktree_batch_join_rescan(CustomScanState *node);
static void bktree_batch_join_explain(CustomScanState *node,
									  List *ancestors,
									  ExplainState *es);

/* CustomPath methods */
static const CustomPathMethods bktree_batch_join_path_methods = {
	.CustomName = "BktreeBatchJoin",
	.PlanCustomPath = bktree_batch_join_plan,
};

/* CustomScan methods */
static const CustomScanMethods bktree_batch_join_scan_methods = {
	.CustomName = "BktreeBatchJoin",
	.CreateCustomScanState = bktree_batch_join_create_state,
};

/* CustomExec methods */
static const CustomExecMethods bktree_batch_join_exec_methods = {
	.CustomName = "BktreeBatchJoin",
	.BeginCustomScan = bktree_batch_join_begin,
	.ExecCustomScan = bktree_batch_join_exec,
	.EndCustomScan = bktree_batch_join_end,
	.ReScanCustomScan = bktree_batch_join_rescan,
	.ExplainCustomScan = bktree_batch_join_explain,
};

/*
 * Register the set_join_pathlist_hook.
 * Called from _PG_init().
 */
void
bktree_register_hooks(void)
{
	/* Register custom scan provider */
	RegisterCustomScanMethods(&bktree_batch_join_scan_methods);

	/* Install join hook */
	prev_join_hook = set_join_pathlist_hook;
	set_join_pathlist_hook = bktree_join_pathlist_hook;
}

/*
 * Check if a relation is a FunctionScan on UNNEST.
 */
static bool
is_unnest_function_scan(PlannerInfo *root, RelOptInfo *rel)
{
	RangeTblEntry *rte;

	if (rel->rtekind != RTE_FUNCTION)
		return false;

	rte = planner_rt_fetch(rel->relid, root);
	if (rte->rtekind != RTE_FUNCTION)
		return false;

	/* Check if it's the UNNEST function */
	if (list_length(rte->functions) == 1)
	{
		RangeTblFunction *rtfunc = linitial(rte->functions);
		if (IsA(rtfunc->funcexpr, FuncExpr))
		{
			FuncExpr *fexpr = (FuncExpr *) rtfunc->funcexpr;
			char *funcname = get_func_name(fexpr->funcid);
			if (funcname && strcmp(funcname, "unnest") == 0)
			{
				pfree(funcname);
				return true;
			}
			if (funcname)
				pfree(funcname);
		}
	}
	return false;
}

/*
 * Find a bktree SP-GiST index on the given relation.
 * Returns InvalidOid if not found.
 */
static Oid
find_bktree_index_for_rel(RelOptInfo *rel, AttrNumber *indexedAttno)
{
	ListCell *lc;

	foreach(lc, rel->indexlist)
	{
		IndexOptInfo *index = lfirst(lc);

		/* Must be SP-GiST */
		if (index->relam != SPGIST_AM_OID)
			continue;

		/* Must have at least one column */
		if (index->nkeycolumns < 1)
			continue;

		/*
		 * Check if it uses bktree_ops by looking up the opclass from
		 * opfamily and opcintype. IndexOptInfo doesn't store opclass OID
		 * directly, so we query pg_opclass.
		 */
		{
			HeapTuple	opcTuple;
			Form_pg_opclass opcForm;
			SysScanDesc scan;
			ScanKeyData skey[2];
			Relation	opcRel;
			bool		found = false;

			opcRel = table_open(OperatorClassRelationId, AccessShareLock);

			ScanKeyInit(&skey[0],
						Anum_pg_opclass_opcfamily,
						BTEqualStrategyNumber, F_OIDEQ,
						ObjectIdGetDatum(index->opfamily[0]));
			ScanKeyInit(&skey[1],
						Anum_pg_opclass_opcintype,
						BTEqualStrategyNumber, F_OIDEQ,
						ObjectIdGetDatum(index->opcintype[0]));

			scan = systable_beginscan(opcRel, OpclassAmNameNspIndexId, false,
									  NULL, 2, skey);

			while ((opcTuple = systable_getnext(scan)) != NULL)
			{
				opcForm = (Form_pg_opclass) GETSTRUCT(opcTuple);
				if (strcmp(NameStr(opcForm->opcname), "bktree_ops") == 0)
				{
					found = true;
					break;
				}
			}

			systable_endscan(scan);
			table_close(opcRel, AccessShareLock);

			if (!found)
				continue;
		}

		/* Found a bktree index */
		*indexedAttno = index->indexkeys[0];
		return index->indexoid;
	}

	*indexedAttno = InvalidAttrNumber;
	return InvalidOid;
}

/*
 * Check if a clause matches: indexed_col <@ (outer_var, const)::bktree_area
 * or similar patterns with the containment operator.
 */
/*
 * Helper to extract outer var and distance from a RowExpr.
 */
static bool
extract_from_rowexpr(RowExpr *rowexpr, RelOptInfo *outerrel,
					 Var **outerVar, Expr **distanceExpr)
{
	Node	   *first;
	Node	   *second;
	Var		   *testVar;

	if (list_length(rowexpr->args) != 2)
		return false;

	first = linitial(rowexpr->args);
	second = lsecond(rowexpr->args);

	/* First element should be a Var from outerrel */
	while (IsA(first, RelabelType))
		first = (Node *) ((RelabelType *) first)->arg;

	if (!IsA(first, Var))
		return false;

	testVar = (Var *) first;
	if (!bms_is_member(testVar->varno, outerrel->relids))
		return false;

	*outerVar = testVar;
	*distanceExpr = (Expr *) second;
	return true;
}

static bool
match_bktree_join_clause(PlannerInfo *root,
						 RestrictInfo *rinfo,
						 RelOptInfo *innerrel,
						 RelOptInfo *outerrel,
						 AttrNumber indexedAttno,
						 Expr **distanceExpr,
						 Var **outerVar)
{
	OpExpr	   *op;
	Oid			opno;
	Node	   *leftarg;
	Node	   *rightarg;
	Var		   *indexedVar;
	Oid			bktree_area_oid;
	RowExpr	   *rowexpr;

	/* Must be an operator expression */
	if (!IsA(rinfo->clause, OpExpr))
	{
		return false;
	}

	op = (OpExpr *) rinfo->clause;
	opno = op->opno;

	/* Check if it's the <@ operator */
	if (opno != bktree_containment_op_oid())
	{
		return false;
	}

	/* Should have exactly 2 arguments */
	if (list_length(op->args) != 2)
		return false;

	leftarg = linitial(op->args);
	rightarg = lsecond(op->args);

	/*
	 * Left arg should be a Var from innerrel matching the indexed column.
	 */
	if (!IsA(leftarg, Var))
	{
		return false;
	}

	indexedVar = (Var *) leftarg;
	if (!bms_is_member(indexedVar->varno, innerrel->relids))
	{
		return false;
	}
	if (indexedVar->varattno != indexedAttno)
	{
		return false;
	}

	/*
	 * Right arg should be a bktree_area. It could be:
	 *  - A RowExpr with (outer_var, distance)
	 *  - A cast/coerce to bktree_area type
	 *
	 * We need to extract the outer var and distance from it.
	 */
	bktree_area_oid = bktree_area_type_oid();

	/* Strip any RelabelType nodes */
	while (IsA(rightarg, RelabelType))
		rightarg = (Node *) ((RelabelType *) rightarg)->arg;

	/* Check for RowExpr (explicit tuple construction) */
	if (IsA(rightarg, RowExpr))
	{
		rowexpr = (RowExpr *) rightarg;
		return extract_from_rowexpr(rowexpr, outerrel, outerVar, distanceExpr);
	}

	/* Check for CoerceViaIO or other coercion of a row expr */
	if (IsA(rightarg, CoerceViaIO))
	{
		CoerceViaIO *coerce = (CoerceViaIO *) rightarg;
		if (coerce->resulttype == bktree_area_oid && IsA(coerce->arg, RowExpr))
		{
			rowexpr = (RowExpr *) coerce->arg;
			return extract_from_rowexpr(rowexpr, outerrel, outerVar, distanceExpr);
		}
	}
	return false;
}

/*
 * Check if this join matches the UNNEST + bktree pattern.
 */
static bool
is_bktree_batch_join(PlannerInfo *root,
					 RelOptInfo *outerrel,
					 RelOptInfo *innerrel,
					 List *restrictlist,
					 BktreeBatchJoinInfo *info)
{
	ListCell   *lc;
	Oid			indexOid;
	AttrNumber	indexedAttno;

	/* Check outer is UNNEST function scan */
	if (!is_unnest_function_scan(root, outerrel))
	{
		return false;
	}

	/* Check inner has bktree index */
	indexOid = find_bktree_index_for_rel(innerrel, &indexedAttno);
	if (!OidIsValid(indexOid))
	{
		return false;
	}

	info->indexOid = indexOid;
	info->indexedAttno = indexedAttno;
	info->innerRelid = innerrel->relid;

	/* Check join condition matches pattern */
	foreach(lc, restrictlist)
	{
		RestrictInfo *rinfo = lfirst(lc);

		if (match_bktree_join_clause(root, rinfo, innerrel, outerrel,
									 indexedAttno, &info->distanceExpr,
									 &info->outerVar))
		{
			info->joinClause = rinfo;
			return true;
		}
	}
	return false;
}

/*
 * Compute costs for the batch join path.
 */
static void
cost_bktree_batch_join(PlannerInfo *root,
					   BktreeBatchJoinPath *path,
					   RelOptInfo *joinrel,
					   RelOptInfo *outerrel,
					   RelOptInfo *innerrel)
{
	Cost		startup_cost = 0;
	Cost		run_cost = 0;
	double		outer_rows = outerrel->rows;
	double		index_pages;
	double		result_rows;

	/* Must materialize outer first */
	startup_cost += path->outerpath->total_cost;

	/*
	 * Batch search cost model:
	 *
	 * The key insight is that we traverse the index ONCE for all targets,
	 * vs N separate traversals for nested loop. At each inner node, we
	 * check all N targets, but this is just N cheap hamming distance
	 * calculations - much cheaper than N separate I/O operations.
	 *
	 * For SP-GiST BK-tree, the tree is shallow (log base ~65 of tuples).
	 * A single traversal visits far fewer pages than N separate scans.
	 */
	index_pages = Max(5.0, log(innerrel->tuples + 1) * 2);

	/*
	 * Index I/O cost: single traversal for all targets.
	 * This is the main advantage - N targets share the same traversal cost.
	 * Use seq_page_cost since traversal is mostly sequential within the index.
	 */
	run_cost += index_pages * seq_page_cost;

	/* CPU cost: at each node, check all targets (cheap hamming distance) */
	run_cost += index_pages * outer_rows * cpu_operator_cost * 0.1;

	/*
	 * Use the planner's row estimate for this join - ensures consistency
	 * with other paths so the planner can make fair cost comparisons.
	 */
	result_rows = joinrel->rows;

	/*
	 * Heap tuple fetches. Results are often clustered (similar hashes near
	 * each other in heap), so use a reduced I/O cost. Also, with batch
	 * fetching we can amortize costs better than per-row nested loop.
	 */
	run_cost += result_rows * seq_page_cost * 0.5;
	run_cost += result_rows * cpu_tuple_cost;

	path->cpath.path.startup_cost = startup_cost;
	path->cpath.path.total_cost = startup_cost + run_cost;
	path->cpath.path.rows = result_rows;
}

/*
 * Create the custom join path.
 */
static void
create_bktree_batch_join_path(PlannerInfo *root,
							  RelOptInfo *joinrel,
							  RelOptInfo *outerrel,
							  RelOptInfo *innerrel,
							  BktreeBatchJoinInfo *info)
{
	BktreeBatchJoinPath *pathnode;
	Path	   *outerpath;

	/* Get cheapest outer path */
	outerpath = outerrel->cheapest_total_path;
	if (outerpath == NULL)
	{
		return;
	}

	/* Limit to MAX_BATCH_TARGETS rows */
	if (outerrel->rows > MAX_BATCH_TARGETS)
	{
		return;
	}

	/* Create the custom path */
	pathnode = palloc0(sizeof(BktreeBatchJoinPath));

	pathnode->cpath.path.type = T_CustomPath;
	pathnode->cpath.path.pathtype = T_CustomScan;
	pathnode->cpath.path.parent = joinrel;
	pathnode->cpath.path.pathtarget = joinrel->reltarget;
	pathnode->cpath.path.param_info = NULL;  /* No parameterization for now */
	pathnode->cpath.path.parallel_aware = false;
	pathnode->cpath.path.parallel_safe = false;
	pathnode->cpath.path.parallel_workers = 0;
	pathnode->cpath.path.pathkeys = NIL;  /* Unordered output */

	pathnode->cpath.flags = 0;
	pathnode->cpath.custom_paths = list_make1(outerpath);
	pathnode->cpath.custom_private = NIL;
	pathnode->cpath.methods = &bktree_batch_join_path_methods;

	pathnode->outerpath = outerpath;
	pathnode->indexOid = info->indexOid;
	pathnode->indexedAttno = info->indexedAttno;
	pathnode->distanceExpr = info->distanceExpr;
	pathnode->outerVar = info->outerVar;
	pathnode->innerRelid = info->innerRelid;

	/* Compute costs - use joinrel->rows for consistent row estimate */
	cost_bktree_batch_join(root, pathnode, joinrel, outerrel, innerrel);

	/* Add the path to the joinrel */
	add_path(joinrel, &pathnode->cpath.path);
}

/*
 * Join hook - called during join path generation.
 */
static void
bktree_join_pathlist_hook(PlannerInfo *root,
						  RelOptInfo *joinrel,
						  RelOptInfo *outerrel,
						  RelOptInfo *innerrel,
						  JoinType jointype,
						  JoinPathExtraData *extra)
{
	BktreeBatchJoinInfo info;

	/* Call previous hook first */
	if (prev_join_hook)
		prev_join_hook(root, joinrel, outerrel, innerrel, jointype, extra);

	/* Only optimize inner joins for now */
	if (jointype != JOIN_INNER)
		return;

	/* Check if this matches our pattern */
	if (!is_bktree_batch_join(root, outerrel, innerrel,
							  extra->restrictlist, &info))
	{
		RelOptInfo *tmp;

		/* Also try swapped order (inner/outer might be reversed) */
		if (!is_bktree_batch_join(root, innerrel, outerrel,
								  extra->restrictlist, &info))
			return;

		/* Swapped: outer becomes inner and vice versa */
		tmp = outerrel;
		outerrel = innerrel;
		innerrel = tmp;
	}

	/* Create and add custom join path */
	create_bktree_batch_join_path(root, joinrel, outerrel, innerrel, &info);
}

/*
 * Plan creation - convert CustomPath to CustomScan.
 *
 * For CustomScan with scanrelid=0, we need custom_scan_tlist to describe
 * the output using INDEX_VAR references. The actual targetlist can then
 * reference this via Vars with varno=INDEX_VAR.
 */
static Plan *
bktree_batch_join_plan(PlannerInfo *root,
					   RelOptInfo *rel,
					   CustomPath *best_path,
					   List *tlist,
					   List *clauses,
					   List *custom_plans)
{
	BktreeBatchJoinPath *path = (BktreeBatchJoinPath *) best_path;
	CustomScan *cscan;
	Plan	   *outerPlan;
	List	   *scan_tlist = NIL;
	ListCell   *lc;
	int			resno;

	/* Get the outer plan */
	outerPlan = linitial(custom_plans);

	cscan = makeNode(CustomScan);
	cscan->scan.plan.qual = NIL;  /* We don't have post-scan filters */
	cscan->scan.scanrelid = 0;	  /* Not scanning a single relation */
	cscan->flags = best_path->flags;

	/*
	 * Store custom private data as List. See PRIVATE_* constants for layout.
	 * The distanceExpr must be copied since it comes from the planner.
	 */
	cscan->custom_private = list_make1(			/* PRIVATE_INDEX_OID */
		makeConst(OIDOID, -1, InvalidOid, sizeof(Oid),
				  ObjectIdGetDatum(path->indexOid), false, true));
	cscan->custom_private = lappend(cscan->custom_private,	/* PRIVATE_INDEXED_ATTNO */
		makeConst(INT2OID, -1, InvalidOid, sizeof(int16),
				  Int16GetDatum(path->indexedAttno), false, true));
	cscan->custom_private = lappend(cscan->custom_private,	/* PRIVATE_INNER_RELID */
		makeConst(INT4OID, -1, InvalidOid, sizeof(int32),
				  Int32GetDatum(path->innerRelid), false, true));
	cscan->custom_private = lappend(cscan->custom_private,	/* PRIVATE_OUTER_VAR_ATTNO */
		makeConst(INT2OID, -1, InvalidOid, sizeof(int16),
				  Int16GetDatum(path->outerVar->varattno), false, true));
	cscan->custom_private = lappend(cscan->custom_private,	/* PRIVATE_DISTANCE_EXPR */
		copyObject(path->distanceExpr));
	cscan->custom_private = lappend(cscan->custom_private,	/* PRIVATE_OUTER_RELID */
		makeConst(INT4OID, -1, InvalidOid, sizeof(int32),
				  Int32GetDatum(path->outerVar->varno), false, true));

	/*
	 * Build custom_scan_tlist - this describes what our scan produces.
	 * Like postgres_fdw, we keep the original Vars with their original
	 * varnos. setrefs will use this to build an indexed_tlist and convert
	 * the plan's targetlist Vars to INDEX_VAR references.
	 *
	 * We store the original varnos in custom_private so exec knows
	 * which slot to get each column from.
	 */
	resno = 1;
	foreach(lc, tlist)
	{
		TargetEntry *tle = lfirst(lc);
		TargetEntry *scan_tle;

		/* Copy the original expression into custom_scan_tlist */
		scan_tle = makeTargetEntry(copyObject(tle->expr),
								   resno,
								   tle->resname,
								   tle->resjunk);
		scan_tlist = lappend(scan_tlist, scan_tle);

		/* Store varno info for this column in custom_private */
		if (IsA(tle->expr, Var))
		{
			Var *var = (Var *) tle->expr;
			cscan->custom_private = lappend(cscan->custom_private,
				makeConst(INT4OID, -1, InvalidOid, sizeof(int32),
						  Int32GetDatum(var->varno), false, true));
			cscan->custom_private = lappend(cscan->custom_private,
				makeConst(INT2OID, -1, InvalidOid, sizeof(int16),
						  Int16GetDatum(var->varattno), false, true));
		}
		else
		{
			/* Non-Var expression - mark as special */
			cscan->custom_private = lappend(cscan->custom_private,
				makeConst(INT4OID, -1, InvalidOid, sizeof(int32),
						  Int32GetDatum(-1), false, true));
			cscan->custom_private = lappend(cscan->custom_private,
				makeConst(INT2OID, -1, InvalidOid, sizeof(int16),
						  Int16GetDatum(0), false, true));
		}
		resno++;
	}

	/*
	 * Set plan targetlist to the original tlist (with original varnos).
	 * setrefs will find these Vars in custom_scan_tlist and convert them
	 * to INDEX_VAR references automatically.
	 */
	cscan->scan.plan.targetlist = copyObject(tlist);
	cscan->custom_scan_tlist = scan_tlist;
	cscan->custom_plans = list_make1(outerPlan);
	cscan->methods = &bktree_batch_join_scan_methods;

	return (Plan *) cscan;
}

/*
 * Create execution state.
 */
static Node *
bktree_batch_join_create_state(CustomScan *cscan)
{
	BktreeBatchJoinState *state;

	state = (BktreeBatchJoinState *) newNode(sizeof(BktreeBatchJoinState),
										   T_CustomScanState);
	state->css.methods = &bktree_batch_join_exec_methods;

	return (Node *) state;
}

/*
 * Begin execution.
 */
static void
bktree_batch_join_begin(CustomScanState *node, EState *estate, int eflags)
{
	BktreeBatchJoinState *state = (BktreeBatchJoinState *) node;
	CustomScan *cscan = (CustomScan *) node->ss.ps.plan;
	List	   *priv = cscan->custom_private;
	Plan	   *outerPlan;
	Expr	   *distanceExpr;
	List	   *tlist;
	int			ncolumns;
	int			i;
	int			privIdx;

	/* Extract fixed private data using named indices */
	state->indexOid = PRIVATE_GET_OID(priv, PRIVATE_INDEX_OID);
	state->indexedAttno = PRIVATE_GET_INT16(priv, PRIVATE_INDEXED_ATTNO);
	state->innerRelid = PRIVATE_GET_INT32(priv, PRIVATE_INNER_RELID);
	state->outerVarAttno = PRIVATE_GET_INT16(priv, PRIVATE_OUTER_VAR_ATTNO);
	state->outerRelid = PRIVATE_GET_INT32(priv, PRIVATE_OUTER_RELID);
	distanceExpr = (Expr *) list_nth(priv, PRIVATE_DISTANCE_EXPR);

	/*
	 * Extract column mappings from custom_private.
	 * After the fixed entries, we have pairs of (varno INT4, varattno INT2).
	 */
	tlist = cscan->scan.plan.targetlist;
	ncolumns = list_length(tlist);
	state->ncolMappings = ncolumns;
	state->colMappings = palloc(sizeof(ColumnMapping) * ncolumns);

	privIdx = PRIVATE_COL_MAPPINGS;
	for (i = 0; i < ncolumns; i++)
	{
		state->colMappings[i].varno = PRIVATE_GET_INT32(priv, privIdx);
		state->colMappings[i].varattno = PRIVATE_GET_INT16(priv, privIdx + 1);
		privIdx += 2;
	}

	/* Compile distance expression */
	state->distanceExprState = ExecInitExpr(distanceExpr, &node->ss.ps);

	/* Open index */
	state->index = index_open(state->indexOid, AccessShareLock);

	/* Open inner relation (for heap fetches) */
	{
		RangeTblEntry *rte = exec_rt_fetch(state->innerRelid, estate);
		state->innerRel = table_open(rte->relid, AccessShareLock);
		/* Create a slot suitable for this relation's tuples */
		state->innerSlot = table_slot_create(state->innerRel,
											 &estate->es_tupleTable);
	}

	/* Initialize outer plan */
	outerPlan = linitial(cscan->custom_plans);
	state->outerPlan = ExecInitNode(outerPlan, estate, eflags);

	/* Create memory context for batch data (targets and results) */
	state->batchContext = AllocSetContextCreate(estate->es_query_cxt,
												"BktreeBatchJoin",
												ALLOCSET_DEFAULT_SIZES);

	/* Initialize batch state - allocate in batch context */
	state->max_targets = MAX_BATCH_TARGETS;
	state->targets = MemoryContextAlloc(state->batchContext,
										sizeof(int64) * state->max_targets);
	state->ntargets = 0;
	state->distance = 0;

	state->results = NULL;
	state->nresults = 0;
	state->curr_result = 0;
	state->search_done = false;
}

/*
 * Collect all targets from outer plan.
 */
static void
collect_targets(BktreeBatchJoinState *state)
{
	TupleTableSlot *slot;
	ExprContext *econtext = state->css.ss.ps.ps_ExprContext;

	while ((slot = ExecProcNode(state->outerPlan)) != NULL &&
		   !TupIsNull(slot))
	{
		Datum		val;
		bool		isnull;

		/* Extract the target hash value */
		val = slot_getattr(slot, state->outerVarAttno, &isnull);
		if (isnull)
			continue;

		/* Grow targets array if needed */
		if (state->ntargets >= state->max_targets)
		{
			int		new_max = state->max_targets * 2;
			int64  *new_targets;

			new_targets = MemoryContextAlloc(state->batchContext,
											 sizeof(int64) * new_max);
			memcpy(new_targets, state->targets,
				   sizeof(int64) * state->ntargets);
			/* Old array will be freed when context is reset */
			state->targets = new_targets;
			state->max_targets = new_max;
		}

		state->targets[state->ntargets++] = DatumGetInt64(val);

		/* Evaluate distance expression (usually a constant) */
		if (state->ntargets == 1)
		{
			econtext->ecxt_scantuple = slot;
			val = ExecEvalExpr(state->distanceExprState, econtext, &isnull);
			if (isnull)
				ereport(ERROR,
						(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
						 errmsg("distance cannot be NULL")));
			state->distance = DatumGetInt64(val);
		}
	}

}

/*
 * Execute batch search.
 */
static void
execute_batch_search(BktreeBatchJoinState *state)
{
	BatchSearchState bss;
	MemoryContext oldcxt;

	if (state->ntargets == 0)
	{
		state->results = NULL;
		state->nresults = 0;
		return;
	}

	/* Allocate results in batch context for easy cleanup on rescan */
	oldcxt = MemoryContextSwitchTo(state->batchContext);

	/* Initialize batch search state */
	memset(&bss, 0, sizeof(bss));
	bss.targets = state->targets;
	bss.ntargets = state->ntargets;
	bss.distance = state->distance;
	bss.index = state->index;
	bss.max_results = 1024;
	bss.results = palloc(sizeof(BatchResult) * bss.max_results);
	bss.nresults = 0;

	MemoryContextSwitchTo(oldcxt);

	/* Execute the batch search */
	bktree_batch_execute(&bss);

	/* Transfer results to our state */
	state->results = bss.results;
	state->nresults = bss.nresults;
}

/*
 * Fetch next tuple from our batch results.
 */
static TupleTableSlot *
bktree_batch_join_next(CustomScanState *node)
{
	BktreeBatchJoinState *state = (BktreeBatchJoinState *) node;
	TupleTableSlot *scanSlot = node->ss.ps.ps_ResultTupleSlot;
	Snapshot	snapshot;

	/* First call: collect targets and execute batch search */
	if (!state->search_done)
	{
		collect_targets(state);
		execute_batch_search(state);
		state->search_done = true;
		state->curr_result = 0;
	}

	snapshot = GetActiveSnapshot();

	/* Return next result */
	while (state->curr_result < state->nresults)
	{
		BatchResult *r = &state->results[state->curr_result++];
		bool		found;
		int			i;

		/*
		 * Fetch the heap tuple using the TID from the batch result.
		 * We need to verify it's still visible.
		 */
		found = table_tuple_fetch_row_version(state->innerRel,
											  &r->heap_tid,
											  snapshot,
											  state->innerSlot);
		if (!found)
			continue;  /* Tuple no longer visible, skip */

		/*
		 * Build the scan tuple using the cached column mappings.
		 * Each column comes from either inner relation or is the target hash.
		 */
		ExecClearTuple(scanSlot);

		for (i = 0; i < state->ncolMappings; i++)
		{
			ColumnMapping *cm = &state->colMappings[i];
			Datum		val;
			bool		isnull;

			if ((Index) cm->varno == state->innerRelid)
			{
				/* Column from inner relation */
				val = slot_getattr(state->innerSlot, cm->varattno, &isnull);
			}
			else if ((Index) cm->varno == state->outerRelid)
			{
				/* Column from outer relation (target hash) */
				val = Int64GetDatum(r->target_hash);
				isnull = false;
			}
			else
			{
				elog(ERROR, "bktree: unexpected varno %d in column mapping", cm->varno);
			}

			scanSlot->tts_values[i] = val;
			scanSlot->tts_isnull[i] = isnull;
		}

		ExecStoreVirtualTuple(scanSlot);
		return scanSlot;
	}
	return NULL;
}

/*
 * Execute the custom scan - directly call our next function.
 */
static TupleTableSlot *
bktree_batch_join_exec(CustomScanState *node)
{
	return bktree_batch_join_next(node);
}

/*
 * End execution.
 */
static void
bktree_batch_join_end(CustomScanState *node)
{
	BktreeBatchJoinState *state = (BktreeBatchJoinState *) node;

	/* End outer plan */
	if (state->outerPlan)
		ExecEndNode(state->outerPlan);

	/* Close relations */
	if (state->index)
		index_close(state->index, AccessShareLock);
	if (state->innerRel)
		table_close(state->innerRel, AccessShareLock);

	/* Delete batch context (frees targets, results) */
	if (state->batchContext)
		MemoryContextDelete(state->batchContext);

	/* Free column mappings (allocated in executor context) */
	if (state->colMappings)
		pfree(state->colMappings);
}

/*
 * Rescan.
 */
static void
bktree_batch_join_rescan(CustomScanState *node)
{
	BktreeBatchJoinState *state = (BktreeBatchJoinState *) node;

	/* Reset batch context - frees both targets and results efficiently */
	MemoryContextReset(state->batchContext);

	/* Reallocate targets array in fresh context */
	state->targets = MemoryContextAlloc(state->batchContext,
										sizeof(int64) * state->max_targets);

	/* Reset state */
	state->ntargets = 0;
	state->results = NULL;
	state->nresults = 0;
	state->curr_result = 0;
	state->search_done = false;

	/* Rescan outer */
	if (state->outerPlan)
		ExecReScan(state->outerPlan);
}

/*
 * EXPLAIN output.
 */
static void
bktree_batch_join_explain(CustomScanState *node,
						  List *ancestors,
						  ExplainState *es)
{
	BktreeBatchJoinState *state = (BktreeBatchJoinState *) node;

	appendStringInfoString(es->str, "Batch BK-tree Join");

	if (es->verbose)
	{
		char *idxname = get_rel_name(state->indexOid);
		if (idxname)
		{
			appendStringInfo(es->str, "\n  Index: %s", idxname);
			pfree(idxname);
		}
	}

	if (es->analyze && state->search_done)
	{
		appendStringInfo(es->str, "\n  Targets: %d, Distance: %ld, Results: %d",
						 state->ntargets, (long) state->distance, state->nresults);
	}
}
