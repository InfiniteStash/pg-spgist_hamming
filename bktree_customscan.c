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
 * Custom scan state for execution
 */
typedef struct BktreeBatchJoinState
{
	CustomScanState css;

	/* Outer scan state */
	PlanState  *outerPlan;
	bool		outer_exhausted;

	/* Batch search state */
	Relation	index;
	int64	   *targets;
	int			ntargets;
	int			max_targets;
	int64		distance;
	ExprState  *distanceExprState;

	/* Results */
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
		elog(DEBUG1, "bktree: clause is not OpExpr (tag=%d)", nodeTag(rinfo->clause));
		return false;
	}

	op = (OpExpr *) rinfo->clause;
	opno = op->opno;

	elog(DEBUG1, "bktree: checking opno %u vs bktree op %u",
		 opno, bktree_containment_op_oid());

	/* Check if it's the <@ operator */
	if (opno != bktree_containment_op_oid())
	{
		elog(DEBUG1, "bktree: operator mismatch");
		return false;
	}

	/* Should have exactly 2 arguments */
	if (list_length(op->args) != 2)
		return false;

	leftarg = linitial(op->args);
	rightarg = lsecond(op->args);

	elog(DEBUG1, "bktree: leftarg type=%d, rightarg type=%d",
		 nodeTag(leftarg), nodeTag(rightarg));

	/*
	 * Left arg should be a Var from innerrel matching the indexed column.
	 */
	if (!IsA(leftarg, Var))
	{
		elog(DEBUG1, "bktree: leftarg is not Var");
		return false;
	}

	indexedVar = (Var *) leftarg;
	if (!bms_is_member(indexedVar->varno, innerrel->relids))
	{
		elog(DEBUG1, "bktree: leftarg varno %d not in innerrel", indexedVar->varno);
		return false;
	}
	if (indexedVar->varattno != indexedAttno)
	{
		elog(DEBUG1, "bktree: leftarg attno %d != indexed attno %d",
			 indexedVar->varattno, indexedAttno);
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

	elog(DEBUG1, "bktree: rightarg after strip type=%d", nodeTag(rightarg));

	/* Strip any RelabelType nodes */
	while (IsA(rightarg, RelabelType))
		rightarg = (Node *) ((RelabelType *) rightarg)->arg;

	elog(DEBUG1, "bktree: rightarg after RelabelType strip type=%d", nodeTag(rightarg));

	/* Check for RowExpr (explicit tuple construction) */
	if (IsA(rightarg, RowExpr))
	{
		elog(DEBUG1, "bktree: rightarg is RowExpr");
		rowexpr = (RowExpr *) rightarg;
		return extract_from_rowexpr(rowexpr, outerrel, outerVar, distanceExpr);
	}

	/* Check for CoerceViaIO or other coercion of a row expr */
	if (IsA(rightarg, CoerceViaIO))
	{
		CoerceViaIO *coerce = (CoerceViaIO *) rightarg;
		elog(DEBUG1, "bktree: rightarg is CoerceViaIO, resulttype=%u, arg type=%d",
			 coerce->resulttype, nodeTag(coerce->arg));
		if (coerce->resulttype == bktree_area_oid && IsA(coerce->arg, RowExpr))
		{
			rowexpr = (RowExpr *) coerce->arg;
			return extract_from_rowexpr(rowexpr, outerrel, outerVar, distanceExpr);
		}
	}

	elog(DEBUG1, "bktree: rightarg pattern not matched");
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
		elog(DEBUG1, "bktree: outer is not UNNEST function scan");
		return false;
	}

	elog(DEBUG1, "bktree: found UNNEST function scan");

	/* Check inner has bktree index */
	indexOid = find_bktree_index_for_rel(innerrel, &indexedAttno);
	if (!OidIsValid(indexOid))
	{
		elog(DEBUG1, "bktree: no bktree index found on inner rel");
		return false;
	}

	elog(DEBUG1, "bktree: found bktree index %u on attno %d", indexOid, indexedAttno);

	info->indexOid = indexOid;
	info->indexedAttno = indexedAttno;
	info->innerRelid = innerrel->relid;

	/* Check join condition matches pattern */
	foreach(lc, restrictlist)
	{
		RestrictInfo *rinfo = lfirst(lc);

		elog(DEBUG1, "bktree: checking restrictinfo, clause type = %d",
			 nodeTag(rinfo->clause));

		if (match_bktree_join_clause(root, rinfo, innerrel, outerrel,
									 indexedAttno, &info->distanceExpr,
									 &info->outerVar))
		{
			info->joinClause = rinfo;
			elog(DEBUG1, "bktree: matched join clause!");
			return true;
		}
	}

	elog(DEBUG1, "bktree: no matching join clause found");
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

	elog(DEBUG1, "bktree: create_bktree_batch_join_path called");

	/* Get cheapest outer path */
	outerpath = outerrel->cheapest_total_path;
	if (outerpath == NULL)
	{
		elog(DEBUG1, "bktree: no cheapest outer path");
		return;
	}

	/* Limit to MAX_BATCH_TARGETS rows */
	if (outerrel->rows > MAX_BATCH_TARGETS)
	{
		elog(DEBUG1, "bktree: too many outer rows: %f > %d",
			 outerrel->rows, MAX_BATCH_TARGETS);
		return;
	}

	elog(DEBUG1, "bktree: outer rows = %f", outerrel->rows);

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

	elog(DEBUG1, "bktree: created path with cost %.2f..%.2f rows=%.0f",
		 pathnode->cpath.path.startup_cost,
		 pathnode->cpath.path.total_cost,
		 pathnode->cpath.path.rows);

	/* Add the path to the joinrel */
	add_path(joinrel, &pathnode->cpath.path);

	elog(DEBUG1, "bktree: path added to joinrel");
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

	elog(DEBUG1, "bktree: plan - entered, tlist length=%d, clauses length=%d",
		 list_length(tlist), list_length(clauses));

	/* Log incoming tlist */
	resno = 0;
	foreach(lc, tlist)
	{
		TargetEntry *tle = lfirst(lc);
		elog(DEBUG1, "bktree: plan - tlist[%d] nodeTag=%d, resno=%d, resname=%s",
			 resno, nodeTag(tle->expr), tle->resno,
			 tle->resname ? tle->resname : "(null)");
		if (IsA(tle->expr, Var))
		{
			Var *v = (Var *) tle->expr;
			elog(DEBUG1, "bktree: plan - tlist[%d] is Var: varno=%d, varattno=%d",
				 resno, v->varno, v->varattno);
		}
		resno++;
	}

	/* Log clauses */
	resno = 0;
	foreach(lc, clauses)
	{
		Node *clause = lfirst(lc);
		elog(DEBUG1, "bktree: plan - clause[%d] nodeTag=%d", resno, nodeTag(clause));
		resno++;
	}

	/* Log distanceExpr */
	elog(DEBUG1, "bktree: plan - distanceExpr nodeTag=%d",
		 path->distanceExpr ? nodeTag(path->distanceExpr) : -1);
	if (path->distanceExpr && IsA(path->distanceExpr, Const))
	{
		Const *c = (Const *) path->distanceExpr;
		elog(DEBUG1, "bktree: plan - distanceExpr is Const, consttype=%u, isnull=%d",
			 c->consttype, c->constisnull);
	}
	else if (path->distanceExpr && IsA(path->distanceExpr, Var))
	{
		Var *v = (Var *) path->distanceExpr;
		elog(DEBUG1, "bktree: plan - distanceExpr is Var: varno=%d, varattno=%d",
			 v->varno, v->varattno);
	}

	/* Log outerVar */
	elog(DEBUG1, "bktree: plan - outerVar: varno=%d, varattno=%d",
		 path->outerVar->varno, path->outerVar->varattno);

	/* Get the outer plan */
	outerPlan = linitial(custom_plans);
	elog(DEBUG1, "bktree: plan - outerPlan nodeTag=%d", nodeTag(outerPlan));

	cscan = makeNode(CustomScan);
	cscan->scan.plan.qual = NIL;  /* We don't have post-scan filters */
	cscan->scan.scanrelid = 0;	  /* Not scanning a single relation */
	cscan->flags = best_path->flags;

	/*
	 * Store custom private data as List of Consts.
	 * We need: indexOid, indexedAttno, innerRelid, outerVar->varattno, distanceExpr, outerRelid
	 * The distanceExpr must be copied since it comes from the planner.
	 */
	cscan->custom_private = list_make1(
		makeConst(OIDOID, -1, InvalidOid, sizeof(Oid),
				  ObjectIdGetDatum(path->indexOid), false, true));
	cscan->custom_private = lappend(cscan->custom_private,
		makeConst(INT2OID, -1, InvalidOid, sizeof(int16),
				  Int16GetDatum(path->indexedAttno), false, true));
	cscan->custom_private = lappend(cscan->custom_private,
		makeConst(INT4OID, -1, InvalidOid, sizeof(int32),
				  Int32GetDatum(path->innerRelid), false, true));
	cscan->custom_private = lappend(cscan->custom_private,
		makeConst(INT2OID, -1, InvalidOid, sizeof(int16),
				  Int16GetDatum(path->outerVar->varattno), false, true));
	cscan->custom_private = lappend(cscan->custom_private,
		copyObject(path->distanceExpr));
	cscan->custom_private = lappend(cscan->custom_private,
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

		elog(DEBUG1, "bktree: plan - scan_tlist[%d] = %s",
			 resno, IsA(tle->expr, Var) ? "Var" : "other");
		if (IsA(tle->expr, Var))
		{
			Var *var = (Var *) tle->expr;
			elog(DEBUG1, "bktree: plan - Var varno=%d, varattno=%d",
				 var->varno, var->varattno);
		}

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

	/* Log what we produced */
	elog(DEBUG1, "bktree: plan - plan.targetlist length=%d", list_length(cscan->scan.plan.targetlist));
	elog(DEBUG1, "bktree: plan - scan_tlist length=%d", list_length(scan_tlist));
	resno = 0;
	foreach(lc, scan_tlist)
	{
		TargetEntry *tle = lfirst(lc);
		elog(DEBUG1, "bktree: plan - scan_tlist[%d] nodeTag=%d", resno, nodeTag(tle->expr));
		resno++;
	}

	elog(DEBUG1, "bktree: plan - custom_private length=%d", list_length(cscan->custom_private));
	resno = 0;
	foreach(lc, cscan->custom_private)
	{
		Node *item = lfirst(lc);
		elog(DEBUG1, "bktree: plan - custom_private[%d] nodeTag=%d", resno, nodeTag(item));
		if (IsA(item, Var))
		{
			Var *v = (Var *) item;
			elog(DEBUG1, "bktree: plan - custom_private[%d] is Var! varno=%d, varattno=%d",
				 resno, v->varno, v->varattno);
		}
		resno++;
	}

	elog(DEBUG1, "bktree: plan - complete, returning CustomScan");
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
	Plan	   *outerPlan;
	Const	   *indexOidConst;
	Const	   *indexedAttnoConst;
	Const	   *innerRelidConst;
	Const	   *outerVarAttnoConst;
	Const	   *outerRelidConst;
	Expr	   *distanceExpr;
	List	   *tlist;
	int			ncolumns;
	int			i;
	int			privIdx;

	elog(DEBUG1, "bktree: begin - extracting private data, list length = %d",
		 list_length(cscan->custom_private));

	/* Extract fixed private data */
	indexOidConst = linitial(cscan->custom_private);
	indexedAttnoConst = lsecond(cscan->custom_private);
	innerRelidConst = lthird(cscan->custom_private);
	outerVarAttnoConst = lfourth(cscan->custom_private);
	distanceExpr = (Expr *) list_nth(cscan->custom_private, 4);
	outerRelidConst = (Const *) list_nth(cscan->custom_private, 5);

	elog(DEBUG1, "bktree: begin - distanceExpr type = %d",
		 distanceExpr ? nodeTag(distanceExpr) : -1);

	state->indexOid = DatumGetObjectId(indexOidConst->constvalue);
	state->indexedAttno = DatumGetInt16(indexedAttnoConst->constvalue);
	state->outerVarAttno = DatumGetInt16(outerVarAttnoConst->constvalue);
	state->innerRelid = DatumGetInt32(innerRelidConst->constvalue);
	state->outerRelid = DatumGetInt32(outerRelidConst->constvalue);

	elog(DEBUG1, "bktree: begin - indexOid=%u, indexedAttno=%d, outerVarAttno=%d, innerRelid=%d, outerRelid=%d",
		 state->indexOid, state->indexedAttno, state->outerVarAttno,
		 state->innerRelid, state->outerRelid);

	/*
	 * Extract column mappings from custom_private.
	 * After the first 6 entries, we have pairs of (varno, varattno) for each column.
	 */
	tlist = cscan->scan.plan.targetlist;
	ncolumns = list_length(tlist);
	state->ncolMappings = ncolumns;
	state->colMappings = palloc(sizeof(ColumnMapping) * ncolumns);

	privIdx = 6;  /* Start after fixed entries */
	for (i = 0; i < ncolumns; i++)
	{
		Const *varnoConst = (Const *) list_nth(cscan->custom_private, privIdx);
		Const *varattnoConst = (Const *) list_nth(cscan->custom_private, privIdx + 1);

		state->colMappings[i].varno = DatumGetInt32(varnoConst->constvalue);
		state->colMappings[i].varattno = DatumGetInt16(varattnoConst->constvalue);

		elog(DEBUG1, "bktree: begin - column %d: varno=%d, varattno=%d",
			 i, state->colMappings[i].varno, state->colMappings[i].varattno);

		privIdx += 2;
	}

	/* Compile distance expression */
	elog(DEBUG1, "bktree: begin - about to ExecInitExpr");
	state->distanceExprState = ExecInitExpr(distanceExpr, &node->ss.ps);
	elog(DEBUG1, "bktree: begin - ExecInitExpr done");

	/* Open index */
	elog(DEBUG1, "bktree: begin - opening index");
	state->index = index_open(state->indexOid, AccessShareLock);

	/* Open inner relation (for heap fetches) */
	elog(DEBUG1, "bktree: begin - opening inner relation");
	{
		RangeTblEntry *rte = exec_rt_fetch(state->innerRelid, estate);
		state->innerRel = table_open(rte->relid, AccessShareLock);
		/* Create a slot suitable for this relation's tuples */
		state->innerSlot = table_slot_create(state->innerRel,
											 &estate->es_tupleTable);
	}
	elog(DEBUG1, "bktree: begin - inner relation opened");

	/* Initialize outer plan */
	elog(DEBUG1, "bktree: begin - initializing outer plan");
	outerPlan = linitial(cscan->custom_plans);
	state->outerPlan = ExecInitNode(outerPlan, estate, eflags);
	state->outer_exhausted = false;
	elog(DEBUG1, "bktree: begin - outer plan initialized");

	/* Initialize batch state */
	state->max_targets = MAX_BATCH_TARGETS;
	state->targets = palloc(sizeof(int64) * state->max_targets);
	state->ntargets = 0;
	state->distance = 0;

	state->results = NULL;
	state->nresults = 0;
	state->curr_result = 0;
	state->search_done = false;

	/* Log projection state */
	elog(DEBUG1, "bktree: begin - ps_ProjInfo=%p", node->ss.ps.ps_ProjInfo);
	if (node->ss.ps.ps_ProjInfo)
	{
		ProjectionInfo *proj = node->ss.ps.ps_ProjInfo;
		elog(DEBUG1, "bktree: begin - projection pi_exprContext=%p",
			 proj->pi_exprContext);
	}

	elog(DEBUG1, "bktree: begin - complete");
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

		if (state->ntargets >= state->max_targets)
		{
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("too many targets for batch join, maximum is %d",
							MAX_BATCH_TARGETS)));
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

	state->outer_exhausted = true;
}

/*
 * Execute batch search.
 */
static void
execute_batch_search(BktreeBatchJoinState *state)
{
	BatchSearchState bss;

	if (state->ntargets == 0)
	{
		state->results = NULL;
		state->nresults = 0;
		return;
	}

	/* Initialize batch search state */
	memset(&bss, 0, sizeof(bss));
	bss.targets = state->targets;
	bss.ntargets = state->ntargets;
	bss.distance = state->distance;
	bss.index = state->index;
	bss.max_results = 1024;
	bss.results = palloc(sizeof(BatchResult) * bss.max_results);
	bss.nresults = 0;

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

	elog(DEBUG1, "bktree: next called, search_done=%d", state->search_done);

	/* First call: collect targets and execute batch search */
	if (!state->search_done)
	{
		elog(DEBUG1, "bktree: next - collecting targets");
		collect_targets(state);
		elog(DEBUG1, "bktree: next - collected %d targets", state->ntargets);
		execute_batch_search(state);
		elog(DEBUG1, "bktree: next - batch search done, %d results", state->nresults);
		state->search_done = true;
		state->curr_result = 0;
	}

	snapshot = GetActiveSnapshot();

	elog(DEBUG1, "bktree: next - at result %d of %d",
		 state->curr_result, state->nresults);

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

		elog(DEBUG1, "bktree: next - returning tuple for result %d",
			 state->curr_result - 1);
		return scanSlot;
	}

	elog(DEBUG1, "bktree: next - no more results");
	return NULL;
}

/*
 * Recheck method for ExecScan - we don't need rechecks.
 */
static bool
bktree_batch_join_recheck(CustomScanState *node, TupleTableSlot *slot)
{
	/* No recheck needed - we already verified visibility */
	return true;
}

/*
 * Execute the custom scan - directly call our next function.
 */
static TupleTableSlot *
bktree_batch_join_exec(CustomScanState *node)
{
	TupleTableSlot *slot;

	elog(DEBUG1, "bktree: exec called");

	slot = bktree_batch_join_next(node);

	if (slot == NULL)
	{
		/* Return empty result slot instead of NULL */
		elog(DEBUG1, "bktree: exec - returning cleared result slot");
		return ExecClearTuple(node->ss.ps.ps_ResultTupleSlot);
	}

	elog(DEBUG1, "bktree: exec returning tuple");
	return slot;
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

	/* Free allocated memory */
	if (state->targets)
		pfree(state->targets);
	if (state->results)
		pfree(state->results);
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

	/* Reset state */
	state->ntargets = 0;
	state->nresults = 0;
	state->curr_result = 0;
	state->search_done = false;
	state->outer_exhausted = false;

	/* Free previous results */
	if (state->results)
	{
		pfree(state->results);
		state->results = NULL;
	}

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
