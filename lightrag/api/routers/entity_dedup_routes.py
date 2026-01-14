"""
Entity Deduplication API Routes

Provides endpoints for detecting and merging duplicate entities.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from typing import List, Optional
from pydantic import BaseModel, Field

from lightrag.utils import logger
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.api.config import get_env_value

router = APIRouter(prefix="/entity-dedup", tags=["entity-deduplication"])

# Get API key for authentication (None if not set)
api_key = get_env_value("API_KEY", None)
auth_dependency = get_combined_auth_dependency(api_key)


class DeduplicationRequest(BaseModel):
    """Request model for entity deduplication"""

    entity_names: Optional[List[str]] = Field(
        None,
        description="Optional list of specific entities to check (default: all entities)",
    )
    dry_run: bool = Field(
        False,
        description="If true, only report what would be merged without actually merging",
    )
    similarity_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Similarity confidence threshold (0.0-1.0, default: 0.85)",
    )
    workspace: Optional[str] = Field(
        None,
        description="Workspace name (default: default workspace)",
    )


class MergeEntitiesRequest(BaseModel):
    """Request model for manual entity merging"""

    source_entity: str = Field(
        ..., description="Source entity name (will be merged and removed)"
    )
    target_entity: str = Field(
        ..., description="Target entity name (will be updated with merged data)"
    )
    workspace: Optional[str] = Field(
        None,
        description="Workspace name (default: default workspace)",
    )


@router.post("/deduplicate", dependencies=[Depends(auth_dependency)])
async def deduplicate_entities(
    request: DeduplicationRequest,
    background_tasks: BackgroundTasks = None,
):
    """
    Automatically detect and merge duplicate entities using LLM-based similarity detection.

    This endpoint:
    1. Finds candidate entity pairs with similar normalized names
    2. Uses LLM to verify if candidates are actually the same entity
    3. Merges verified duplicates while preserving all information

    Quality guarantees:
    - Only merges entities above similarity threshold (default: 0.85)
    - Preserves all descriptions, source_ids, and verification evidence
    - Uses LLM reasoning to avoid false positives
    - Supports dry-run mode to preview changes

    Args:
        request: Deduplication request with optional entity filter and settings

    Returns:
        Deduplication statistics and merge details
    """
    from lightrag import LightRAG
    from lightrag.kg.entity_deduplication import EntityDeduplicationManager

    try:
        # Get workspace
        workspace_name = request.workspace or "default"
        working_dir = f"./workspace_{workspace_name}"

        # Initialize LightRAG
        rag = LightRAG(working_dir=working_dir)

        # Get configuration
        similarity_threshold = request.similarity_threshold or float(
            get_env_value("ENTITY_DEDUP_THRESHOLD", 0.85)
        )
        batch_size = int(get_env_value("ENTITY_DEDUP_BATCH_SIZE", 50))

        # Create deduplication manager
        dedup_manager = EntityDeduplicationManager(
            knowledge_graph_inst=rag.chunk_entity_relation_graph,
            entities_vdb=rag.entities_vdb,
            chunks_vdb=rag.chunks_vdb,
            llm_model_func=rag.llm_model_func,
            similarity_threshold=similarity_threshold,
            batch_size=batch_size,
        )

        # Run deduplication
        logger.info(
            f"Starting entity deduplication for workspace '{workspace_name}' "
            f"(dry_run={request.dry_run}, threshold={similarity_threshold})"
        )

        result = await dedup_manager.deduplicate_entities(
            entity_names=request.entity_names,
            dry_run=request.dry_run,
        )

        return {
            "status": "success",
            "workspace": workspace_name,
            "similarity_threshold": similarity_threshold,
            **result,
        }

    except Exception as e:
        logger.error(f"Entity deduplication failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Entity deduplication failed: {str(e)}"
        )


@router.post("/merge", dependencies=[Depends(auth_dependency)])
async def merge_entities_manual(request: MergeEntitiesRequest):
    """
    Manually merge two entities.

    This endpoint forces a merge between two specific entities without
    LLM verification. Use this when you're certain two entities should be merged.

    The merge process:
    1. Combines all descriptions (deduplicated)
    2. Merges source_ids and file_paths
    3. Redirects all relationships to target entity
    4. Deletes source entity

    Args:
        request: Merge request with source and target entity names

    Returns:
        Merge result details
    """
    from lightrag import LightRAG

    try:
        # Get workspace
        workspace_name = request.workspace or "default"
        working_dir = f"./workspace_{workspace_name}"

        # Initialize LightRAG
        rag = LightRAG(working_dir=working_dir)

        # Use the built-in merge_entities method
        logger.info(
            f"Manually merging entities: '{request.source_entity}' → '{request.target_entity}' "
            f"in workspace '{workspace_name}'"
        )

        # Perform merge
        await rag.amerge_entities(
            entity_names=[request.source_entity],
            target_entity_name=request.target_entity,
        )

        return {
            "status": "success",
            "message": f"Successfully merged '{request.source_entity}' into '{request.target_entity}'",
            "workspace": workspace_name,
            "source_entity": request.source_entity,
            "target_entity": request.target_entity,
        }

    except Exception as e:
        logger.error(f"Manual entity merge failed: {e}")
        raise HTTPException(status_code=500, detail=f"Entity merge failed: {str(e)}")


@router.get("/candidates", dependencies=[Depends(auth_dependency)])
async def find_duplicate_candidates(
    workspace: Optional[str] = Query(None, description="Workspace name"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of candidates to return"),
):
    """
    Find candidate duplicate entities without performing merges.

    This endpoint identifies entities with similar names that might be duplicates,
    but does not verify or merge them. Use this for analysis and review.

    Returns:
        List of candidate entity pairs with normalized names
    """
    from lightrag import LightRAG
    from lightrag.kg.entity_deduplication import EntityDeduplicationManager

    try:
        # Get workspace
        workspace_name = workspace or "default"
        working_dir = f"./workspace_{workspace_name}"

        # Initialize LightRAG
        rag = LightRAG(working_dir=working_dir)

        # Create deduplication manager
        dedup_manager = EntityDeduplicationManager(
            knowledge_graph_inst=rag.chunk_entity_relation_graph,
            entities_vdb=rag.entities_vdb,
            chunks_vdb=rag.chunks_vdb,
            llm_model_func=rag.llm_model_func,
            similarity_threshold=0.85,
            batch_size=50,
        )

        # Get all entities
        entities = await rag.chunk_entity_relation_graph.get_all_nodes()

        # Find candidates
        candidate_pairs = dedup_manager._find_candidate_pairs(entities)

        # Limit results
        candidate_pairs = candidate_pairs[:limit]

        # Add normalized names for each candidate
        candidates_with_info = []
        for entity1, entity2 in candidate_pairs:
            candidates_with_info.append({
                "entity1": entity1,
                "entity2": entity2,
                "normalized1": dedup_manager._normalize_entity_name(entity1),
                "normalized2": dedup_manager._normalize_entity_name(entity2),
                "entity1_type": entities.get(entity1, {}).get("entity_type", "unknown"),
                "entity2_type": entities.get(entity2, {}).get("entity_type", "unknown"),
            })

        return {
            "status": "success",
            "workspace": workspace_name,
            "total_entities": len(entities),
            "candidates_found": len(candidate_pairs),
            "candidates_returned": len(candidates_with_info),
            "candidates": candidates_with_info,
        }

    except Exception as e:
        logger.error(f"Failed to find duplicate candidates: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to find candidates: {str(e)}"
        )


@router.get("/statistics", dependencies=[Depends(auth_dependency)])
async def get_deduplication_statistics(
    workspace: Optional[str] = Query(None, description="Workspace name"),
):
    """
    Get entity deduplication statistics for a workspace.

    Returns information about:
    - Total number of entities
    - Number of entities with similar names
    - Entity type distribution
    - Potential duplicate candidates

    Returns:
        Deduplication statistics
    """
    from lightrag import LightRAG
    from lightrag.kg.entity_deduplication import EntityDeduplicationManager
    from collections import Counter

    try:
        # Get workspace
        workspace_name = workspace or "default"
        working_dir = f"./workspace_{workspace_name}"

        # Initialize LightRAG
        rag = LightRAG(working_dir=working_dir)

        # Get all entities
        entities = await rag.chunk_entity_relation_graph.get_all_nodes()

        # Create deduplication manager for candidate finding
        dedup_manager = EntityDeduplicationManager(
            knowledge_graph_inst=rag.chunk_entity_relation_graph,
            entities_vdb=rag.entities_vdb,
            chunks_vdb=rag.chunks_vdb,
            llm_model_func=rag.llm_model_func,
            similarity_threshold=0.85,
            batch_size=50,
        )

        # Find candidates
        candidate_pairs = dedup_manager._find_candidate_pairs(entities)

        # Count entity types
        entity_types = Counter()
        for entity_data in entities.values():
            entity_type = entity_data.get("entity_type", "unknown")
            entity_types[entity_type] += 1

        # Count normalized name groups
        from collections import defaultdict
        normalized_groups = defaultdict(list)
        for entity_name in entities.keys():
            normalized = dedup_manager._normalize_entity_name(entity_name)
            normalized_groups[normalized].append(entity_name)

        # Count groups with duplicates
        duplicate_groups = {
            normalized: names
            for normalized, names in normalized_groups.items()
            if len(names) > 1
        }

        return {
            "status": "success",
            "workspace": workspace_name,
            "total_entities": len(entities),
            "entity_types": dict(entity_types),
            "candidate_pairs": len(candidate_pairs),
            "normalized_groups": len(normalized_groups),
            "duplicate_candidate_groups": len(duplicate_groups),
            "largest_duplicate_group": max((len(names) for names in duplicate_groups.values()), default=0),
        }

    except Exception as e:
        logger.error(f"Failed to get deduplication statistics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get statistics: {str(e)}"
        )
