"""
Workspace TTL Management API Routes

Provides endpoints for managing workspace TTL, cleanup, and statistics.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Optional
from datetime import datetime

from lightrag.kg.workspace_ttl import (
    get_ttl_manager,
    WorkspaceMetadata,
    WorkspaceTTLManager,
)
from lightrag.utils import logger
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.api.config import get_env_value

router = APIRouter(prefix="/workspace-ttl", tags=["workspace-ttl"])

# Get API key for authentication
api_key = get_env_value("API_KEY")
auth_dependency = get_combined_auth_dependency(api_key)


@router.post("/cleanup", dependencies=[Depends(auth_dependency)])
async def cleanup_expired_workspaces(
    dry_run: bool = False,
    background_tasks: BackgroundTasks = None,
):
    """
    Clean up expired workspaces to free storage.

    Args:
        dry_run: If True, only return what would be deleted without actually deleting

    Returns:
        Statistics about the cleanup operation
    """
    from lightrag import LightRAG

    manager = get_ttl_manager()

    if not manager.enable_ttl:
        return {
            "status": "disabled",
            "message": "Workspace TTL is disabled. Enable it in .env with ENABLE_WORKSPACE_TTL=true",
            "deleted": 0,
            "freed_mb": 0,
        }

    # Get expired workspaces
    expired_workspaces = await manager.get_expired_workspaces()

    if not expired_workspaces:
        return {
            "status": "success",
            "message": "No expired workspaces found",
            "deleted": 0,
            "freed_mb": 0,
            "workspaces": [],
        }

    # Prepare cleanup results
    cleanup_results = []
    total_freed_mb = 0

    for metadata in expired_workspaces:
        workspace_name = metadata.workspace_name

        try:
            if dry_run:
                # Just report what would be deleted
                cleanup_results.append(
                    {
                        "workspace": workspace_name,
                        "status": "would_delete",
                        "last_accessed": metadata.last_accessed_at.isoformat(),
                        "access_count": metadata.access_count,
                        "document_count": metadata.document_count,
                    }
                )
            else:
                # Actually delete the workspace
                logger.info(f"Deleting expired workspace: {workspace_name}")

                # Delete from LightRAG (this will clean Neo4j and PostgreSQL)
                try:
                    rag = LightRAG(working_dir=f"./workspace_{workspace_name}")
                    # Delete all data from this workspace
                    await rag.delete_all_data()
                    logger.info(f"Successfully deleted workspace data: {workspace_name}")
                except Exception as e:
                    logger.error(f"Failed to delete workspace data for {workspace_name}: {e}")
                    # Continue with marking as expired even if deletion fails
                    pass

                # Mark as expired in TTL manager
                await manager.mark_workspace_expired(workspace_name)

                # Estimate freed space (rough estimate: 8.5 MB per 50 docs)
                estimated_mb = (metadata.document_count / 50) * 8.5
                total_freed_mb += estimated_mb

                cleanup_results.append(
                    {
                        "workspace": workspace_name,
                        "status": "deleted",
                        "last_accessed": metadata.last_accessed_at.isoformat(),
                        "access_count": metadata.access_count,
                        "document_count": metadata.document_count,
                        "freed_mb": round(estimated_mb, 2),
                    }
                )

        except Exception as e:
            logger.error(f"Error cleaning up workspace {workspace_name}: {e}")
            cleanup_results.append(
                {
                    "workspace": workspace_name,
                    "status": "error",
                    "error": str(e),
                }
            )

    return {
        "status": "success",
        "message": f"{'Would delete' if dry_run else 'Deleted'} {len(cleanup_results)} workspace(s)",
        "dry_run": dry_run,
        "deleted": len(cleanup_results),
        "freed_mb": round(total_freed_mb, 2),
        "workspaces": cleanup_results,
    }


@router.get("/statistics", dependencies=[Depends(auth_dependency)])
async def get_ttl_statistics():
    """
    Get statistics about workspace TTL status.

    Returns:
        Dictionary with TTL statistics including active, expired, and expiring workspaces
    """
    manager = get_ttl_manager()
    stats = await manager.get_statistics()

    # Add more detailed info
    all_workspaces = await manager.get_all_workspaces()

    workspaces_by_status = {
        "active": [],
        "expired": [],
        "expiring_soon": [],
    }

    now = datetime.now()

    for ws in all_workspaces:
        workspace_info = {
            "name": ws.workspace_name,
            "last_accessed": ws.last_accessed_at.isoformat(),
            "access_count": ws.access_count,
            "document_count": ws.document_count,
            "expires_at": ws.expires_at.isoformat() if ws.expires_at else None,
        }

        if ws.status == "expired":
            workspaces_by_status["expired"].append(workspace_info)
        elif ws.status == "active" and ws.expires_at:
            time_until_expiry = ws.expires_at - now
            if time_until_expiry.total_seconds() < 6 * 3600:  # 6 hours
                workspace_info["hours_until_expiry"] = round(
                    time_until_expiry.total_seconds() / 3600, 1
                )
                workspaces_by_status["expiring_soon"].append(workspace_info)
            else:
                workspaces_by_status["active"].append(workspace_info)

    stats["workspaces"] = workspaces_by_status

    return stats


@router.get("/workspaces", dependencies=[Depends(auth_dependency)])
async def list_all_workspaces():
    """
    List all tracked workspaces with their metadata.

    Returns:
        List of workspace metadata
    """
    manager = get_ttl_manager()
    workspaces = await manager.get_all_workspaces()

    return {
        "total": len(workspaces),
        "workspaces": [
            {
                "name": ws.workspace_name,
                "created_at": ws.created_at.isoformat(),
                "last_accessed_at": ws.last_accessed_at.isoformat(),
                "access_count": ws.access_count,
                "expires_at": ws.expires_at.isoformat() if ws.expires_at else None,
                "document_count": ws.document_count,
                "status": ws.status,
            }
            for ws in workspaces
        ],
    }


@router.get("/workspace/{workspace_name}", dependencies=[Depends(auth_dependency)])
async def get_workspace_info(workspace_name: str):
    """
    Get TTL information for a specific workspace.

    Args:
        workspace_name: Name of the workspace

    Returns:
        Workspace metadata
    """
    manager = get_ttl_manager()
    metadata = await manager.get_workspace_metadata(workspace_name)

    if not metadata:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_name}' not found in TTL tracking")

    return {
        "name": metadata.workspace_name,
        "created_at": metadata.created_at.isoformat(),
        "last_accessed_at": metadata.last_accessed_at.isoformat(),
        "access_count": metadata.access_count,
        "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
        "document_count": metadata.document_count,
        "status": metadata.status,
    }


@router.post("/workspace/{workspace_name}/extend", dependencies=[Depends(auth_dependency)])
async def extend_workspace_ttl(workspace_name: str):
    """
    Manually extend TTL for a workspace.

    Args:
        workspace_name: Name of the workspace

    Returns:
        Updated workspace metadata
    """
    manager = get_ttl_manager()
    metadata = await manager.access_workspace(workspace_name)

    return {
        "message": f"TTL extended for workspace '{workspace_name}'",
        "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
    }


@router.delete("/workspace/{workspace_name}", dependencies=[Depends(auth_dependency)])
async def delete_workspace_immediately(workspace_name: str):
    """
    Immediately delete a workspace regardless of TTL.

    Args:
        workspace_name: Name of the workspace

    Returns:
        Deletion status
    """
    from lightrag import LightRAG

    manager = get_ttl_manager()

    try:
        # Delete from LightRAG
        rag = LightRAG(working_dir=f"./workspace_{workspace_name}")
        await rag.delete_all_data()

        # Remove from TTL tracking
        await manager.remove_workspace(workspace_name)

        return {
            "status": "success",
            "message": f"Workspace '{workspace_name}' deleted successfully",
        }

    except Exception as e:
        logger.error(f"Failed to delete workspace {workspace_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete workspace: {str(e)}")
