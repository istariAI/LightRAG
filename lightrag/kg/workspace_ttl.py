"""
Workspace TTL (Time To Live) Management Module

This module provides functionality to track workspace access times and automatically
clean up expired workspaces to save storage space.
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass, field
import json

from lightrag.utils import logger
from lightrag.constants import (
    DEFAULT_ENABLE_WORKSPACE_TTL,
    DEFAULT_WORKSPACE_TTL_HOURS,
    DEFAULT_WORKSPACE_CLEANUP_INTERVAL_HOURS,
)


@dataclass
class WorkspaceMetadata:
    """Metadata for workspace TTL tracking"""

    workspace_name: str
    created_at: datetime
    last_accessed_at: datetime
    access_count: int = 0
    expires_at: Optional[datetime] = None
    document_count: int = 0
    status: str = "active"  # active, expired, processing

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "workspace_name": self.workspace_name,
            "created_at": self.created_at.isoformat(),
            "last_accessed_at": self.last_accessed_at.isoformat(),
            "access_count": self.access_count,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "document_count": self.document_count,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkspaceMetadata":
        """Create from dictionary"""
        return cls(
            workspace_name=data["workspace_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed_at=datetime.fromisoformat(data["last_accessed_at"]),
            access_count=data.get("access_count", 0),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            document_count=data.get("document_count", 0),
            status=data.get("status", "active"),
        )


class WorkspaceTTLManager:
    """
    Manages workspace TTL tracking and cleanup.

    Stores workspace metadata in a JSON file for persistence across server restarts.
    """

    def __init__(
        self,
        metadata_file: str = "workspace_ttl_metadata.json",
        enable_ttl: bool = None,
        ttl_hours: int = None,
    ):
        """
        Initialize the TTL manager.

        Args:
            metadata_file: Path to JSON file storing workspace metadata
            enable_ttl: Enable automatic workspace expiration (default from env)
            ttl_hours: TTL duration in hours (default from env)
        """
        self.metadata_file = metadata_file

        # Load configuration from environment or use defaults
        self.enable_ttl = (
            enable_ttl
            if enable_ttl is not None
            else os.getenv("ENABLE_WORKSPACE_TTL", str(DEFAULT_ENABLE_WORKSPACE_TTL)).lower() == "true"
        )
        self.ttl_hours = (
            ttl_hours if ttl_hours is not None else int(os.getenv("WORKSPACE_TTL_HOURS", DEFAULT_WORKSPACE_TTL_HOURS))
        )

        # In-memory cache of workspace metadata
        self._metadata_cache: Dict[str, WorkspaceMetadata] = {}
        self._lock = asyncio.Lock()

        # Load existing metadata
        self._load_metadata()

        logger.info(
            f"WorkspaceTTLManager initialized: enabled={self.enable_ttl}, "
            f"ttl_hours={self.ttl_hours}"
        )

    def _load_metadata(self):
        """Load workspace metadata from JSON file"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r") as f:
                    data = json.load(f)
                    for workspace_name, metadata_dict in data.items():
                        self._metadata_cache[workspace_name] = WorkspaceMetadata.from_dict(metadata_dict)
                logger.info(f"Loaded TTL metadata for {len(self._metadata_cache)} workspaces")
            except Exception as e:
                logger.error(f"Failed to load workspace TTL metadata: {e}")
                self._metadata_cache = {}
        else:
            logger.info("No existing workspace TTL metadata found, starting fresh")

    def _save_metadata(self):
        """Save workspace metadata to JSON file"""
        try:
            data = {name: metadata.to_dict() for name, metadata in self._metadata_cache.items()}
            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save workspace TTL metadata: {e}")

    def _calculate_expires_at(self, last_accessed: datetime) -> datetime:
        """Calculate expiration time based on last access"""
        return last_accessed + timedelta(hours=self.ttl_hours)

    async def register_workspace(
        self, workspace_name: str, document_count: int = 0
    ) -> WorkspaceMetadata:
        """
        Register a new workspace or update existing one.

        Args:
            workspace_name: Name of the workspace
            document_count: Number of documents in the workspace

        Returns:
            WorkspaceMetadata for the workspace
        """
        async with self._lock:
            now = datetime.now()

            if workspace_name in self._metadata_cache:
                # Update existing workspace
                metadata = self._metadata_cache[workspace_name]
                metadata.document_count = document_count
                metadata.last_accessed_at = now
                metadata.status = "active"
            else:
                # Create new workspace metadata
                metadata = WorkspaceMetadata(
                    workspace_name=workspace_name,
                    created_at=now,
                    last_accessed_at=now,
                    document_count=document_count,
                    status="active",
                )
                self._metadata_cache[workspace_name] = metadata

            # Update expiration time
            if self.enable_ttl:
                metadata.expires_at = self._calculate_expires_at(now)
            else:
                metadata.expires_at = None

            self._save_metadata()
            logger.info(
                f"Registered workspace '{workspace_name}' (expires: {metadata.expires_at})"
            )

            return metadata

    async def access_workspace(self, workspace_name: str) -> WorkspaceMetadata:
        """
        Record workspace access and extend TTL.

        Args:
            workspace_name: Name of the workspace

        Returns:
            Updated WorkspaceMetadata
        """
        async with self._lock:
            now = datetime.now()

            if workspace_name not in self._metadata_cache:
                # Workspace not tracked yet, register it
                return await self.register_workspace(workspace_name)

            metadata = self._metadata_cache[workspace_name]
            metadata.last_accessed_at = now
            metadata.access_count += 1

            # Extend TTL on access
            if self.enable_ttl:
                metadata.expires_at = self._calculate_expires_at(now)
                metadata.status = "active"

            self._save_metadata()

            logger.debug(
                f"Workspace '{workspace_name}' accessed (count: {metadata.access_count}, "
                f"expires: {metadata.expires_at})"
            )

            return metadata

    async def get_expired_workspaces(self) -> List[WorkspaceMetadata]:
        """
        Get list of expired workspaces ready for cleanup.

        Returns:
            List of expired WorkspaceMetadata objects
        """
        if not self.enable_ttl:
            return []

        async with self._lock:
            now = datetime.now()

            expired = []
            for metadata in self._metadata_cache.values():
                if metadata.expires_at and metadata.status == "active":
                    # Check if expired (no grace period)
                    if now > metadata.expires_at:
                        expired.append(metadata)

            return expired

    async def mark_workspace_expired(self, workspace_name: str):
        """
        Mark a workspace as expired.

        Args:
            workspace_name: Name of the workspace
        """
        async with self._lock:
            if workspace_name in self._metadata_cache:
                self._metadata_cache[workspace_name].status = "expired"
                self._save_metadata()
                logger.info(f"Marked workspace '{workspace_name}' as expired")

    async def remove_workspace(self, workspace_name: str):
        """
        Remove workspace from TTL tracking.

        Args:
            workspace_name: Name of the workspace
        """
        async with self._lock:
            if workspace_name in self._metadata_cache:
                del self._metadata_cache[workspace_name]
                self._save_metadata()
                logger.info(f"Removed workspace '{workspace_name}' from TTL tracking")

    async def get_workspace_metadata(self, workspace_name: str) -> Optional[WorkspaceMetadata]:
        """
        Get metadata for a specific workspace.

        Args:
            workspace_name: Name of the workspace

        Returns:
            WorkspaceMetadata or None if not found
        """
        async with self._lock:
            return self._metadata_cache.get(workspace_name)

    async def get_all_workspaces(self) -> List[WorkspaceMetadata]:
        """
        Get metadata for all tracked workspaces.

        Returns:
            List of all WorkspaceMetadata objects
        """
        async with self._lock:
            return list(self._metadata_cache.values())

    async def get_statistics(self) -> dict:
        """
        Get statistics about workspace TTL status.

        Returns:
            Dictionary with TTL statistics
        """
        async with self._lock:
            now = datetime.now()

            stats = {
                "total_workspaces": len(self._metadata_cache),
                "active_workspaces": 0,
                "expired_workspaces": 0,
                "expiring_soon": 0,  # Within 1 hour
                "ttl_enabled": self.enable_ttl,
                "ttl_hours": self.ttl_hours,
            }

            for metadata in self._metadata_cache.values():
                if metadata.status == "active":
                    stats["active_workspaces"] += 1

                    if self.enable_ttl and metadata.expires_at:
                        time_until_expiry = metadata.expires_at - now
                        if time_until_expiry.total_seconds() < 3600:  # 1 hour
                            stats["expiring_soon"] += 1
                elif metadata.status == "expired":
                    stats["expired_workspaces"] += 1

            return stats


# Global TTL manager instance
_ttl_manager: Optional[WorkspaceTTLManager] = None


def get_ttl_manager() -> WorkspaceTTLManager:
    """Get or create the global TTL manager instance"""
    global _ttl_manager
    if _ttl_manager is None:
        _ttl_manager = WorkspaceTTLManager()
    return _ttl_manager


async def track_workspace_access(workspace_name: str) -> WorkspaceMetadata:
    """
    Convenience function to track workspace access.

    Args:
        workspace_name: Name of the workspace

    Returns:
        Updated WorkspaceMetadata
    """
    manager = get_ttl_manager()
    return await manager.access_workspace(workspace_name)
